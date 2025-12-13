export interface Env {
	IMAGES_BUCKET: R2Bucket;
}

const CORS_HEADERS = {
	'Access-Control-Allow-Origin': '*',
	'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
	'Access-Control-Allow-Headers': 'Content-Type',
};

async function computeHash(data: ArrayBuffer): Promise<string> {
	const hashBuffer = await crypto.subtle.digest('SHA-256', data);
	const hashArray = new Uint8Array(hashBuffer).slice(0, 16);
	const base64 = btoa(String.fromCharCode(...hashArray));
	return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function hashFromUrl(url: URL): string | null {
	const match = url.pathname.match(/^\/image\/([A-Za-z0-9_-]{22})$/);
	return match ? match[1] : null;
}

async function handleUpload(request: Request, env: Env): Promise<Response> {
	if (request.method !== 'POST') {
		return new Response('Method not allowed', { status: 405, headers: CORS_HEADERS });
	}

	const contentType = request.headers.get('Content-Type') || '';
	if (!contentType.startsWith('image/')) {
		return new Response('Content-Type must be an image type', { status: 400, headers: CORS_HEADERS });
	}

	const body = await request.arrayBuffer();
	if (body.byteLength === 0) {
		return new Response('Empty body', { status: 400, headers: CORS_HEADERS });
	}

	if (body.byteLength > 10 * 1024 * 1024) {
		return new Response('Image too large (max 10MB)', { status: 413, headers: CORS_HEADERS });
	}

	const hash = await computeHash(body);

	const existing = await env.IMAGES_BUCKET.head(hash);
	if (!existing) {
		await env.IMAGES_BUCKET.put(hash, body, {
			httpMetadata: { contentType },
		});
	}

	return new Response(JSON.stringify({ hash }), {
		status: 200,
		headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
	});
}

async function handleGet(hash: string, env: Env): Promise<Response> {
	const object = await env.IMAGES_BUCKET.get(hash);

	if (!object) {
		return new Response('Image not found', { status: 404, headers: CORS_HEADERS });
	}

	const headers = new Headers(CORS_HEADERS);
	headers.set('Content-Type', object.httpMetadata?.contentType || 'image/png');
	headers.set('Cache-Control', 'public, max-age=31536000, immutable');
	headers.set('ETag', `"${hash}"`);

	return new Response(object.body, { headers });
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);

		if (request.method === 'OPTIONS') {
			return new Response(null, { headers: CORS_HEADERS });
		}

		if (url.pathname === '/upload') {
			return handleUpload(request, env);
		}

		const hash = hashFromUrl(url);
		if (hash && request.method === 'GET') {
			return handleGet(hash, env);
		}

		return new Response('Not found', { status: 404, headers: CORS_HEADERS });
	},
} satisfies ExportedHandler<Env>;
