.PHONY: dev build clean

dev:
	npm install
	npm run dev

build:
	npm install
	npm run build

clean:
	rm -rf node_modules dist
