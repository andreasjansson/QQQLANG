import {
  FnContext,
  Image,
  createSolidImage,
  getPrevImage,
  getPixel,
  setPixel,
  cloneImage,
  hexToRgb,
  rgbToHsl,
  hslToRgb,
  initWebGL,
  createShaderProgram,
  getOldImage,
  createPlaceholderImage,
  emeraldReady,
  bgRemovalReady,
} from "./helpers.js";

function blend(ctx: FnContext, old: Image, modeName: string): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [br, bg, bb] = getPixel(old, x, y);
      const [tr, tg, tb] = getPixel(prev, x, y);

      let r: number, g: number, b: number;

      switch (modeName) {
        case "multiply":
          r = (br * tr) / 255;
          g = (bg * tg) / 255;
          b = (bb * tb) / 255;
          break;

        case "screen":
          r = 255 - ((255 - br) * (255 - tr)) / 255;
          g = 255 - ((255 - bg) * (255 - tg)) / 255;
          b = 255 - ((255 - bb) * (255 - tb)) / 255;
          break;

        case "overlay":
          r =
            br < 128
              ? (2 * br * tr) / 255
              : 255 - (2 * (255 - br) * (255 - tr)) / 255;
          g =
            bg < 128
              ? (2 * bg * tg) / 255
              : 255 - (2 * (255 - bg) * (255 - tg)) / 255;
          b =
            bb < 128
              ? (2 * bb * tb) / 255
              : 255 - (2 * (255 - bb) * (255 - tb)) / 255;
          break;

        case "darken":
          r = Math.min(br, tr);
          g = Math.min(bg, tg);
          b = Math.min(bb, tb);
          break;

        case "lighten":
          r = Math.max(br, tr);
          g = Math.max(bg, tg);
          b = Math.max(bb, tb);
          break;

        case "dodge":
          r = tr === 255 ? 255 : Math.min(255, (br * 255) / (255 - tr));
          g = tg === 255 ? 255 : Math.min(255, (bg * 255) / (255 - tg));
          b = tb === 255 ? 255 : Math.min(255, (bb * 255) / (255 - tb));
          break;

        case "burn":
          r = tr === 0 ? 0 : Math.max(0, 255 - ((255 - br) * 255) / tr);
          g = tg === 0 ? 0 : Math.max(0, 255 - ((255 - bg) * 255) / tg);
          b = tb === 0 ? 0 : Math.max(0, 255 - ((255 - bb) * 255) / tb);
          break;

        case "hardlight":
          r =
            tr < 128
              ? (2 * br * tr) / 255
              : 255 - (2 * (255 - br) * (255 - tr)) / 255;
          g =
            tg < 128
              ? (2 * bg * tg) / 255
              : 255 - (2 * (255 - bg) * (255 - tg)) / 255;
          b =
            tb < 128
              ? (2 * bb * tb) / 255
              : 255 - (2 * (255 - bb) * (255 - tb)) / 255;
          break;

        case "softlight": {
          const softLight = (b: number, t: number) => {
            const tb = t / 255,
              bb = b / 255;
            return (
              (tb < 0.5
                ? bb - (1 - 2 * tb) * bb * (1 - bb)
                : bb +
                  (2 * tb - 1) *
                    (bb < 0.25
                      ? ((16 * bb - 12) * bb + 4) * bb
                      : Math.sqrt(bb) - bb)) * 255
            );
          };
          r = softLight(br, tr);
          g = softLight(bg, tg);
          b = softLight(bb, tb);
          break;
        }

        case "difference":
          r = Math.abs(br - tr);
          g = Math.abs(bg - tg);
          b = Math.abs(bb - tb);
          break;

        case "exclusion":
          r = br + tr - (2 * br * tr) / 255;
          g = bg + tg - (2 * bg * tg) / 255;
          b = bb + tb - (2 * bb * tb) / 255;
          break;

        case "add":
          r = Math.min(255, br + tr);
          g = Math.min(255, bg + tg);
          b = Math.min(255, bb + tb);
          break;

        case "subtract":
          r = Math.max(0, br - tr);
          g = Math.max(0, bg - tg);
          b = Math.max(0, bb - tb);
          break;

        case "xor":
          r = br ^ tr;
          g = bg ^ tg;
          b = bb ^ tb;
          break;

        case "and":
          r = br & tr;
          g = bg & tg;
          b = bb & tb;
          break;

        case "or":
          r = br | tr;
          g = bg | tg;
          b = bb | tb;
          break;

        case "nand":
          r = 255 - (br & tr);
          g = 255 - (bg & tg);
          b = 255 - (bb & tb);
          break;

        case "nor":
          r = 255 - (br | tr);
          g = 255 - (bg | tg);
          b = 255 - (bb | tb);
          break;

        case "xnor":
          r = 255 - (br ^ tr);
          g = 255 - (bg ^ tg);
          b = 255 - (bb ^ tb);
          break;

        case "average":
          r = (br + tr) / 2;
          g = (bg + tg) / 2;
          b = (bb + tb) / 2;
          break;

        case "divide":
          r = tr === 0 ? 255 : Math.min(255, (br * 255) / tr);
          g = tg === 0 ? 255 : Math.min(255, (bg * 255) / tg);
          b = tb === 0 ? 255 : Math.min(255, (bb * 255) / tb);
          break;

        case "grain-extract":
          r = Math.max(0, Math.min(255, br - tr + 128));
          g = Math.max(0, Math.min(255, bg - tg + 128));
          b = Math.max(0, Math.min(255, bb - tb + 128));
          break;

        case "grain-merge":
          r = Math.max(0, Math.min(255, br + tr - 128));
          g = Math.max(0, Math.min(255, bg + tg - 128));
          b = Math.max(0, Math.min(255, bb + tb - 128));
          break;

        case "vivid":
          r =
            tr < 128
              ? tr === 0
                ? 0
                : Math.max(0, 255 - ((255 - br) * 255) / (2 * tr))
              : tr === 255
                ? 255
                : Math.min(255, (br * 255) / (2 * (255 - tr)));
          g =
            tg < 128
              ? tg === 0
                ? 0
                : Math.max(0, 255 - ((255 - bg) * 255) / (2 * tg))
              : tg === 255
                ? 255
                : Math.min(255, (bg * 255) / (2 * (255 - tg)));
          b =
            tb < 128
              ? tb === 0
                ? 0
                : Math.max(0, 255 - ((255 - bb) * 255) / (2 * tb))
              : tb === 255
                ? 255
                : Math.min(255, (bb * 255) / (2 * (255 - tb)));
          break;

        case "linear":
          r = Math.max(0, Math.min(255, br + 2 * tr - 255));
          g = Math.max(0, Math.min(255, bg + 2 * tg - 255));
          b = Math.max(0, Math.min(255, bb + 2 * tb - 255));
          break;

        case "pin":
          r = tr < 128 ? Math.min(br, 2 * tr) : Math.max(br, 2 * tr - 255);
          g = tg < 128 ? Math.min(bg, 2 * tg) : Math.max(bg, 2 * tg - 255);
          b = tb < 128 ? Math.min(bb, 2 * tb) : Math.max(bb, 2 * tb - 255);
          break;

        case "hardmix":
          r = br + tr < 255 ? 0 : 255;
          g = bg + tg < 255 ? 0 : 255;
          b = bb + tb < 255 ? 0 : 255;
          break;

        case "hue": {
          const [bh, bs, bl] = rgbToHsl(br, bg, bb);
          const [th] = rgbToHsl(tr, tg, tb);
          [r, g, b] = hslToRgb(th, bs, bl);
          break;
        }

        case "saturation": {
          const [bh, , bl] = rgbToHsl(br, bg, bb);
          const [, ts] = rgbToHsl(tr, tg, tb);
          [r, g, b] = hslToRgb(bh, ts, bl);
          break;
        }

        case "color": {
          const [, , bl] = rgbToHsl(br, bg, bb);
          const [th, ts] = rgbToHsl(tr, tg, tb);
          [r, g, b] = hslToRgb(th, ts, bl);
          break;
        }

        case "luminosity": {
          const [bh, bs] = rgbToHsl(br, bg, bb);
          const [, , tl] = rgbToHsl(tr, tg, tb);
          [r, g, b] = hslToRgb(bh, bs, tl);
          break;
        }

        case "replace-dark-third": {
          const [, , bl] = rgbToHsl(br, bg, bb);
          if (bl < 0.333) {
            r = tr;
            g = tg;
            b = tb;
          } else {
            r = br;
            g = bg;
            b = bb;
          }
          break;
        }

        case "replace-mid-third": {
          const [, , bl] = rgbToHsl(br, bg, bb);
          if (bl >= 0.333 && bl < 0.667) {
            r = tr;
            g = tg;
            b = tb;
          } else {
            r = br;
            g = bg;
            b = bb;
          }
          break;
        }

        case "replace-light-third": {
          const [, , bl] = rgbToHsl(br, bg, bb);
          if (bl >= 0.667) {
            r = tr;
            g = tg;
            b = tb;
          } else {
            r = br;
            g = bg;
            b = bb;
          }
          break;
        }

        case "opacity-25":
          r = br * 0.75 + tr * 0.25;
          g = bg * 0.75 + tg * 0.25;
          b = bb * 0.75 + tb * 0.25;
          break;

        case "opacity-50":
          r = br * 0.5 + tr * 0.5;
          g = bg * 0.5 + tg * 0.5;
          b = bb * 0.5 + tb * 0.5;
          break;

        case "opacity-75":
          r = br * 0.25 + tr * 0.75;
          g = bg * 0.25 + tg * 0.75;
          b = bb * 0.25 + tb * 0.75;
          break;

        case "glow":
          // Glow: screen + extra brightness boost
          r = 255 - ((255 - br) * (255 - tr)) / 255;
          g = 255 - ((255 - bg) * (255 - tg)) / 255;
          b = 255 - ((255 - bb) * (255 - tb)) / 255;
          r = Math.min(255, r * 1.2);
          g = Math.min(255, g * 1.2);
          b = Math.min(255, b * 1.2);
          break;

        case "negation":
          // Inverts based on top layer
          r = 255 - Math.abs(255 - br - tr);
          g = 255 - Math.abs(255 - bg - tg);
          b = 255 - Math.abs(255 - bb - tb);
          break;

        case "phoenix":
          // Phoenix: min + max - 255
          r = Math.min(br, tr) - Math.max(br, tr) + 255;
          g = Math.min(bg, tg) - Math.max(bg, tg) + 255;
          b = Math.min(bb, tb) - Math.max(bb, tb) + 255;
          break;

        case "reflect":
          // Reflect: like dodge but squared
          r = tr === 255 ? 255 : Math.min(255, (br * br) / (255 - tr));
          g = tg === 255 ? 255 : Math.min(255, (bg * bg) / (255 - tg));
          b = tb === 255 ? 255 : Math.min(255, (bb * bb) / (255 - tb));
          break;

        case "freeze":
          // Freeze: inverse of reflect
          r = tr === 0 ? 0 : Math.max(0, 255 - ((255 - br) * (255 - br)) / tr);
          g = tg === 0 ? 0 : Math.max(0, 255 - ((255 - bg) * (255 - bg)) / tg);
          b = tb === 0 ? 0 : Math.max(0, 255 - ((255 - bb) * (255 - bb)) / tb);
          break;

        case "heat":
          // Heat: reflect with swapped layers
          r = br === 255 ? 255 : Math.min(255, (tr * tr) / (255 - br));
          g = bg === 255 ? 255 : Math.min(255, (tg * tg) / (255 - bg));
          b = bb === 255 ? 255 : Math.min(255, (tb * tb) / (255 - bb));
          break;

        case "stamp":
          // Stamp: emboss-like effect
          r = Math.max(0, Math.min(255, br + 2 * tr - 256));
          g = Math.max(0, Math.min(255, bg + 2 * tg - 256));
          b = Math.max(0, Math.min(255, bb + 2 * tb - 256));
          break;

        case "geometric":
          // Geometric mean
          r = Math.sqrt(br * tr);
          g = Math.sqrt(bg * tg);
          b = Math.sqrt(bb * tb);
          break;

        case "hypot":
          // Hypotenuse blend
          r = Math.min(255, Math.sqrt(br * br + tr * tr) / Math.SQRT2);
          g = Math.min(255, Math.sqrt(bg * bg + tg * tg) / Math.SQRT2);
          b = Math.min(255, Math.sqrt(bb * bb + tb * tb) / Math.SQRT2);
          break;

        default:
          r = tr;
          g = tg;
          b = tb;
      }

      setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
    }
  }

  return out;
}

export { blend };
