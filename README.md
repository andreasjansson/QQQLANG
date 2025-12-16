![QQQLANG Logo](public/logo.png)

# QQQLANG: A syntax-free programming language for image synthesis

**https://qqqlang.com**

In QQQLANG, any string of visible uppercase ASCII characters is a valid program.

Each character has three properties:
- An integer ('A'=1, 'B'=2, [...], '}'=67, '~'=68)
- A color
- A function

Functions can take zero or more arguments. If a function takes arguments, the characters that follow are interpreted as arguments. Otherwise characters are interpreted as functions. The exception is the first character of the program string which sets an initial solid color.

For example, the program `ABCD` has the following interpretation:

- `A` sets the initial color to #78A10F
- `B` is the 'border' function that creates a circular gradient around the edges. It takes one argument, the border color.
- `C` becomes the argument to 'B', the color of 'C' is #FF6B35
- `D` is the 'drip' function, which creates a water drop effect. It takes no arguments.

If the program string ends before the last function has had arguments defined, it will use its own number and color as default arguments. For example, the programs `AL`, `ALL`, and `ALLL` are equivalent.

The question mark character `?` is also a function that displays help text. `?1` and `??` show the first page of help, and `?A`, `?B`, etc. show subsequent pages of help text.

Some functions take an image index as an argument, and uses that old image in some way. `?#` shows the history of images and the characters to use to retrieve each image.

---

# About

QQQLANG is built by me, Andreas Jansson, and is MIT licensed. The code is on github.com/andreasjansson/qqqlang.

QQQ is short for QQQEJOTTONO, a word that my three-year old son wrote on a label maker. He then went on to write fifty or so other words, until the label roll ran out. I thought it'd be nice if these labels could be treated like code.

So QQQLANG is really a language designed for three year olds. It's a Turing complete* stack-based language that can accept any string of characters as a valid program, because each character is either a function name or an argument, depending on context.

(* Turing complete because it includes a Rule 110 function)

There are 68 functions, some are normal image editing functions like '1' (colorize), and some are weird, like 'L' (3D Lissajous tubes) or 'V' (overlay another image in the stack in Voronoi patterns).

The output images are completely deterministic given the program string and canvas size. You can share a qqqlang.com URL to replicate and fork the image.

QQQLANG is both an image synthesis and editing language. You can upload an image as the starting image, or as arguments to functions that take image inputs. You can also paste images from the clipboard, or paste image URLs.

The language is complete and won't change (other than bug fixes). But anyone can fork the language and add new functions as a different language. It would be both fun and possible to build languages like QQQ-AUDIO, QQQ-VIDEO, QQQ-3D, etc.

---

# Character Reference

## `A` — number 1, color ![#78A10F](assets/01-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywA"><img align="right" width="384" src="assets/01-example.png"></a>

**Function:** `spheres` — Renders image as texture on two 3D spheres with lighting.

<br clear="right">

---

## `B` — number 2, color ![#8B4513](assets/02-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywBAA"><img align="right" width="384" src="assets/02-example.png"></a>

**Function:** `border` — Border effect with various shapes.

**Arguments:**
   1. Border shape style (A=circular-solid, B=circular-blur, C=horizontal-solid, D=horizontal-blur, E=vertical-solid, F=vertical-blur, G=rectangular-solid, H=rectangular-blur, ...)
   2. Border color

<br clear="right">

---

## `C` — number 3, color ![#FF6B35](assets/03-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywCP"><img align="right" width="384" src="assets/03-example.png"></a>

**Function:** `concentric-hue` — Alternating original and hue-shifted concentric circles.

**Arguments:**
   1. Number of concentric circles

<br clear="right">

---

## `D` — number 4, color ![#FF1493](assets/04-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywD"><img align="right" width="384" src="assets/04-example.png"></a>

**Function:** `drip` — Metaball-based dripping water drops effect.

<br clear="right">

---

## `E` — number 5, color ![#50C878](assets/05-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywE"><img align="right" width="384" src="assets/05-example.png"></a>

**Function:** `emerald` — Renders reflective 3D emeralds in symmetric pattern.

<br clear="right">

---

## `F` — number 6, color ![#FFD700](assets/06-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywFB"><img align="right" width="384" src="assets/06-example.png"></a>

**Function:** `fft-overflow` — 2D FFT with magnitude overflow and chromatic phase shifts.

**Arguments:**
   1. FFT multiplier strength

<br clear="right">

---

## `G` — number 7, color ![#9370DB](assets/07-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywGG"><img align="right" width="384" src="assets/07-example.png"></a>

**Function:** `grayscale-colorize` — Converts to grayscale then applies rainbow palette.

**Arguments:**
   1. Number of posterize colors

<br clear="right">

---

## `H` — number 8, color ![#DC143C](assets/08-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywH"><img align="right" width="384" src="assets/08-example.png"></a>

**Function:** `hourglass` — Hourglass gradient with bitwise color blending.

<br clear="right">

---

## `I` — number 9, color ![#00FF7F](assets/09-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywI"><img align="right" width="384" src="assets/09-example.png"></a>

**Function:** `invert-edges` — Inverts colors then adds Sobel edge detection.

<br clear="right">

---

## `J` — number 10, color ![#FF8C00](assets/10-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywJM"><img align="right" width="384" src="assets/10-example.png"></a>

**Function:** `julia-fractal` — Julia set fractal masking the previous image.

**Arguments:**
   1. Fractal zoom depth

<br clear="right">

---

## `K` — number 11, color ![#9966FF](assets/11-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywKE"><img align="right" width="384" src="assets/11-example.png"></a>

**Function:** `kaleidoscope` — N-way kaleidoscope effect with zoom.

**Arguments:**
   1. Number of kaleidoscope segments

<br clear="right">

---

## `L` — number 12, color ![#20B2AA](assets/12-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywLLL"><img align="right" width="384" src="assets/12-example.png"></a>

**Function:** `lissajous` — 3D Lissajous tube with textured surface.

**Arguments:**
   1. Old image for tube texture
   2. Rotation angle multiplier

<br clear="right">

---

## `M` — number 13, color ![#FF69B4](assets/13-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywMC"><img align="right" width="384" src="assets/13-example.png"></a>

**Function:** `moire` — Moiré interference pattern with color zones.

**Arguments:**
   1. Pattern complexity seed

<br clear="right">

---

## `N` — number 14, color ![#8A2BE2](assets/14-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywN"><img align="right" width="384" src="assets/14-example.png"></a>

**Function:** `neon` — Neon glow effect on bright edges.

<br clear="right">

---

## `O` — number 15, color ![#FF6347](assets/15-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywO9"><img align="right" width="384" src="assets/15-example.png"></a>

**Function:** `oil-slick` — Domain warping with iridescent lighting.

**Arguments:**
   1. Warp intensity and depth

<br clear="right">

---

## `P` — number 16, color ![#4682B4](assets/16-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywP6"><img align="right" width="384" src="assets/16-example.png"></a>

**Function:** `pixelate` — Pixelate with diagonal split using average/saturated colors.

**Arguments:**
   1. Pixel cell size

<br clear="right">

---

## `Q` — number 17, color ![#32CD32](assets/17-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywQ"><img align="right" width="384" src="assets/17-example.png"></a>

**Function:** `quad-prism` — Negative prism with diagonal inversion and mirroring.

<br clear="right">

---

## `R` — number 18, color ![#DA70D6](assets/18-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywR"><img align="right" width="384" src="assets/18-example.png"></a>

**Function:** `room` — 3D room with textured walls, ceiling, and floor.

<br clear="right">

---

## `S` — number 19, color ![#87CEEB](assets/19-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywSSS"><img align="right" width="384" src="assets/19-example.png"></a>

**Function:** `sierpinski` — Sierpiński triangle fractal with color effects.

**Arguments:**
   1. Old image for triangle interior
   2. Fractal detail level (A-~)

<br clear="right">

---

## `T` — number 20, color ![#F0E68C](assets/20-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywT8"><img align="right" width="384" src="assets/20-example.png"></a>

**Function:** `tiles` — Grid of 3D tiles covering entire canvas, heights based on seed with multiplier.

**Arguments:**
   1. Building height multiplier (A=short, ~=tall)

<br clear="right">

---

## `U` — number 21, color ![#DDA0DD](assets/21-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywUB"><img align="right" width="384" src="assets/21-example.png"></a>

**Function:** `dither` — Apply one of 16 dithering algorithms, from subtle to aggressive 2-color modes.

**Arguments:**
   1. Dithering algorithm (A=ordered-5level, B=bayer-bw, C=threshold-bw, D=ordered-2bit, E=floyd-rgb, F=floyd-bw, G=atkinson-4level, H=atkinson-bw, ...)

<br clear="right">

---

## `V` — number 22, color ![#40E0D0](assets/22-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw)~1AVBJ"><img align="right" width="384" src="assets/22-example.png"></a>

**Function:** `voronoi` — Voronoi cells alternating between current and old image, with variable pattern shape.

**Arguments:**
   1. Old image to alternate with
   2. Pattern shape

<br clear="right">

---

## `W` — number 23, color ![#EE82EE](assets/23-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywWW"><img align="right" width="384" src="assets/23-example.png"></a>

**Function:** `whirl` — Swirl distortion from center with quadratic falloff.

**Arguments:**
   1. Rotation multiplier (×20°)

<br clear="right">

---

## `X` — number 24, color ![#F5DEB3](assets/24-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywX~"><img align="right" width="384" src="assets/24-example.png"></a>

**Function:** `cppn` — Compositional Pattern Producing Network warps and modulates saturation/value using a neural network. Fully deterministic based on input.

**Arguments:**
   1. Effect strength

<br clear="right">

---

## `Y` — number 25, color ![#98FB98](assets/25-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywYY"><img align="right" width="384" src="assets/25-example.png"></a>

**Function:** `yuv-shift` — YUV color shift with three gradient directions 120° apart: luminance, blue chrominance, and red chrominance shifts.

**Arguments:**
   1. Controls angle and intensity of color shifts

<br clear="right">

---

## `Z` — number 26, color ![#AFEEEE](assets/26-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywZK"><img align="right" width="384" src="assets/26-example.png"></a>

**Function:** `zoom-blur` — Radial motion blur from center with sharp center.

**Arguments:**
   1. Blur strength multiplier (×4px)

<br clear="right">

---

## `0` — number 27, color ![#E6E6FA](assets/27-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw0A"><img align="right" width="384" src="assets/27-example.png"></a>

**Function:** `bg-remove` — Removes background from prev image using ML, composites on specified background.

**Arguments:**
   1. Background image to composite behind

<br clear="right">

---

## `1` — number 28, color ![#FFA07A](assets/28-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw1E"><img align="right" width="384" src="assets/28-example.png"></a>

**Function:** `colorize` — Tints the image with the specified color using gamma-corrected luminance and boosted saturation.

**Arguments:**
   1. Tint color applied based on luminance

<br clear="right">

---

## `2` — number 29, color ![#98D8C8](assets/29-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw2BAB"><img align="right" width="384" src="assets/29-example.png"></a>

**Function:** `third-stamp` — Replace a vertical third of current image with a third from old image.

**Arguments:**
   1. Old image to extract third from
   2. Old image third (1=left, 2=mid, 3=right, cycling)
   3. Current image third to replace (1=left, 2=mid, 3=right, cycling)

<br clear="right">

---

## `3` — number 30, color ![#F7DC6F](assets/30-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw3"><img align="right" width="384" src="assets/30-example.png"></a>

**Function:** `triple-rotate` — Three vertical strips with different rotations.

<br clear="right">

---

## `4` — number 31, color ![#BB8FCE](assets/31-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw4"><img align="right" width="384" src="assets/31-example.png"></a>

**Function:** `quad-rotate` — Four quadrants each rotated 0°, 90°, 180°, 270°.

<br clear="right">

---

## `5` — number 32, color ![#85C1E9](assets/32-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw55"><img align="right" width="384" src="assets/32-example.png"></a>

**Function:** `triangular-split` — Triangular grid with hue shifts and lightness variation.

**Arguments:**
   1. Cell size multiplier

<br clear="right">

---

## `6` — number 33, color ![#F1948A](assets/33-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw6"><img align="right" width="384" src="assets/33-example.png"></a>

**Function:** `posterize` — Posterize to 4 levels per channel.

<br clear="right">

---

## `7` — number 34, color ![#82E0AA](assets/34-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw777"><img align="right" width="384" src="assets/34-example.png"></a>

**Function:** `chromatic` — Chromatic aberration with RGB channel shifts.

<br clear="right">

---

## `8` — number 35, color ![#F8C471](assets/35-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw8T"><img align="right" width="384" src="assets/35-example.png"></a>

**Function:** `lemniscate` — Infinity-loop lemniscate distortion.

**Arguments:**
   1. Distortion strength

<br clear="right">

---

## `9` — number 36, color ![#D7BDE2](assets/36-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywYY9B"><img align="right" width="384" src="assets/36-example.png"></a>

**Function:** `xor-blend` — XOR blend creating glitchy digital artifacts.

**Arguments:**
   1. Old image to XOR with

<br clear="right">

---

## `<` — number 37, color ![#E74C3C](assets/37-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3CJ"><img align="right" width="384" src="assets/37-example.png"></a>

**Function:** `horizontal-shift` — Horizontal shift with wraparound.

**Arguments:**
   1. Shift amount (A=left, 7=none, ~=right)

<br clear="right">

---

## `>` — number 38, color ![#3498DB](assets/38-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3E"><img align="right" width="384" src="assets/38-example.png"></a>

**Function:** `rotate-90` — Rotate 90 degrees clockwise.

<br clear="right">

---

## `^` — number 39, color ![#2ECC71](assets/39-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5EV"><img align="right" width="384" src="assets/39-example.png"></a>

**Function:** `vertical-shift` — Vertical shift with wraparound.

**Arguments:**
   1. Shift amount (A=up, 7=none, ~=down)

<br clear="right">

---

## `!` — number 40, color ![#FF4500](assets/40-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw!"><img align="right" width="384" src="assets/40-example.png"></a>

**Function:** `godrays` — Volumetric light scattering from center.

<br clear="right">

---

## `"` — number 41, color ![#9932CC](assets/41-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%22"><img align="right" width="384" src="assets/41-example.png"></a>

**Function:** `band-transform` — Horizontal bands with alternating hue/saturation transforms.

**Arguments:**
   1. Number of horizontal bands

<br clear="right">

---

## `#` — number 42, color ![#228B22](assets/42-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%23B"><img align="right" width="384" src="assets/42-example.png"></a>

**Function:** `insert` — Replaces current image with specified old image.

**Arguments:**
   1. Old image index to insert

<br clear="right">

---

## `$` — number 43, color ![#FFD700](assets/43-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%24"><img align="right" width="384" src="assets/43-example.png"></a>

**Function:** `segment-hue-sort` — Color-based segmentation, then sorts pixels by hue within each segment.

<br clear="right">

---

## `%` — number 44, color ![#8B0000](assets/44-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%25A"><img align="right" width="384" src="assets/44-example.png"></a>

**Function:** `flip` — Flips image horizontally or vertically based on argument parity.

**Arguments:**
   1. Flip direction (even=horizontal, odd=vertical)

<br clear="right">

---

## `&` — number 45, color ![#4169E1](assets/45-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%26P"><img align="right" width="384" src="assets/45-example.png"></a>

**Function:** `quadtree-compress` — Adaptive quadtree compression - detailed areas keep resolution while uniform areas become large blocks, creating geometric patterns.

**Arguments:**
   1. Compression level (A=minimal/detailed, ~=maximal/geometric blocks)

<br clear="right">

---

## `'` — number 46, color ![#FF1493](assets/46-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1%27%27B"><img align="right" width="384" src="assets/46-example.png"></a>

**Function:** `variable-checkerboard` — Checkerboard blend with increasing square size from corner to corner.

**Arguments:**
   1. Old image to checkerboard with

<br clear="right">

---

## `(` — number 47, color ![#00CED1](assets/47-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw(3F%26"><img align="right" width="384" src="assets/47-example.png"></a>

**Function:** `shear-radial` — Combined shear and radial distortion. Shear amount couples to horizontal offset and radial strength.

**Arguments:**
   1. Center X offset (A=left, M=center, ~=right)
   2. Center Y offset (A=bottom, M=center, ~=top)
   3. Radial/shear strength (A=barrel, M=none, ~=pincushion)

<br clear="right">

---

## `)` — number 48, color ![#FF69B4](assets/48-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))"><img align="right" width="384" src="assets/48-example.png"></a>

**Function:** `blur` — Gaussian blur with adjustable radius using two-pass convolution.

**Arguments:**
   1. Blur radius (A=subtle, ~=heavy)

<br clear="right">

---

## `*` — number 49, color ![#FFD700](assets/49-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw*"><img align="right" width="384" src="assets/49-example.png"></a>

**Function:** `fur` — Fur/hair strands growing from pixels based on hue and noise.

<br clear="right">

---

## `+` — number 50, color ![#32CD32](assets/50-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2B%2B%2B%2B%2B%2B%2B"><img align="right" width="384" src="assets/50-example.png"></a>

**Function:** `zoom` — Zoom in 1.2× from center.

<br clear="right">

---

## `,` — number 51, color ![#BA55D3](assets/51-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2CD"><img align="right" width="384" src="assets/51-example.png"></a>

**Function:** `stipple` — Stipple dots at luminance-based positions.

**Arguments:**
   1. Stipple dot color

<br clear="right">

---

## `-` — number 52, color ![#FF7F50](assets/52-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1D-B6"><img align="right" width="384" src="assets/52-example.png"></a>

**Function:** `blend` — Blend old image with current using specified mode.

**Arguments:**
   1. Old image to blend with
   2. Blend mode (A=multiply, B=screen, C=overlay, D=darken, E=lighten, F=dodge, G=burn, H=hardlight, ...)

<br clear="right">

---

## `.` — number 53, color ![#20B2AA](assets/53-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw.M"><img align="right" width="384" src="assets/53-example.png"></a>

**Function:** `pointillism` — Pointillism effect with saturated circular dots.

**Arguments:**
   1. Dot radius base (mod 8 + 2)

<br clear="right">

---

## `/` — number 54, color ![#CD853F](assets/54-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2FB073C"><img align="right" width="384" src="assets/54-example.png"></a>

**Function:** `circle-stamp` — Stamp circular region from old image center onto current.

**Arguments:**
   1. Old image source
   2. X position (A=left, 7=center, ~=right)
   3. Y position (A=top, 7=center, ~=bottom)
   4. Circle size (A=tiny, ~=full)
   5. Blend mode (A=normal, B=xor, C=nand, D=and, E=or, F=multiply, G=screen, H=overlay, ...)

<br clear="right">

---

## `:` — number 55, color ![#6B8E23](assets/55-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw1D%3AB"><img align="right" width="384" src="assets/55-example.png"></a>

**Function:** `porthole` — Circular window showing current image with old image as background.

**Arguments:**
   1. Old image for background

<br clear="right">

---

## `;` — number 56, color ![#DB7093](assets/56-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3B"><img align="right" width="384" src="assets/56-example.png"></a>

**Function:** `semicircle-reflect` — Top semicircle preserved, bottom reflected with wave distortion.

<br clear="right">

---

## `=` — number 57, color ![#5F9EA0](assets/57-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3DX"><img align="right" width="384" src="assets/57-example.png"></a>

**Function:** `shifted-stripes` — Horizontal stripes with alternating shifts.

**Arguments:**
   1. Stripe height in pixels

<br clear="right">

---

## `?` — number 58, color ![#D2691E](assets/58-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3F%3F"><img align="right" width="384" src="assets/58-example.png"></a>

**Function:** `help` — Display help text or image history table.

**Arguments:**
   1. Page number (A=intro, B+=reference, #=history)

<br clear="right">

---

## `@` — number 59, color ![#7B68EE](assets/59-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywGA%23B1.%23B1C%40CEGAE"><img align="right" width="384" src="assets/59-example.png"></a>

**Function:** `cond` — Per-pixel conditional: extracts channel from condition image, outputs true-image pixel where value >= threshold, otherwise false-image pixel.

**Arguments:**
   1. Condition image sampled for threshold comparison
   2. Source image when condition >= threshold
   3. Source image when condition < threshold
   4. Color channel to extract from condition image (A=hue, B=saturation, C=lightness, D=red, E=green, F=blue)
   5. Threshold (A=0%, ~=100% of channel range)

<br clear="right">

---

## `[` — number 60, color ![#48D1CC](assets/60-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5BX"><img align="right" width="384" src="assets/60-example.png"></a>

**Function:** `rotate` — Rotate around center.

**Arguments:**
   1. Rotation amount (A=left, 7=none, ~=right)

<br clear="right">

---

## `\\` — number 61, color ![#C71585](assets/61-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5CB4VXMVH-0%7DA"><img align="right" width="384" src="assets/61-example.png"></a>

**Function:** `composite` — Composite transformed region from old image onto current.

**Arguments:**
   1. Old image source
   2. Source X (normalized 0-1)
   3. Source Y (normalized 0-1)
   4. Source width (normalized 0-1)
   5. Source height (normalized 0-1)
   6. Dest X (normalized 0-1)
   7. Dest Y (normalized 0-1)
   8. Dest width (normalized 0-1)
   9. Dest height (normalized 0-1)
   10. Rotation (normalized 0-1 → 0-360°)
   11. Blend mode (mod 16: normal, xor, nand, and, or, multiply, screen, overlay, darken, lighten, diff, excl, add, sub, hard, soft)

<br clear="right">

---

## `]` — number 62, color ![#00FA9A](assets/62-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5D"><img align="right" width="384" src="assets/62-example.png"></a>

**Function:** `left-half-offset` — Shift left half vertically by 20% with wraparound.

<br clear="right">

---

## `_` — number 63, color ![#708090](assets/63-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw_"><img align="right" width="384" src="assets/63-example.png"></a>

**Function:** `scanlines` — CRT scanline effect with darkening and displacement.

<br clear="right">

---

## `\`` — number 64, color ![#6495ED](assets/64-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%60M"><img align="right" width="384" src="assets/64-example.png"></a>

**Function:** `rule110` — Rule 110 cellular automaton - a Turing-complete 1D CA applied horizontally to each row.

**Arguments:**
   1. Number of generations (×8)

<br clear="right">

---

## `{` — number 65, color ![#DC143C](assets/65-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%7BR"><img align="right" width="384" src="assets/65-example.png"></a>

**Function:** `skew` — Skew horizontal with wraparound.

**Arguments:**
   1. Skew amount (A=left, 7=none, ~=right)

<br clear="right">

---

## `|` — number 66, color ![#00BFFF](assets/66-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1%7C%7C%7C"><img align="right" width="384" src="assets/66-example.png"></a>

**Function:** `vertical-split` — Vertical split with wavy blend zone using multiple blend modes.

**Arguments:**
   1. Old image for right half

<br clear="right">

---

## `}` — number 67, color ![#9400D3](assets/67-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%7D"><img align="right" width="384" src="assets/67-example.png"></a>

**Function:** `sharpen` — Sharpen using convolution kernel to enhance edges.

<br clear="right">

---

## `~` — number 68, color ![#FF6347](assets/68-color.png)

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw~F"><img align="right" width="384" src="assets/68-example.png"></a>

**Function:** `wave-chromatic` — Horizontal wave distortion with chromatic aberration.

**Arguments:**
   1. Wave amplitude and chromatic shift

<br clear="right">

---

## License

MIT

## Author

Andreas Jansson ([@andreasjansson](https://github.com/andreasjansson))
