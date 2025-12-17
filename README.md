[![QQQLANG Logo](public/logo.png)](https://qqqlang.com)

# QQQLANG: A syntax-free programming language for image synthesis

**https://qqqlang.com**

In QQQLANG, any string of visible uppercase ASCII characters is a valid program.

Each character has three properties:
- An integer (`A`=1, `B`=2, [...], `}`=67, `~`=68)
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

# Gallery

<table>
<tr>
<td><a href="https://qqqlang.com/?p=AF%60H%2BF%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%2B%7B4%7DHX~WDCBY5N%24UA77"><img src="assets/gallery-0.png" width="256"></a></td>
<td><a href="https://qqqlang.com/?p=%E2%98%80CWc022HqJzLmt018fuUEUALBL%2B%2B8.%26%26FF((((-I%5EXA%2CV%23F.ANH-BFVL0((((D"><img src="assets/gallery-1.png" width="256"></a></td>
<td><a href="https://qqqlang.com/?p=AFFGERE"><img src="assets/gallery-2.png" width="256"></a></td>
</tr>
<tr>
<td><a href="https://qqqlang.com/?p=AVWNA.~%7D5YMDJA%40FGIBJ8G-K~H%3AJJSX.%3C%3D%7CJ%3CE%7C%7C%25%25"><img src="assets/gallery-3.png" width="256"></a></td>
<td><a href="https://qqqlang.com/?p=KL%3BSWW6%7D%23ALLL%2B%3E%7B3%3E%3E%3EQ1D(..!%3DE%23F0R2FBBXH"><img src="assets/gallery-4.png" width="256"></a></td>
<td><a href="https://qqqlang.com/?p=FSEDSH2H%25A8%40"><img src="assets/gallery-5.png" width="256"></a></td>
</tr>
<tr>
<td><a href="https://qqqlang.com/?p=WEXO6%7D655%3A%3A%7B%23FAYF-J3"><img src="assets/gallery-6.png" width="256"></a></td>
<td><a href="https://qqqlang.com/?p=A5XWF%7DJD55T%3D(665"><img src="assets/gallery-7.png" width="256"></a></td>
<td><a href="https://qqqlang.com/?p=1QQ(FX6JERHSQ3WF%25%251B-WQ"><img src="assets/gallery-8.png" width="256"></a></td>
</tr>
</table>

# About

QQQ is short for QQQEJOTTONO, a word that my three-year old son wrote on a label maker. He then went on to write fifty or so other words, until the label roll ran out. I thought it'd be nice if these labels could be treated like code.

So QQQLANG is really a language designed for three year olds. It's a Turing complete* stack-based language that can accept any string of characters as a valid program, because each character is either a function name or an argument, depending on context.

(* Turing complete because it includes a Rule 110 function)

There are 68 functions, some are normal image editing functions like '1' (colorize), and some are weird, like `L` (3D Lissajous tubes) or `V` (overlay another image in the stack in Voronoi patterns).

The output images are completely deterministic given the program string and canvas size. You can share a qqqlang.com URL to replicate and fork the image.

QQQLANG is both an image synthesis and editing language. You can upload an image as the starting image, or as arguments to functions that take image inputs. You can also paste images from the clipboard, or paste image URLs.

This project is now finished and the language won't change (other than bug fixes). But anyone can fork the language and add new functions as a different language. It would be both fun and possible to build languages like QQQ-AUDIO, QQQ-VIDEO, QQQ-3D, etc.

# Character reference

## `A`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywA"><img align="right" width="384" src="assets/01-example.png"></a>

**Number:** 1 · **Color:** ![#78A10F](assets/01-color.png) #78A10F

**Function:** `spheres` — Renders image as texture on two 3D spheres with lighting.

<br clear="right">

## `B`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywBAA"><img align="right" width="384" src="assets/02-example.png"></a>

**Number:** 2 · **Color:** ![#8B4513](assets/02-color.png) #8B4513

**Function:** `border` — Border effect with various shapes.

**Arguments:**
   1. Border shape style (A=circular-solid, B=circular-blur, C=horizontal-solid, D=horizontal-blur, E=vertical-solid, F=vertical-blur, G=rectangular-solid, H=rectangular-blur, I=diamond-solid, J=diamond-blur, K=hexagon-solid, L=hexagon-blur, M=sine-horizontal-solid, N=sine-horizontal-blur, O=sine-vertical-solid, P=sine-vertical-blur, Q=triangle-horizontal-solid, R=triangle-horizontal-blur, S=triangle-vertical-solid, T=triangle-vertical-blur)
   2. Border color

<br clear="right">

## `C`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywCP"><img align="right" width="384" src="assets/03-example.png"></a>

**Number:** 3 · **Color:** ![#FF6B35](assets/03-color.png) #FF6B35

**Function:** `concentric-hue` — Alternating original and hue-shifted concentric circles.

**Arguments:**
   1. Number of concentric circles

<br clear="right">

## `D`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywD"><img align="right" width="384" src="assets/04-example.png"></a>

**Number:** 4 · **Color:** ![#FF1493](assets/04-color.png) #FF1493

**Function:** `drip` — Metaball-based dripping water drops effect.

<br clear="right">

## `E`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywE"><img align="right" width="384" src="assets/05-example.png"></a>

**Number:** 5 · **Color:** ![#50C878](assets/05-color.png) #50C878

**Function:** `emerald` — Renders reflective 3D emeralds in symmetric pattern.

<br clear="right">

## `F`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywFB"><img align="right" width="384" src="assets/06-example.png"></a>

**Number:** 6 · **Color:** ![#FFD700](assets/06-color.png) #FFD700

**Function:** `fft-overflow` — 2D FFT with magnitude overflow and chromatic phase shifts.

**Arguments:**
   1. FFT multiplier strength

<br clear="right">

## `G`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywGG"><img align="right" width="384" src="assets/07-example.png"></a>

**Number:** 7 · **Color:** ![#9370DB](assets/07-color.png) #9370DB

**Function:** `grayscale-colorize` — Converts to grayscale then applies rainbow palette.

**Arguments:**
   1. Number of posterize colors

<br clear="right">

## `H`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywH"><img align="right" width="384" src="assets/08-example.png"></a>

**Number:** 8 · **Color:** ![#DC143C](assets/08-color.png) #DC143C

**Function:** `hourglass` — Hourglass gradient with bitwise color blending.

<br clear="right">

## `I`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywI"><img align="right" width="384" src="assets/09-example.png"></a>

**Number:** 9 · **Color:** ![#00FF7F](assets/09-color.png) #00FF7F

**Function:** `invert-edges` — Inverts colors then adds Sobel edge detection.

<br clear="right">

## `J`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywJM"><img align="right" width="384" src="assets/10-example.png"></a>

**Number:** 10 · **Color:** ![#FF8C00](assets/10-color.png) #FF8C00

**Function:** `julia-fractal` — Julia set fractal masking the previous image.

**Arguments:**
   1. Fractal zoom depth

<br clear="right">

## `K`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywKE"><img align="right" width="384" src="assets/11-example.png"></a>

**Number:** 11 · **Color:** ![#9966FF](assets/11-color.png) #9966FF

**Function:** `kaleidoscope` — N-way kaleidoscope effect with zoom.

**Arguments:**
   1. Number of kaleidoscope segments

<br clear="right">

## `L`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywLLL"><img align="right" width="384" src="assets/12-example.png"></a>

**Number:** 12 · **Color:** ![#20B2AA](assets/12-color.png) #20B2AA

**Function:** `lissajous` — 3D Lissajous tube with textured surface.

**Arguments:**
   1. Old image for tube texture
   2. Rotation angle multiplier

<br clear="right">

## `M`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywMC"><img align="right" width="384" src="assets/13-example.png"></a>

**Number:** 13 · **Color:** ![#FF69B4](assets/13-color.png) #FF69B4

**Function:** `moire` — Moiré interference pattern with color zones.

**Arguments:**
   1. Pattern complexity seed

<br clear="right">

## `N`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywN"><img align="right" width="384" src="assets/14-example.png"></a>

**Number:** 14 · **Color:** ![#8A2BE2](assets/14-color.png) #8A2BE2

**Function:** `neon` — Neon glow effect on bright edges.

<br clear="right">

## `O`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywO9"><img align="right" width="384" src="assets/15-example.png"></a>

**Number:** 15 · **Color:** ![#FF6347](assets/15-color.png) #FF6347

**Function:** `oil-slick` — Domain warping with iridescent lighting.

**Arguments:**
   1. Warp intensity and depth

<br clear="right">

## `P`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywP6"><img align="right" width="384" src="assets/16-example.png"></a>

**Number:** 16 · **Color:** ![#4682B4](assets/16-color.png) #4682B4

**Function:** `pixelate` — Pixelate with diagonal split using average/saturated colors.

**Arguments:**
   1. Pixel cell size

<br clear="right">

## `Q`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywQ"><img align="right" width="384" src="assets/17-example.png"></a>

**Number:** 17 · **Color:** ![#32CD32](assets/17-color.png) #32CD32

**Function:** `quad-prism` — Negative prism with diagonal inversion and mirroring.

<br clear="right">

## `R`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywR"><img align="right" width="384" src="assets/18-example.png"></a>

**Number:** 18 · **Color:** ![#DA70D6](assets/18-color.png) #DA70D6

**Function:** `room` — 3D room with textured walls, ceiling, and floor.

<br clear="right">

## `S`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywSSS"><img align="right" width="384" src="assets/19-example.png"></a>

**Number:** 19 · **Color:** ![#87CEEB](assets/19-color.png) #87CEEB

**Function:** `sierpinski` — Sierpiński triangle fractal with color effects.

**Arguments:**
   1. Old image for triangle interior
   2. Fractal detail level (A-~)

<br clear="right">

## `T`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywT8"><img align="right" width="384" src="assets/20-example.png"></a>

**Number:** 20 · **Color:** ![#F0E68C](assets/20-color.png) #F0E68C

**Function:** `tiles` — Grid of 3D tiles covering entire canvas, heights based on seed with multiplier.

**Arguments:**
   1. Building height multiplier (A=short, ~=tall)

<br clear="right">

## `U`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywUB"><img align="right" width="384" src="assets/21-example.png"></a>

**Number:** 21 · **Color:** ![#DDA0DD](assets/21-color.png) #DDA0DD

**Function:** `dither` — Apply one of 16 dithering algorithms, from subtle to aggressive 2-color modes.

**Arguments:**
   1. Dithering algorithm (A=ordered-5level, B=bayer-bw, C=threshold-bw, D=ordered-2bit, E=floyd-rgb, F=floyd-bw, G=atkinson-4level, H=atkinson-bw, I=stucki-6level, J=burkes, K=sierra, L=random-bw, M=cluster-2bit, N=bluenoise-bw, O=bayer2x2-2bit, P=noise-2bit)

<br clear="right">

## `V`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw)~1AVBJ"><img align="right" width="384" src="assets/22-example.png"></a>

**Number:** 22 · **Color:** ![#40E0D0](assets/22-color.png) #40E0D0

**Function:** `voronoi` — Voronoi cells alternating between current and old image, with variable pattern shape.

**Arguments:**
   1. Old image to alternate with
   2. Pattern shape

<br clear="right">

## `W`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywWW"><img align="right" width="384" src="assets/23-example.png"></a>

**Number:** 23 · **Color:** ![#EE82EE](assets/23-color.png) #EE82EE

**Function:** `whirl` — Swirl distortion from center with quadratic falloff.

**Arguments:**
   1. Rotation multiplier (×20°)

<br clear="right">

## `X`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywX~"><img align="right" width="384" src="assets/24-example.png"></a>

**Number:** 24 · **Color:** ![#F5DEB3](assets/24-color.png) #F5DEB3

**Function:** `cppn` — Compositional Pattern Producing Network warps and modulates saturation/value using a neural network. Fully deterministic based on input.

**Arguments:**
   1. Effect strength

<br clear="right">

## `Y`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywYY"><img align="right" width="384" src="assets/25-example.png"></a>

**Number:** 25 · **Color:** ![#98FB98](assets/25-color.png) #98FB98

**Function:** `yuv-shift` — YUV color shift with three gradient directions 120° apart: luminance, blue chrominance, and red chrominance shifts.

**Arguments:**
   1. Controls angle and intensity of color shifts

<br clear="right">

## `Z`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywZK"><img align="right" width="384" src="assets/26-example.png"></a>

**Number:** 26 · **Color:** ![#AFEEEE](assets/26-color.png) #AFEEEE

**Function:** `zoom-blur` — Radial motion blur from center with sharp center.

**Arguments:**
   1. Blur strength multiplier (×4px)

<br clear="right">

## `0`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw0A"><img align="right" width="384" src="assets/27-example.png"></a>

**Number:** 27 · **Color:** ![#E6E6FA](assets/27-color.png) #E6E6FA

**Function:** `bg-remove` — Removes background from prev image using ML, composites on specified background.

**Arguments:**
   1. Background image to composite behind

<br clear="right">

## `1`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw1E"><img align="right" width="384" src="assets/28-example.png"></a>

**Number:** 28 · **Color:** ![#FFA07A](assets/28-color.png) #FFA07A

**Function:** `colorize` — Tints the image with the specified color using gamma-corrected luminance and boosted saturation.

**Arguments:**
   1. Tint color applied based on luminance

<br clear="right">

## `2`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw2BAB"><img align="right" width="384" src="assets/29-example.png"></a>

**Number:** 29 · **Color:** ![#98D8C8](assets/29-color.png) #98D8C8

**Function:** `third-stamp` — Replace a vertical third of current image with a third from old image.

**Arguments:**
   1. Old image to extract third from
   2. Old image third (1=left, 2=mid, 3=right, cycling)
   3. Current image third to replace (1=left, 2=mid, 3=right, cycling)

<br clear="right">

## `3`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw3"><img align="right" width="384" src="assets/30-example.png"></a>

**Number:** 30 · **Color:** ![#F7DC6F](assets/30-color.png) #F7DC6F

**Function:** `triple-rotate` — Three vertical strips with different rotations.

<br clear="right">

## `4`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw4"><img align="right" width="384" src="assets/31-example.png"></a>

**Number:** 31 · **Color:** ![#BB8FCE](assets/31-color.png) #BB8FCE

**Function:** `quad-rotate` — Four quadrants each rotated 0°, 90°, 180°, 270°.

<br clear="right">

## `5`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw55"><img align="right" width="384" src="assets/32-example.png"></a>

**Number:** 32 · **Color:** ![#85C1E9](assets/32-color.png) #85C1E9

**Function:** `triangular-split` — Triangular grid with hue shifts and lightness variation.

**Arguments:**
   1. Cell size multiplier

<br clear="right">

## `6`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw6"><img align="right" width="384" src="assets/33-example.png"></a>

**Number:** 33 · **Color:** ![#F1948A](assets/33-color.png) #F1948A

**Function:** `posterize` — Posterize to 4 levels per channel.

<br clear="right">

## `7`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw777"><img align="right" width="384" src="assets/34-example.png"></a>

**Number:** 34 · **Color:** ![#82E0AA](assets/34-color.png) #82E0AA

**Function:** `chromatic` — Chromatic aberration with RGB channel shifts.

<br clear="right">

## `8`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw8T"><img align="right" width="384" src="assets/35-example.png"></a>

**Number:** 35 · **Color:** ![#F8C471](assets/35-color.png) #F8C471

**Function:** `lemniscate` — Infinity-loop lemniscate distortion.

**Arguments:**
   1. Distortion strength

<br clear="right">

## `9`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywYY9B"><img align="right" width="384" src="assets/36-example.png"></a>

**Number:** 36 · **Color:** ![#D7BDE2](assets/36-color.png) #D7BDE2

**Function:** `xor-blend` — XOR blend creating glitchy digital artifacts.

**Arguments:**
   1. Old image to XOR with

<br clear="right">

## `<`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3CJ"><img align="right" width="384" src="assets/37-example.png"></a>

**Number:** 37 · **Color:** ![#E74C3C](assets/37-color.png) #E74C3C

**Function:** `horizontal-shift` — Horizontal shift with wraparound.

**Arguments:**
   1. Shift amount (A=left, 7=none, ~=right)

<br clear="right">

## `>`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3E"><img align="right" width="384" src="assets/38-example.png"></a>

**Number:** 38 · **Color:** ![#3498DB](assets/38-color.png) #3498DB

**Function:** `rotate-90` — Rotate 90 degrees clockwise.

<br clear="right">

## `^`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5EV"><img align="right" width="384" src="assets/39-example.png"></a>

**Number:** 39 · **Color:** ![#2ECC71](assets/39-color.png) #2ECC71

**Function:** `vertical-shift` — Vertical shift with wraparound.

**Arguments:**
   1. Shift amount (A=up, 7=none, ~=down)

<br clear="right">

## `!`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw!"><img align="right" width="384" src="assets/40-example.png"></a>

**Number:** 40 · **Color:** ![#FF4500](assets/40-color.png) #FF4500

**Function:** `godrays` — Volumetric light scattering from center.

<br clear="right">

## `"`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%22"><img align="right" width="384" src="assets/41-example.png"></a>

**Number:** 41 · **Color:** ![#9932CC](assets/41-color.png) #9932CC

**Function:** `band-transform` — Horizontal bands with alternating hue/saturation transforms.

**Arguments:**
   1. Number of horizontal bands

<br clear="right">

## `#`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%23B"><img align="right" width="384" src="assets/42-example.png"></a>

**Number:** 42 · **Color:** ![#228B22](assets/42-color.png) #228B22

**Function:** `insert` — Replaces current image with specified old image.

**Arguments:**
   1. Old image index to insert

<br clear="right">

## `$`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%24"><img align="right" width="384" src="assets/43-example.png"></a>

**Number:** 43 · **Color:** ![#FFD700](assets/43-color.png) #FFD700

**Function:** `segment-hue-sort` — Color-based segmentation, then sorts pixels by hue within each segment.

<br clear="right">

## `%`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%25A"><img align="right" width="384" src="assets/44-example.png"></a>

**Number:** 44 · **Color:** ![#8B0000](assets/44-color.png) #8B0000

**Function:** `flip` — Flips image horizontally or vertically based on argument parity.

**Arguments:**
   1. Flip direction (even=horizontal, odd=vertical)

<br clear="right">

## `&`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%26P"><img align="right" width="384" src="assets/45-example.png"></a>

**Number:** 45 · **Color:** ![#4169E1](assets/45-color.png) #4169E1

**Function:** `quadtree-compress` — Adaptive quadtree compression - detailed areas keep resolution while uniform areas become large blocks, creating geometric patterns.

**Arguments:**
   1. Compression level (A=minimal/detailed, ~=maximal/geometric blocks)

<br clear="right">

## `'`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1%27%27B"><img align="right" width="384" src="assets/46-example.png"></a>

**Number:** 46 · **Color:** ![#FF1493](assets/46-color.png) #FF1493

**Function:** `variable-checkerboard` — Checkerboard blend with increasing square size from corner to corner.

**Arguments:**
   1. Old image to checkerboard with

<br clear="right">

## `(`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw(3F%26"><img align="right" width="384" src="assets/47-example.png"></a>

**Number:** 47 · **Color:** ![#00CED1](assets/47-color.png) #00CED1

**Function:** `shear-radial` — Combined shear and radial distortion. Shear amount couples to horizontal offset and radial strength.

**Arguments:**
   1. Center X offset (A=left, M=center, ~=right)
   2. Center Y offset (A=bottom, M=center, ~=top)
   3. Radial/shear strength (A=barrel, M=none, ~=pincushion)

<br clear="right">

## `)`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))"><img align="right" width="384" src="assets/48-example.png"></a>

**Number:** 48 · **Color:** ![#FF69B4](assets/48-color.png) #FF69B4

**Function:** `blur` — Gaussian blur with adjustable radius using two-pass convolution.

**Arguments:**
   1. Blur radius (A=subtle, ~=heavy)

<br clear="right">

## `*`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw*"><img align="right" width="384" src="assets/49-example.png"></a>

**Number:** 49 · **Color:** ![#FFD700](assets/49-color.png) #FFD700

**Function:** `fur` — Fur/hair strands growing from pixels based on hue and noise.

<br clear="right">

## `+`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2B%2B%2B%2B%2B%2B%2B"><img align="right" width="384" src="assets/50-example.png"></a>

**Number:** 50 · **Color:** ![#32CD32](assets/50-color.png) #32CD32

**Function:** `zoom` — Zoom in 1.2× from center.

<br clear="right">

## `,`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2CD"><img align="right" width="384" src="assets/51-example.png"></a>

**Number:** 51 · **Color:** ![#BA55D3](assets/51-color.png) #BA55D3

**Function:** `stipple` — Stipple dots at luminance-based positions.

**Arguments:**
   1. Stipple dot color

<br clear="right">

## `-`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1D-B6"><img align="right" width="384" src="assets/52-example.png"></a>

**Number:** 52 · **Color:** ![#FF7F50](assets/52-color.png) #FF7F50

**Function:** `blend` — Blend old image with current using specified mode.

**Arguments:**
   1. Old image to blend with
   2. Blend mode (A=multiply, B=screen, C=overlay, D=darken, E=lighten, F=dodge, G=burn, H=hardlight, I=softlight, J=difference, K=exclusion, L=add, M=subtract, N=xor, O=and, P=or, Q=nand, R=nor, S=xnor, T=average, U=divide, V=grain-extract, W=grain-merge, X=vivid, Y=linear, Z=pin, 0=hardmix, 1=hue, 2=saturation, 3=color, 4=luminosity, 5=replace-dark-third, 6=replace-mid-third, 7=replace-light-third, 8=opacity-25, 9=opacity-50, <=opacity-75, >=glow, ^=negation, !=phoenix, "=reflect, #=freeze, $=heat, %=stamp, &=geometric, '=hypot, (=modulo, )=modulo-reverse, *=sin-blend, +=cos-blend, ,=bitshift-left, -=bitshift-right, .=threshold-max, /=threshold-min, :=threshold-swap, ;=posterize-blend)

<br clear="right">

## `.`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw.M"><img align="right" width="384" src="assets/53-example.png"></a>

**Number:** 53 · **Color:** ![#20B2AA](assets/53-color.png) #20B2AA

**Function:** `pointillism` — Pointillism effect with saturated circular dots.

**Arguments:**
   1. Dot radius base (mod 8 + 2)

<br clear="right">

## `/`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2FB073C"><img align="right" width="384" src="assets/54-example.png"></a>

**Number:** 54 · **Color:** ![#CD853F](assets/54-color.png) #CD853F

**Function:** `circle-stamp` — Stamp circular region from old image center onto current.

**Arguments:**
   1. Old image source
   2. X position (A=left, 7=center, ~=right)
   3. Y position (A=top, 7=center, ~=bottom)
   4. Circle size (A=tiny, ~=full)
   5. Blend mode (A=normal, B=xor, C=nand, D=and, E=or, F=multiply, G=screen, H=overlay, I=darken, J=lighten, K=difference, L=exclusion, M=add, N=subtract, O=hardlight, P=softlight)

<br clear="right">

## `:`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw1D%3AB"><img align="right" width="384" src="assets/55-example.png"></a>

**Number:** 55 · **Color:** ![#6B8E23](assets/55-color.png) #6B8E23

**Function:** `porthole` — Circular window showing current image with old image as background.

**Arguments:**
   1. Old image for background

<br clear="right">

## `;`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3B"><img align="right" width="384" src="assets/56-example.png"></a>

**Number:** 56 · **Color:** ![#DB7093](assets/56-color.png) #DB7093

**Function:** `semicircle-reflect` — Top semicircle preserved, bottom reflected with wave distortion.

<br clear="right">

## `=`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3DX"><img align="right" width="384" src="assets/57-example.png"></a>

**Number:** 57 · **Color:** ![#5F9EA0](assets/57-color.png) #5F9EA0

**Function:** `shifted-stripes` — Horizontal stripes with alternating shifts.

**Arguments:**
   1. Stripe height in pixels

<br clear="right">

## `?`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3F%3F"><img align="right" width="384" src="assets/58-example.png"></a>

**Number:** 58 · **Color:** ![#D2691E](assets/58-color.png) #D2691E

**Function:** `help` — Display help text or image history table.

**Arguments:**
   1. Page number (A=intro, B+=reference, #=history)

<br clear="right">

## `@`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywGA%23B1.%23B1C%40CEGAE"><img align="right" width="384" src="assets/59-example.png"></a>

**Number:** 59 · **Color:** ![#7B68EE](assets/59-color.png) #7B68EE

**Function:** `cond` — Per-pixel conditional: extracts channel from condition image, outputs true-image pixel where value >= threshold, otherwise false-image pixel.

**Arguments:**
   1. Condition image sampled for threshold comparison
   2. Source image when condition >= threshold
   3. Source image when condition < threshold
   4. Color channel to extract from condition image (A=hue, B=saturation, C=lightness, D=red, E=green, F=blue)
   5. Threshold (A=0%, ~=100% of channel range)

<br clear="right">

## `[`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5BX"><img align="right" width="384" src="assets/60-example.png"></a>

**Number:** 60 · **Color:** ![#48D1CC](assets/60-color.png) #48D1CC

**Function:** `rotate` — Rotate around center.

**Arguments:**
   1. Rotation amount (A=left, 7=none, ~=right)

<br clear="right">

## `\\`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5CB4VXMVH-0%7DA"><img align="right" width="384" src="assets/61-example.png"></a>

**Number:** 61 · **Color:** ![#C71585](assets/61-color.png) #C71585

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

## `]`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5D"><img align="right" width="384" src="assets/62-example.png"></a>

**Number:** 62 · **Color:** ![#00FA9A](assets/62-color.png) #00FA9A

**Function:** `left-half-offset` — Shift left half vertically by 20% with wraparound.

<br clear="right">

## `_`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw_"><img align="right" width="384" src="assets/63-example.png"></a>

**Number:** 63 · **Color:** ![#708090](assets/63-color.png) #708090

**Function:** `scanlines` — CRT scanline effect with darkening and displacement.

<br clear="right">

## `\``

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%60M"><img align="right" width="384" src="assets/64-example.png"></a>

**Number:** 64 · **Color:** ![#6495ED](assets/64-color.png) #6495ED

**Function:** `rule110` — Rule 110 cellular automaton - a Turing-complete 1D CA applied horizontally to each row.

**Arguments:**
   1. Number of generations (×8)

<br clear="right">

## `{`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%7BR"><img align="right" width="384" src="assets/65-example.png"></a>

**Number:** 65 · **Color:** ![#DC143C](assets/65-color.png) #DC143C

**Function:** `skew` — Skew horizontal with wraparound.

**Arguments:**
   1. Skew amount (A=left, 7=none, ~=right)

<br clear="right">

## `|`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1%7C%7C%7C"><img align="right" width="384" src="assets/66-example.png"></a>

**Number:** 66 · **Color:** ![#00BFFF](assets/66-color.png) #00BFFF

**Function:** `vertical-split` — Vertical split with wavy blend zone using multiple blend modes.

**Arguments:**
   1. Old image for right half

<br clear="right">

## `}`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%7D"><img align="right" width="384" src="assets/67-example.png"></a>

**Number:** 67 · **Color:** ![#9400D3](assets/67-color.png) #9400D3

**Function:** `gradientify` — Turns flat single-color areas into subtle gradients with hue shifts.

<br clear="right">

## `~`

<a href="https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw~F"><img align="right" width="384" src="assets/68-example.png"></a>

**Number:** 68 · **Color:** ![#FF6347](assets/68-color.png) #FF6347

**Function:** `wave-chromatic` — Horizontal wave distortion with chromatic aberration.

**Arguments:**
   1. Wave amplitude and chromatic shift

<br clear="right">

## License

MIT

## Author

Andreas Jansson ([@andreasjansson](https://github.com/andreasjansson))
