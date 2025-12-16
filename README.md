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

# Character Reference

## `A` (number 1, color <span style="color:#78A10F">#78A10F</span>)

[![Example for A](assets/01-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywA)

**Function:** `spheres` — Renders image as texture on two 3D spheres with lighting.

**Example:** `A`

---

## `B` (number 2, color <span style="color:#8B4513">#8B4513</span>)

[![Example for B](assets/02-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywBAA)

**Function:** `border` — Border effect with various shapes.

**Arguments:**
   1. Border shape style (A=circular-solid, B=circular-blur, C=horizontal-solid, D=horizontal-blur, E=vertical-solid, F=vertical-blur, G=rectangular-solid, H=rectangular-blur, ...)
   2. Border color

**Example:** `BAA`

---

## `C` (number 3, color <span style="color:#FF6B35">#FF6B35</span>)

[![Example for C](assets/03-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywCP)

**Function:** `concentric-hue` — Alternating original and hue-shifted concentric circles.

**Arguments:**
   1. Number of concentric circles

**Example:** `CP`

---

## `D` (number 4, color <span style="color:#FF1493">#FF1493</span>)

[![Example for D](assets/04-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywD)

**Function:** `drip` — Metaball-based dripping water drops effect.

**Example:** `D`

---

## `E` (number 5, color <span style="color:#50C878">#50C878</span>)

[![Example for E](assets/05-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywE)

**Function:** `emerald` — Renders reflective 3D emeralds in symmetric pattern.

**Example:** `E`

---

## `F` (number 6, color <span style="color:#FFD700">#FFD700</span>)

[![Example for F](assets/06-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywFB)

**Function:** `fft-overflow` — 2D FFT with magnitude overflow and chromatic phase shifts.

**Arguments:**
   1. FFT multiplier strength

**Example:** `FB`

---

## `G` (number 7, color <span style="color:#9370DB">#9370DB</span>)

[![Example for G](assets/07-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywGG)

**Function:** `grayscale-colorize` — Converts to grayscale then applies rainbow palette.

**Arguments:**
   1. Number of posterize colors

**Example:** `GG`

---

## `H` (number 8, color <span style="color:#DC143C">#DC143C</span>)

[![Example for H](assets/08-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywH)

**Function:** `hourglass` — Hourglass gradient with bitwise color blending.

**Example:** `H`

---

## `I` (number 9, color <span style="color:#00FF7F">#00FF7F</span>)

[![Example for I](assets/09-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywI)

**Function:** `invert-edges` — Inverts colors then adds Sobel edge detection.

**Example:** `I`

---

## `J` (number 10, color <span style="color:#FF8C00">#FF8C00</span>)

[![Example for J](assets/10-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywJM)

**Function:** `julia-fractal` — Julia set fractal masking the previous image.

**Arguments:**
   1. Fractal zoom depth

**Example:** `JM`

---

## `K` (number 11, color <span style="color:#9966FF">#9966FF</span>)

[![Example for K](assets/11-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywKE)

**Function:** `kaleidoscope` — N-way kaleidoscope effect with zoom.

**Arguments:**
   1. Number of kaleidoscope segments

**Example:** `KE`

---

## `L` (number 12, color <span style="color:#20B2AA">#20B2AA</span>)

[![Example for L](assets/12-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywLLL)

**Function:** `lissajous` — 3D Lissajous tube with textured surface.

**Arguments:**
   1. Old image for tube texture
   2. Rotation angle multiplier

**Example:** `LLL`

---

## `M` (number 13, color <span style="color:#FF69B4">#FF69B4</span>)

[![Example for M](assets/13-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywMC)

**Function:** `moire` — Moiré interference pattern with color zones.

**Arguments:**
   1. Pattern complexity seed

**Example:** `MC`

---

## `N` (number 14, color <span style="color:#8A2BE2">#8A2BE2</span>)

[![Example for N](assets/14-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywN)

**Function:** `neon` — Neon glow effect on bright edges.

**Example:** `N`

---

## `O` (number 15, color <span style="color:#FF6347">#FF6347</span>)

[![Example for O](assets/15-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywO9)

**Function:** `oil-slick` — Domain warping with iridescent lighting.

**Arguments:**
   1. Warp intensity and depth

**Example:** `O9`

---

## `P` (number 16, color <span style="color:#4682B4">#4682B4</span>)

[![Example for P](assets/16-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywP6)

**Function:** `pixelate` — Pixelate with diagonal split using average/saturated colors.

**Arguments:**
   1. Pixel cell size

**Example:** `P6`

---

## `Q` (number 17, color <span style="color:#32CD32">#32CD32</span>)

[![Example for Q](assets/17-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywQ)

**Function:** `quad-prism` — Negative prism with diagonal inversion and mirroring.

**Example:** `Q`

---

## `R` (number 18, color <span style="color:#DA70D6">#DA70D6</span>)

[![Example for R](assets/18-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywR)

**Function:** `room` — 3D room with textured walls, ceiling, and floor.

**Example:** `R`

---

## `S` (number 19, color <span style="color:#87CEEB">#87CEEB</span>)

[![Example for S](assets/19-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywSSS)

**Function:** `sierpinski` — Sierpiński triangle fractal with color effects.

**Arguments:**
   1. Old image for triangle interior
   2. Fractal detail level (A-~)

**Example:** `SSS`

---

## `T` (number 20, color <span style="color:#F0E68C">#F0E68C</span>)

[![Example for T](assets/20-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywT8)

**Function:** `tiles` — Grid of 3D tiles covering entire canvas, heights based on seed with multiplier.

**Arguments:**
   1. Building height multiplier (A=short, ~=tall)

**Example:** `T8`

---

## `U` (number 21, color <span style="color:#DDA0DD">#DDA0DD</span>)

[![Example for U](assets/21-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywUB)

**Function:** `dither` — Apply one of 16 dithering algorithms, from subtle to aggressive 2-color modes.

**Arguments:**
   1. Dithering algorithm (A=ordered-5level, B=bayer-bw, C=threshold-bw, D=ordered-2bit, E=floyd-rgb, F=floyd-bw, G=atkinson-4level, H=atkinson-bw, ...)

**Example:** `UB`

---

## `V` (number 22, color <span style="color:#40E0D0">#40E0D0</span>)

[![Example for V](assets/22-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw)~1AVBJ)

**Function:** `voronoi` — Voronoi cells alternating between current and old image, with variable pattern shape.

**Arguments:**
   1. Old image to alternate with
   2. Pattern shape

**Example:** `)~1AVBJ`

---

## `W` (number 23, color <span style="color:#EE82EE">#EE82EE</span>)

[![Example for W](assets/23-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywWW)

**Function:** `whirl` — Swirl distortion from center with quadratic falloff.

**Arguments:**
   1. Rotation multiplier (×20°)

**Example:** `WW`

---

## `X` (number 24, color <span style="color:#F5DEB3">#F5DEB3</span>)

[![Example for X](assets/24-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywX~)

**Function:** `cppn` — Compositional Pattern Producing Network warps and modulates saturation/value using a neural network. Fully deterministic based on input.

**Arguments:**
   1. Effect strength

**Example:** `X~`

---

## `Y` (number 25, color <span style="color:#98FB98">#98FB98</span>)

[![Example for Y](assets/25-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywYY)

**Function:** `yuv-shift` — YUV color shift with three gradient directions 120° apart: luminance, blue chrominance, and red chrominance shifts.

**Arguments:**
   1. Controls angle and intensity of color shifts

**Example:** `YY`

---

## `Z` (number 26, color <span style="color:#AFEEEE">#AFEEEE</span>)

[![Example for Z](assets/26-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywZK)

**Function:** `zoom-blur` — Radial motion blur from center with sharp center.

**Arguments:**
   1. Blur strength multiplier (×4px)

**Example:** `ZK`

---

## `0` (number 27, color <span style="color:#E6E6FA">#E6E6FA</span>)

[![Example for 0](assets/27-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw0A)

**Function:** `bg-remove` — Removes background from prev image using ML, composites on specified background.

**Arguments:**
   1. Background image to composite behind

**Example:** `0A`

---

## `1` (number 28, color <span style="color:#FFA07A">#FFA07A</span>)

[![Example for 1](assets/28-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw1E)

**Function:** `colorize` — Tints the image with the specified color using gamma-corrected luminance and boosted saturation.

**Arguments:**
   1. Tint color applied based on luminance

**Example:** `1E`

---

## `2` (number 29, color <span style="color:#98D8C8">#98D8C8</span>)

[![Example for 2](assets/29-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw2BAB)

**Function:** `third-stamp` — Replace a vertical third of current image with a third from old image.

**Arguments:**
   1. Old image to extract third from
   2. Old image third (1=left, 2=mid, 3=right, cycling)
   3. Current image third to replace (1=left, 2=mid, 3=right, cycling)

**Example:** `2BAB`

---

## `3` (number 30, color <span style="color:#F7DC6F">#F7DC6F</span>)

[![Example for 3](assets/30-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw3)

**Function:** `triple-rotate` — Three vertical strips with different rotations.

**Example:** `3`

---

## `4` (number 31, color <span style="color:#BB8FCE">#BB8FCE</span>)

[![Example for 4](assets/31-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw4)

**Function:** `quad-rotate` — Four quadrants each rotated 0°, 90°, 180°, 270°.

**Example:** `4`

---

## `5` (number 32, color <span style="color:#85C1E9">#85C1E9</span>)

[![Example for 5](assets/32-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw55)

**Function:** `triangular-split` — Triangular grid with hue shifts and lightness variation.

**Arguments:**
   1. Cell size multiplier

**Example:** `55`

---

## `6` (number 33, color <span style="color:#F1948A">#F1948A</span>)

[![Example for 6](assets/33-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw6)

**Function:** `posterize` — Posterize to 4 levels per channel.

**Example:** `6`

---

## `7` (number 34, color <span style="color:#82E0AA">#82E0AA</span>)

[![Example for 7](assets/34-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw777)

**Function:** `chromatic` — Chromatic aberration with RGB channel shifts.

**Example:** `777`

---

## `8` (number 35, color <span style="color:#F8C471">#F8C471</span>)

[![Example for 8](assets/35-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw8T)

**Function:** `lemniscate` — Infinity-loop lemniscate distortion.

**Arguments:**
   1. Distortion strength

**Example:** `8T`

---

## `9` (number 36, color <span style="color:#D7BDE2">#D7BDE2</span>)

[![Example for 9](assets/36-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywYY9B)

**Function:** `xor-blend` — XOR blend creating glitchy digital artifacts.

**Arguments:**
   1. Old image to XOR with

**Example:** `YY9B`

---

## `<` (number 37, color <span style="color:#E74C3C">#E74C3C</span>)

[![Example for <](assets/37-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3CJ)

**Function:** `horizontal-shift` — Horizontal shift with wraparound.

**Arguments:**
   1. Shift amount (A=left, 7=none, ~=right)

**Example:** `<J`

---

## `>` (number 38, color <span style="color:#3498DB">#3498DB</span>)

[![Example for >](assets/38-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3E)

**Function:** `rotate-90` — Rotate 90 degrees clockwise.

**Example:** `>`

---

## `^` (number 39, color <span style="color:#2ECC71">#2ECC71</span>)

[![Example for ^](assets/39-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5EV)

**Function:** `vertical-shift` — Vertical shift with wraparound.

**Arguments:**
   1. Shift amount (A=up, 7=none, ~=down)

**Example:** `^V`

---

## `!` (number 40, color <span style="color:#FF4500">#FF4500</span>)

[![Example for !](assets/40-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw!)

**Function:** `godrays` — Volumetric light scattering from center.

**Example:** `!`

---

## `"` (number 41, color <span style="color:#9932CC">#9932CC</span>)

[![Example for "](assets/41-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%22)

**Function:** `band-transform` — Horizontal bands with alternating hue/saturation transforms.

**Arguments:**
   1. Number of horizontal bands

**Example:** `"`

---

## `#` (number 42, color <span style="color:#228B22">#228B22</span>)

[![Example for #](assets/42-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%23B)

**Function:** `insert` — Replaces current image with specified old image.

**Arguments:**
   1. Old image index to insert

**Example:** `#B`

---

## `$` (number 43, color <span style="color:#FFD700">#FFD700</span>)

[![Example for $](assets/43-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%24)

**Function:** `segment-hue-sort` — Color-based segmentation, then sorts pixels by hue within each segment.

**Example:** `$`

---

## `%` (number 44, color <span style="color:#8B0000">#8B0000</span>)

[![Example for %](assets/44-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%25A)

**Function:** `flip` — Flips image horizontally or vertically based on argument parity.

**Arguments:**
   1. Flip direction (even=horizontal, odd=vertical)

**Example:** `%A`

---

## `&` (number 45, color <span style="color:#4169E1">#4169E1</span>)

[![Example for &](assets/45-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%26P)

**Function:** `quadtree-compress` — Adaptive quadtree compression - detailed areas keep resolution while uniform areas become large blocks, creating geometric patterns.

**Arguments:**
   1. Compression level (A=minimal/detailed, ~=maximal/geometric blocks)

**Example:** `&P`

---

## `'` (number 46, color <span style="color:#FF1493">#FF1493</span>)

[![Example for '](assets/46-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1%27%27B)

**Function:** `variable-checkerboard` — Checkerboard blend with increasing square size from corner to corner.

**Arguments:**
   1. Old image to checkerboard with

**Example:** `))1''B`

---

## `(` (number 47, color <span style="color:#00CED1">#00CED1</span>)

[![Example for (](assets/47-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw(3F%26)

**Function:** `shear-radial` — Combined shear and radial distortion. Shear amount couples to horizontal offset and radial strength.

**Arguments:**
   1. Center X offset (A=left, M=center, ~=right)
   2. Center Y offset (A=bottom, M=center, ~=top)
   3. Radial/shear strength (A=barrel, M=none, ~=pincushion)

**Example:** `(3F&`

---

## `)` (number 48, color <span style="color:#FF69B4">#FF69B4</span>)

[![Example for )](assets/48-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw)))

**Function:** `blur` — Gaussian blur with adjustable radius using two-pass convolution.

**Arguments:**
   1. Blur radius (A=subtle, ~=heavy)

**Example:** `))`

---

## `*` (number 49, color <span style="color:#FFD700">#FFD700</span>)

[![Example for *](assets/49-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw*)

**Function:** `fur` — Fur/hair strands growing from pixels based on hue and noise.

**Example:** `*`

---

## `+` (number 50, color <span style="color:#32CD32">#32CD32</span>)

[![Example for +](assets/50-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2B%2B%2B%2B%2B%2B%2B)

**Function:** `zoom` — Zoom in 1.2× from center.

**Example:** `+++++++`

---

## `,` (number 51, color <span style="color:#BA55D3">#BA55D3</span>)

[![Example for ,](assets/51-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2CD)

**Function:** `stipple` — Stipple dots at luminance-based positions.

**Arguments:**
   1. Stipple dot color

**Example:** `,D`

---

## `-` (number 52, color <span style="color:#FF7F50">#FF7F50</span>)

[![Example for -](assets/52-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1D-B6)

**Function:** `blend` — Blend old image with current using specified mode.

**Arguments:**
   1. Old image to blend with
   2. Blend mode (A=multiply, B=screen, C=overlay, D=darken, E=lighten, F=dodge, G=burn, H=hardlight, ...)

**Example:** `))1D-B6`

---

## `.` (number 53, color <span style="color:#20B2AA">#20B2AA</span>)

[![Example for .](assets/53-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw.M)

**Function:** `pointillism` — Pointillism effect with saturated circular dots.

**Arguments:**
   1. Dot radius base (mod 8 + 2)

**Example:** `.M`

---

## `/` (number 54, color <span style="color:#CD853F">#CD853F</span>)

[![Example for /](assets/54-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%2FB073C)

**Function:** `circle-stamp` — Stamp circular region from old image center onto current.

**Arguments:**
   1. Old image source
   2. X position (A=left, 7=center, ~=right)
   3. Y position (A=top, 7=center, ~=bottom)
   4. Circle size (A=tiny, ~=full)
   5. Blend mode (A=normal, B=xor, C=nand, D=and, E=or, F=multiply, G=screen, H=overlay, ...)

**Example:** `/B073C`

---

## `:` (number 55, color <span style="color:#6B8E23">#6B8E23</span>)

[![Example for :](assets/55-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw1D%3AB)

**Function:** `porthole` — Circular window showing current image with old image as background.

**Arguments:**
   1. Old image for background

**Example:** `1D:B`

---

## `;` (number 56, color <span style="color:#DB7093">#DB7093</span>)

[![Example for ;](assets/56-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3B)

**Function:** `semicircle-reflect` — Top semicircle preserved, bottom reflected with wave distortion.

**Example:** `;`

---

## `=` (number 57, color <span style="color:#5F9EA0">#5F9EA0</span>)

[![Example for =](assets/57-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3DX)

**Function:** `shifted-stripes` — Horizontal stripes with alternating shifts.

**Arguments:**
   1. Stripe height in pixels

**Example:** `=X`

---

## `?` (number 58, color <span style="color:#D2691E">#D2691E</span>)

[![Example for ?](assets/58-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%3F%3F)

**Function:** `help` — Display help text or image history table.

**Arguments:**
   1. Page number (A=intro, B+=reference, #=history)

**Example:** `??`

---

## `@` (number 59, color <span style="color:#7B68EE">#7B68EE</span>)

[![Example for @](assets/59-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIywGA%23B1.%23B1C%40CEGAE)

**Function:** `cond` — Per-pixel conditional: extracts channel from condition image, outputs true-image pixel where value >= threshold, otherwise false-image pixel.

**Arguments:**
   1. Condition image sampled for threshold comparison
   2. Source image when condition >= threshold
   3. Source image when condition < threshold
   4. Color channel to extract from condition image (A=hue, B=saturation, C=lightness, D=red, E=green, F=blue)
   5. Threshold (A=0%, ~=100% of channel range)

**Example:** `GA#B1.#B1C@CEGAE`

---

## `[` (number 60, color <span style="color:#48D1CC">#48D1CC</span>)

[![Example for [](assets/60-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5BX)

**Function:** `rotate` — Rotate around center.

**Arguments:**
   1. Rotation amount (A=left, 7=none, ~=right)

**Example:** `[X`

---

## `\\` (number 61, color <span style="color:#C71585">#C71585</span>)

[![Example for \\](assets/61-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5CB4VXMVH-0%7DA)

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

**Example:** `\B4VXMVH-0}A`

---

## `]` (number 62, color <span style="color:#00FA9A">#00FA9A</span>)

[![Example for ]](assets/62-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%5D)

**Function:** `left-half-offset` — Shift left half vertically by 20% with wraparound.

**Example:** `]`

---

## `_` (number 63, color <span style="color:#708090">#708090</span>)

[![Example for _](assets/63-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw_)

**Function:** `scanlines` — CRT scanline effect with darkening and displacement.

**Example:** `_`

---

## `\`` (number 64, color <span style="color:#6495ED">#6495ED</span>)

[![Example for \`](assets/64-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%60M)

**Function:** `rule110` — Rule 110 cellular automaton - a Turing-complete 1D CA applied horizontally to each row.

**Arguments:**
   1. Number of generations (×8)

**Example:** ``M`

---

## `{` (number 65, color <span style="color:#DC143C">#DC143C</span>)

[![Example for {](assets/65-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%7BR)

**Function:** `skew` — Skew horizontal with wraparound.

**Arguments:**
   1. Skew amount (A=left, 7=none, ~=right)

**Example:** `{R`

---

## `|` (number 66, color <span style="color:#00BFFF">#00BFFF</span>)

[![Example for |](assets/66-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw))1%7C%7C%7C)

**Function:** `vertical-split` — Vertical split with wavy blend zone using multiple blend modes.

**Arguments:**
   1. Old image for right half

**Example:** `))1|||`

---

## `}` (number 67, color <span style="color:#9400D3">#9400D3</span>)

[![Example for }](assets/67-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw%7D)

**Function:** `sharpen` — Sharpen using convolution kernel to enhance edges.

**Example:** `}`

---

## `~` (number 68, color <span style="color:#FF6347">#FF6347</span>)

[![Example for ~](assets/68-example.png)](https://qqqlang.com/?p=%E2%98%80Lh8lX-CEM_8ykW3QtaeIyw~F)

**Function:** `wave-chromatic` — Horizontal wave distortion with chromatic aberration.

**Arguments:**
   1. Wave amplitude and chromatic shift

**Example:** `~F`

---

## License

MIT

## Author

Andreas Jansson ([@andreasjansson](https://github.com/andreasjansson))
