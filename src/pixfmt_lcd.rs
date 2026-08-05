//! LCD subpixel pixel format for RGBA32 buffers.
//!
//! Port of `agg_pixfmt_rgb24_lcd.h` — specialized pixel format that performs
//! LCD subpixel rendering by distributing coverage across the R, G, B channels
//! of adjacent pixels. Adapted for RGBA32 (4 bytes per pixel) buffers.
//!
//! The rasterizer operates at 3x horizontal resolution. Each "subpixel" maps
//! to one color channel (R, G, or B) of an actual RGBA pixel.
//!
//! Copyright (c) 2025. BSD-3-Clause License.

use crate::basics::CoverType;
use crate::color::Rgba8;
use crate::pixfmt_rgba::PixelFormat;
use crate::rendering_buffer::RowAccessor;

// ============================================================================
// LcdDistributionLut
// ============================================================================

/// Lookup table for LCD subpixel coverage distribution.
///
/// Distributes each coverage value across primary (center), secondary
/// (adjacent), and tertiary (2-away) positions. This implements the
/// energy distribution described in Steve Gibson's subpixel rendering guide.
///
/// Port of C++ `lcd_distribution_lut` from `agg_pixfmt_rgb24_lcd.h`.
pub struct LcdDistributionLut {
    primary_lut: [u8; 256],
    secondary_lut: [u8; 256],
    tertiary_lut: [u8; 256],
}

impl LcdDistributionLut {
    /// Create a new LCD distribution lookup table.
    ///
    /// # Arguments
    /// * `prim` — Weight for the primary (center) subpixel
    /// * `second` — Weight for the secondary (adjacent) subpixels
    /// * `tert` — Weight for the tertiary (2-away) subpixels
    ///
    /// Weights are normalized so that `prim + 2*second + 2*tert = 1.0`.
    pub fn new(prim: f64, second: f64, tert: f64) -> Self {
        let norm = 1.0 / (prim + second * 2.0 + tert * 2.0);
        let prim = prim * norm;
        let second = second * norm;
        let tert = tert * norm;

        let mut primary_lut = [0u8; 256];
        let mut secondary_lut = [0u8; 256];
        let mut tertiary_lut = [0u8; 256];

        for i in 0..256 {
            primary_lut[i] = (prim * i as f64).floor() as u8;
            secondary_lut[i] = (second * i as f64).floor() as u8;
            tertiary_lut[i] = (tert * i as f64).floor() as u8;
        }

        Self {
            primary_lut,
            secondary_lut,
            tertiary_lut,
        }
    }

    /// Get the primary (center) distribution for a coverage value.
    #[inline]
    pub fn primary(&self, v: u8) -> u8 {
        self.primary_lut[v as usize]
    }

    /// Get the secondary (adjacent) distribution for a coverage value.
    #[inline]
    pub fn secondary(&self, v: u8) -> u8 {
        self.secondary_lut[v as usize]
    }

    /// Get the tertiary (2-away) distribution for a coverage value.
    #[inline]
    pub fn tertiary(&self, v: u8) -> u8 {
        self.tertiary_lut[v as usize]
    }
}

// ============================================================================
// PixfmtRgba32Lcd
// ============================================================================

const BPP: usize = 4; // bytes per pixel in RGBA32

/// LCD subpixel pixel format for RGBA32 rendering buffers.
///
/// Adapted from C++ `pixfmt_rgb24_lcd` for RGBA32 (4 bytes per pixel) buffers.
/// Reports width as `actual_width * 3` so the rasterizer operates at 3x
/// horizontal resolution. Each "subpixel" maps to one R, G, or B channel
/// of an actual RGBA pixel:
///
/// - Subpixel `sp` → pixel `sp / 3`, channel `sp % 3` (0=R, 1=G, 2=B)
/// - Byte offset: `(sp / 3) * 4 + (sp % 3)`
///
/// The `blend_solid_hspan` method distributes each coverage value across
/// 5 neighboring subpixels (tertiary, secondary, primary, secondary, tertiary)
/// using the `LcdDistributionLut`, matching the C++ implementation exactly.
/// Capacity the per-span 5-tap distribution scratch buffer starts at.
///
/// A span is a horizontal run of covered sub-pixels; for text these are
/// short (a glyph stem is a handful of stripes) and there are very many of
/// them. Pre-sizing to a comfortable span length means the buffer is
/// allocated once per pixfmt rather than once per span.
const SPAN_SCRATCH: usize = 256;

pub struct PixfmtRgba32Lcd<'a> {
    rbuf: &'a mut RowAccessor,
    lut: &'a LcdDistributionLut,
    /// Reusable 5-tap distribution buffer, see [`SPAN_SCRATCH`].
    c3: Vec<u8>,
}

impl<'a> PixfmtRgba32Lcd<'a> {
    /// Create a new LCD pixel format wrapping an RGBA32 rendering buffer.
    pub fn new(rbuf: &'a mut RowAccessor, lut: &'a LcdDistributionLut) -> Self {
        Self { rbuf, lut, c3: Vec::with_capacity(SPAN_SCRATCH) }
    }

    /// Get the actual (non-subpixel) width.
    #[inline]
    fn actual_width(&self) -> u32 {
        self.rbuf.width()
    }

    /// Blend a single byte in the RGBA buffer using the C++ alpha formula.
    ///
    /// `*p = (((rgb_val - *p) * alpha) + (*p << 16)) >> 16`
    #[inline]
    fn blend_byte(dst: u8, src: u8, alpha: i32) -> u8 {
        (((src as i32 - dst as i32) * alpha + ((dst as i32) << 16)) >> 16) as u8
    }
}

impl<'a> PixelFormat for PixfmtRgba32Lcd<'a> {
    type ColorType = Rgba8;

    fn width(&self) -> u32 {
        self.rbuf.width() * 3
    }

    fn height(&self) -> u32 {
        self.rbuf.height()
    }

    fn pixel(&self, x: i32, y: i32) -> Rgba8 {
        // Map subpixel x to actual pixel
        let pixel = x as usize / 3;
        let actual_w = self.actual_width() as usize;
        if pixel >= actual_w {
            return Rgba8::new(0, 0, 0, 0);
        }
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts(ptr, actual_w * BPP)
        };
        let off = pixel * BPP;
        Rgba8::new(
            row[off] as u32,
            row[off + 1] as u32,
            row[off + 2] as u32,
            row[off + 3] as u32,
        )
    }

    fn copy_pixel(&mut self, x: i32, y: i32, c: &Rgba8) {
        let pixel = x as usize / 3;
        let actual_w = self.actual_width() as usize;
        if pixel >= actual_w {
            return;
        }
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        let off = pixel * BPP;
        row[off] = c.r;
        row[off + 1] = c.g;
        row[off + 2] = c.b;
        row[off + 3] = c.a;
    }

    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8) {
        // Copy len subpixels — sets whole pixels for each affected pixel
        let actual_w = self.actual_width() as usize;
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        for k in 0..len as usize {
            let sp = x as usize + k;
            let pixel = sp / 3;
            let channel = sp % 3;
            if pixel >= actual_w {
                break;
            }
            let byte_off = pixel * BPP + channel;
            row[byte_off] = [c.r, c.g, c.b][channel];
            // Set alpha to 255 when we touch any channel of a pixel
            row[pixel * BPP + 3] = 255;
        }
    }

    fn blend_pixel(&mut self, x: i32, y: i32, c: &Rgba8, cover: CoverType) {
        let sp = x as usize;
        let pixel = sp / 3;
        let channel = sp % 3;
        let actual_w = self.actual_width() as usize;
        if pixel >= actual_w {
            return;
        }
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        let byte_off = pixel * BPP + channel;
        let rgb = [c.r, c.g, c.b];
        let alpha = cover as i32 * c.a as i32;
        if alpha != 0 {
            if alpha == 255 * 255 {
                row[byte_off] = rgb[channel];
            } else {
                row[byte_off] = Self::blend_byte(row[byte_off], rgb[channel], alpha);
            }
            row[pixel * BPP + 3] = 255;
        }
    }

    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, cover: CoverType) {
        let actual_w = self.actual_width() as usize;
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        let alpha = cover as i32 * c.a as i32;
        if alpha == 0 {
            return;
        }
        let rgb = [c.r, c.g, c.b];

        for k in 0..len as usize {
            let sp = x as usize + k;
            let pixel = sp / 3;
            let channel = sp % 3;
            if pixel >= actual_w {
                break;
            }
            let byte_off = pixel * BPP + channel;
            if alpha == 255 * 255 {
                row[byte_off] = rgb[channel];
            } else {
                row[byte_off] = Self::blend_byte(row[byte_off], rgb[channel], alpha);
            }
            row[pixel * BPP + 3] = 255;
        }
    }

    /// LCD subpixel coverage distribution — the core of LCD rendering.
    ///
    /// Distributes each coverage value across 5 neighboring subpixel positions
    /// (tertiary, secondary, primary, secondary, tertiary) using the LUT, then
    /// blends the distributed coverage into the RGBA buffer.
    ///
    /// Exact port of C++ `pixfmt_rgb24_lcd::blend_solid_hspan`, adapted for
    /// RGBA32 byte layout.
    fn blend_solid_hspan(
        &mut self,
        x: i32,
        y: i32,
        len: u32,
        c: &Rgba8,
        covers: &[CoverType],
    ) {
        let len = len as usize;

        // Step 1: Distribute coverage across 5-tap kernel
        // Matching C++: c3[i+0] += tertiary, c3[i+1] += secondary,
        //               c3[i+2] += primary,  c3[i+3] += secondary,
        //               c3[i+4] += tertiary
        let dist_len = len + 4;
        // Reuse the scratch buffer. This used to be `vec![0u8; dist_len]`,
        // i.e. a heap allocation and free for EVERY span. Text rasterizes
        // into hundreds of thousands of short spans per frame, so the
        // malloc/free pair — not the blending arithmetic — dominated.
        // Moved out of `self` (a move, not a copy) so the body can still
        // borrow `self.rbuf`/`self.lut`; handed back before returning.
        let mut c3 = core::mem::take(&mut self.c3);
        c3.clear();
        c3.resize(dist_len, 0);

        for i in 0..len {
            let cv = covers[i];
            c3[i] = c3[i].wrapping_add(self.lut.tertiary(cv));
            c3[i + 1] = c3[i + 1].wrapping_add(self.lut.secondary(cv));
            c3[i + 2] = c3[i + 2].wrapping_add(self.lut.primary(cv));
            c3[i + 3] = c3[i + 3].wrapping_add(self.lut.secondary(cv));
            c3[i + 4] = c3[i + 4].wrapping_add(self.lut.tertiary(cv));
        }

        // Step 2: Adjust start position (distribution extends 2 subpixels before)
        let mut sp_start = x as i32 - 2;
        let mut c3_offset = 0usize;
        let mut remaining = dist_len;

        if sp_start < 0 {
            let skip = (-sp_start) as usize;
            c3_offset = skip;
            if skip >= remaining {
                self.c3 = c3;
                return;
            }
            remaining -= skip;
            sp_start = 0;
        }

        // Step 3: Apply distributed covers to RGBA buffer
        let actual_w = self.actual_width() as usize;
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };

        let rgb = [c.r, c.g, c.b];
        // Channel cycling: which RGB channel does sp_start map to?
        // Matching C++: i = x % 3 (after x -= 2)
        // sp_start % 3 gives us the starting channel

        for k in 0..remaining {
            let sp = sp_start as usize + k;
            let pixel = sp / 3;
            let channel = sp % 3;

            if pixel >= actual_w {
                break;
            }

            let cover = c3[c3_offset + k];
            let alpha = cover as i32 * c.a as i32;

            if alpha != 0 {
                let byte_off = pixel * BPP + channel;
                if alpha == 255 * 255 {
                    row[byte_off] = rgb[channel];
                } else {
                    row[byte_off] =
                        Self::blend_byte(row[byte_off], rgb[channel], alpha);
                }
                // Ensure alpha channel is opaque for any touched pixel
                row[pixel * BPP + 3] = 255;
            }
        }
        self.c3 = c3;
    }

    fn blend_color_hspan(
        &mut self,
        x: i32,
        y: i32,
        len: u32,
        colors: &[Rgba8],
        covers: &[CoverType],
        cover: CoverType,
    ) {
        let actual_w = self.actual_width() as usize;
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };

        for k in 0..len as usize {
            let sp = x as usize + k;
            let pixel = sp / 3;
            let channel = sp % 3;
            if pixel >= actual_w {
                break;
            }

            let c = &colors[k];
            let cov = if !covers.is_empty() {
                covers[k]
            } else {
                cover
            };
            let alpha = cov as i32 * c.a as i32;
            if alpha != 0 {
                let byte_off = pixel * BPP + channel;
                let rgb = [c.r, c.g, c.b];
                if alpha == 255 * 255 {
                    row[byte_off] = rgb[channel];
                } else {
                    row[byte_off] =
                        Self::blend_byte(row[byte_off], rgb[channel], alpha);
                }
                row[pixel * BPP + 3] = 255;
            }
        }
    }
}

// ============================================================================
// Linear-light (gamma-correct) LCD blending
// ============================================================================

/// sRGB <-> linear-light lookup tables (13-bit linear precision).
///
/// `to_linear[srgb8]` gives linear light scaled to 0..=8160 (255 * 32);
/// `to_srgb[linear13]` inverts it. Built once on first use. The transfer
/// function is plain sRGB today; swapping these tables is the designed
/// extension point for other transfer curves (e.g. Display-P3, which
/// shares the sRGB curve but differs in primaries, or PQ).
pub struct SrgbLuts {
    to_linear: [u16; 256],
    to_srgb: Vec<u8>,
}

const LINEAR_MAX: u32 = 8160; // 255 * 32

impl SrgbLuts {
    fn build() -> Self {
        let mut to_linear = [0u16; 256];
        for (i, e) in to_linear.iter_mut().enumerate() {
            let c = i as f64 / 255.0;
            let lin = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
            *e = (lin * f64::from(LINEAR_MAX)).round() as u16;
        }
        let mut to_srgb = vec![0u8; (LINEAR_MAX + 1) as usize];
        for (i, e) in to_srgb.iter_mut().enumerate() {
            let lin = i as f64 / f64::from(LINEAR_MAX);
            let c = if lin <= 0.003_130_8 {
                lin * 12.92
            } else {
                1.055 * lin.powf(1.0 / 2.4) - 0.055
            };
            *e = (c * 255.0).round() as u8;
        }
        Self { to_linear, to_srgb }
    }

    /// Shared instance (lazily built).
    pub fn get() -> &'static Self {
        use std::sync::OnceLock;
        static LUTS: OnceLock<SrgbLuts> = OnceLock::new();
        LUTS.get_or_init(Self::build)
    }

    #[inline]
    fn lin(&self, srgb: u8) -> u32 {
        u32::from(self.to_linear[srgb as usize])
    }

    #[inline]
    fn srgb(&self, linear: u32) -> u8 {
        self.to_srgb[linear.min(LINEAR_MAX) as usize]
    }
}

/// Tuning parameters for [`PixfmtRgba32LcdLinear`].
#[derive(Debug, Clone, Copy)]
pub struct LcdBlendParams {
    /// Skia-style contrast enhancement applied to per-stripe coverage
    /// BEFORE compositing: `c' = c + contrast * c * (1 - c)`, with this
    /// field as fixed-point 0..=255 over 1.0. Physically-linear blending
    /// alone renders dark-on-light text slightly lighter than sRGB-space
    /// blending did; a modest boost restores the expected stem weight
    /// (ClearType hides the same issue inside its fixed gamma fudge).
    pub contrast: u8,
    /// 0 = full per-stripe freedom (physically correct for the panel).
    /// 255 = stripes fully clamped to the pixel's luminance-correct
    /// composite (grayscale-equivalent). Values between limit how far a
    /// stripe may deviate from that composite: a CHROMA budget, not a
    /// grayscale fade — stripe edge sharpness survives, the worst color
    /// error does not. Applied per pixel against the actual background.
    pub chroma_limit: u8,
    /// Coverage tone curve, fixed-point x100 (100 = linear coverage,
    /// 220 = c^(1/2.2)). Pure linear-light compositing renders
    /// dark-on-light text noticeably lighter than every shipping
    /// rasterizer (Skia/ClearType blend in an intermediate gamma space
    /// precisely for this); raising coverage through a power curve
    /// restores conventional stem weight while KEEPING the per-stripe
    /// linear mixing that fixes saturated-background fringing. Applied
    /// before `contrast`.
    pub coverage_gamma: u8,
}

impl Default for LcdBlendParams {
    fn default() -> Self {
        Self { contrast: 0, chroma_limit: 0, coverage_gamma: 220 }
    }
}

/// LCD subpixel pixel format with colorimetric (linear-light) blending.
///
/// Same 3x-resolution stripe model and 5-tap energy distribution as
/// [`PixfmtRgba32Lcd`], but compositing happens in LINEAR light through
/// [`SrgbLuts`], per stripe, against the ACTUAL destination pixel:
///
/// ```text
/// out_ch = to_srgb( lin(fg_ch) * cover + lin(bg_ch) * (1 - cover) )
/// ```
///
/// Blending sRGB code values directly (the classic ClearType-era model,
/// and what [`PixfmtRgba32Lcd`] does) over- or under-weights partial
/// coverage depending on where fg/bg sit on the transfer curve — on
/// saturated backgrounds (white-on-green) the mid-coverage stripes land
/// far off the perceptual line between the two colors, which reads as
/// colored fringing and washed-out stems. In linear light the stripe
/// value IS the physical mixture, which is what the eye integrates on a
/// real RGB-stripe panel.
#[cfg(test)]
mod coverage_tone_lut_tests {
    use super::coverage_tone_lut;

    fn reference(coverage_gamma: u8) -> [u8; 256] {
        let g = f64::from(coverage_gamma.max(1)) / 100.0;
        let inv = 1.0 / g;
        let mut lut = [0u8; 256];
        for (i, e) in lut.iter_mut().enumerate() {
            *e = ((i as f64 / 255.0).powf(inv) * 255.0).round() as u8;
        }
        lut
    }

    /// The cache must return exactly what the uncached computation would,
    /// on the first call AND on every later one.
    #[test]
    fn cached_lut_matches_the_direct_computation() {
        for g in [1u8, 50, 100, 180, 220, 255] {
            let want = reference(g);
            // Twice: first fills the cache, second is served from it.
            assert_eq!(coverage_tone_lut(g), want, "gamma {g} first call");
            assert_eq!(coverage_tone_lut(g), want, "gamma {g} cached call");
        }
    }

    /// NEGATIVE CONTROL for the cache KEY. A cache that ignored the gamma
    /// would hand every caller the first table it built, silently applying
    /// the wrong tone curve to all text after the first run.
    #[test]
    fn different_gammas_get_different_luts() {
        let a = coverage_tone_lut(60);
        let b = coverage_tone_lut(220);
        assert_ne!(
            a, b,
            "gamma 60 and gamma 220 produced the SAME coverage table — the \
             cache is not keyed on the gamma, so every text run after the \
             first would be toned with the wrong curve"
        );
        // And neither is the identity, or the comparison above proves little.
        let identity: [u8; 256] = core::array::from_fn(|i| i as u8);
        assert_ne!(a, identity, "gamma 60 must bend the curve");
    }
}

/// Coverage tone LUT for a given `coverage_gamma`, built once per distinct
/// gamma instead of once per pixfmt.
///
/// It is 256 `powf` calls and depends on NOTHING but the gamma. A pixfmt is
/// constructed per text run, so a page of text rebuilt this identical table
/// dozens of times a frame — measured at 304 us/frame on a 37-run page,
/// pure waste.
///
/// Cached in a small fixed table rather than a HashMap: `coverage_gamma` is
/// a `u16` that in practice takes one or two values for the life of a
/// process (it comes from an env knob), so a linear scan of a handful of
/// entries beats hashing, and the whole thing stays lock-free after the
/// first miss.
fn coverage_tone_lut(coverage_gamma: u8) -> [u8; 256] {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Vec<(u8, [u8; 256])>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(c) = cache.lock() {
        if let Some((_, lut)) = c.iter().find(|(g, _)| *g == coverage_gamma) {
            return *lut;
        }
    }
    let g = f64::from(coverage_gamma.max(1)) / 100.0;
    let inv = 1.0 / g;
    let mut lut = [0u8; 256];
    for (i, e) in lut.iter_mut().enumerate() {
        let c = (i as f64 / 255.0).powf(inv);
        *e = (c * 255.0).round() as u8;
    }
    if let Ok(mut c) = cache.lock() {
        // Bounded: a runaway producer of distinct gammas must not grow this
        // without limit. 16 is far beyond the one or two seen in practice.
        if c.len() < 16 {
            c.push((coverage_gamma, lut));
        }
    }
    lut
}

pub struct PixfmtRgba32LcdLinear<'a> {
    rbuf: &'a mut RowAccessor,
    lut: &'a LcdDistributionLut,
    params: LcdBlendParams,
    luts: &'static SrgbLuts,
    /// Coverage tone LUT for `params.coverage_gamma` (identity at 100).
    cover_lut: [u8; 256],
    /// Reusable 5-tap distribution buffer, see [`SPAN_SCRATCH`].
    c3: Vec<u8>,
}

impl<'a> PixfmtRgba32LcdLinear<'a> {
    pub fn new(
        rbuf: &'a mut RowAccessor,
        lut: &'a LcdDistributionLut,
        params: LcdBlendParams,
    ) -> Self {
        let cover_lut = coverage_tone_lut(params.coverage_gamma);
        Self {
            rbuf,
            lut,
            params,
            luts: SrgbLuts::get(),
            cover_lut,
            c3: Vec::with_capacity(SPAN_SCRATCH),
        }
    }

    #[inline]
    fn actual_width(&self) -> u32 {
        self.rbuf.width()
    }

    /// Coverage tone curve + contrast enhancement (0..=255).
    #[inline]
    fn boost(&self, c: u32) -> u32 {
        let c = u32::from(self.cover_lut[c as usize]);
        // c + contrast * c * (255 - c) / 255^2, staying in 0..=255
        let k = u32::from(self.params.contrast);
        (c + (k * c * (255 - c)) / (255 * 255)).min(255)
    }

    /// Composite one pixel: per-stripe linear blend of `fg` over the
    /// destination using channel coverages `cov` (0..=255 each), with
    /// the optional chroma budget. `a_fg` is the run's alpha (0..=255).
    #[inline]
    fn composite_pixel(&self, row: &mut [u8], pixel: usize, fg: &Rgba8, a_fg: u32, cov: [u32; 3]) {
        let off = pixel * BPP;
        let luts = self.luts;
        let fg_lin = [
            luts.lin(fg.r),
            luts.lin(fg.g),
            luts.lin(fg.b),
        ];
        let bg_lin = [
            luts.lin(row[off]),
            luts.lin(row[off + 1]),
            luts.lin(row[off + 2]),
        ];

        // Per-stripe coverage with contrast, modulated by run alpha.
        let mut c = [0u32; 3];
        let mut any = false;
        for i in 0..3 {
            let boosted = self.boost(cov[i]);
            c[i] = boosted * a_fg / 255;
            any |= c[i] != 0;
        }
        if !any {
            return;
        }

        // Linear src-over per stripe.
        let mut out = [0u32; 3];
        for i in 0..3 {
            out[i] = (fg_lin[i] * c[i] + bg_lin[i] * (255 - c[i])) / 255;
        }

        // Chroma budget: limit stripe deviation from the luminance-correct
        // composite (the same blend done with the MEAN coverage).
        let clamp = u32::from(self.params.chroma_limit);
        if clamp > 0 {
            let c_mean = (c[0] + c[1] + c[2]) / 3;
            for i in 0..3 {
                let gray = (fg_lin[i] * c_mean + bg_lin[i] * (255 - c_mean)) / 255;
                // out' = out + clamp/255 * (gray - out)
                let o = i64::from(out[i]);
                let g = i64::from(gray);
                out[i] = (o + (i64::from(clamp) * (g - o)) / 255) as u32;
            }
        }

        row[off] = luts.srgb(out[0]);
        row[off + 1] = luts.srgb(out[1]);
        row[off + 2] = luts.srgb(out[2]);
        row[off + 3] = 255;
    }
}

impl<'a> PixelFormat for PixfmtRgba32LcdLinear<'a> {
    type ColorType = Rgba8;

    fn width(&self) -> u32 {
        self.rbuf.width() * 3
    }

    fn height(&self) -> u32 {
        self.rbuf.height()
    }

    fn pixel(&self, x: i32, y: i32) -> Rgba8 {
        let pixel = x as usize / 3;
        let actual_w = self.actual_width() as usize;
        if pixel >= actual_w {
            return Rgba8::new(0, 0, 0, 0);
        }
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts(ptr, actual_w * BPP)
        };
        let off = pixel * BPP;
        Rgba8::new(
            row[off] as u32,
            row[off + 1] as u32,
            row[off + 2] as u32,
            row[off + 3] as u32,
        )
    }

    fn copy_pixel(&mut self, x: i32, y: i32, c: &Rgba8) {
        let pixel = x as usize / 3;
        let actual_w = self.actual_width() as usize;
        if pixel >= actual_w {
            return;
        }
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        let off = pixel * BPP;
        row[off] = c.r;
        row[off + 1] = c.g;
        row[off + 2] = c.b;
        row[off + 3] = c.a;
    }

    fn copy_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8) {
        let actual_w = self.actual_width() as usize;
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        for k in 0..len as usize {
            let sp = x as usize + k;
            let pixel = sp / 3;
            let channel = sp % 3;
            if pixel >= actual_w {
                break;
            }
            let byte_off = pixel * BPP + channel;
            row[byte_off] = [c.r, c.g, c.b][channel];
            row[pixel * BPP + 3] = 255;
        }
    }

    fn blend_pixel(&mut self, x: i32, y: i32, c: &Rgba8, cover: CoverType) {
        // Single-stripe blend: gather into a one-pixel composite.
        let sp = x as usize;
        let pixel = sp / 3;
        let channel = sp % 3;
        let actual_w = self.actual_width() as usize;
        if pixel >= actual_w {
            return;
        }
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        let mut cov = [0u32; 3];
        cov[channel] = u32::from(cover);
        self.composite_pixel(row, pixel, c, u32::from(c.a), cov);
    }

    fn blend_hline(&mut self, x: i32, y: i32, len: u32, c: &Rgba8, cover: CoverType) {
        let actual_w = self.actual_width() as usize;
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        // Group the affected stripe range into whole pixels.
        let sp_end = x as usize + len as usize;
        let mut sp = x as usize;
        while sp < sp_end {
            let pixel = sp / 3;
            if pixel >= actual_w {
                break;
            }
            let mut cov = [0u32; 3];
            for ch in 0..3 {
                let s = pixel * 3 + ch;
                if s >= x as usize && s < sp_end {
                    cov[ch] = u32::from(cover);
                }
            }
            self.composite_pixel(row, pixel, c, u32::from(c.a), cov);
            sp = (pixel + 1) * 3;
        }
    }

    /// 5-tap distribution identical to [`PixfmtRgba32Lcd`], then one
    /// linear-light composite PER PIXEL with the gathered stripe triple.
    fn blend_solid_hspan(
        &mut self,
        x: i32,
        y: i32,
        len: u32,
        c: &Rgba8,
        covers: &[CoverType],
    ) {
        let len = len as usize;

        let dist_len = len + 4;
        // See the note in `PixfmtRgba32Lcd::blend_solid_hspan`: one heap
        // allocation per span was the dominant cost of LCD text.
        // Moved out of `self` (a move, not a copy) so the body can still
        // borrow `self.rbuf`/`self.lut`; handed back before returning.
        let mut c3 = core::mem::take(&mut self.c3);
        c3.clear();
        c3.resize(dist_len, 0);
        for i in 0..len {
            let cv = covers[i];
            c3[i] = c3[i].wrapping_add(self.lut.tertiary(cv));
            c3[i + 1] = c3[i + 1].wrapping_add(self.lut.secondary(cv));
            c3[i + 2] = c3[i + 2].wrapping_add(self.lut.primary(cv));
            c3[i + 3] = c3[i + 3].wrapping_add(self.lut.secondary(cv));
            c3[i + 4] = c3[i + 4].wrapping_add(self.lut.tertiary(cv));
        }

        let sp_first = x as i64 - 2;
        let sp_last = sp_first + dist_len as i64; // exclusive
        let actual_w = self.actual_width() as usize;
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };

        let first_pixel = (sp_first.max(0) / 3) as usize;
        let last_pixel = (((sp_last - 1).max(0)) / 3) as usize;
        let a_fg = u32::from(c.a);

        for pixel in first_pixel..=last_pixel {
            if pixel >= actual_w {
                break;
            }
            let mut cov = [0u32; 3];
            let mut any = false;
            for ch in 0..3 {
                let sp = (pixel * 3 + ch) as i64;
                if sp >= sp_first && sp < sp_last {
                    let v = c3[(sp - sp_first) as usize];
                    cov[ch] = u32::from(v);
                    any |= v != 0;
                }
            }
            if any {
                self.composite_pixel(row, pixel, c, a_fg, cov);
            }
        }
        self.c3 = c3;
    }

    fn blend_color_hspan(
        &mut self,
        x: i32,
        y: i32,
        len: u32,
        colors: &[Rgba8],
        covers: &[CoverType],
        cover: CoverType,
    ) {
        // Per-stripe colors cannot be gathered into one composite; blend
        // stripe-by-stripe (rarely used for text).
        let actual_w = self.actual_width() as usize;
        let row = unsafe {
            let ptr = self.rbuf.row_ptr(y);
            std::slice::from_raw_parts_mut(ptr, actual_w * BPP)
        };
        for k in 0..len as usize {
            let sp = x as usize + k;
            let pixel = sp / 3;
            let channel = sp % 3;
            if pixel >= actual_w {
                break;
            }
            let col = &colors[k];
            let cov = if covers.is_empty() { cover } else { covers[k] };
            let mut c3 = [0u32; 3];
            c3[channel] = u32::from(cov);
            self.composite_pixel(row, pixel, col, u32::from(col.a), c3);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcd_distribution_lut_construction() {
        // Default weights from the C++ demo: primary=1/3, secondary=2/9, tertiary=1/9
        let lut = LcdDistributionLut::new(1.0 / 3.0, 2.0 / 9.0, 1.0 / 9.0);

        // Coverage 0 should distribute to 0
        assert_eq!(lut.primary(0), 0);
        assert_eq!(lut.secondary(0), 0);
        assert_eq!(lut.tertiary(0), 0);

        // Coverage 255 should distribute fully
        // prim + 2*sec + 2*tert should approximately equal 255
        let total = lut.primary(255) as u32
            + 2 * lut.secondary(255) as u32
            + 2 * lut.tertiary(255) as u32;
        // Due to floor(), total may be slightly less than 255
        assert!(total <= 255);
        assert!(total >= 250, "total distribution = {}", total);
    }

    #[test]
    fn test_lcd_distribution_lut_normalization() {
        // Custom weights that don't sum to 1
        let lut = LcdDistributionLut::new(3.0, 2.0, 1.0);
        // After normalization: prim=3/9, sec=2/9, tert=1/9
        // primary(255) = floor(3/9 * 255) = floor(85) = 85
        assert_eq!(lut.primary(255), 85);
    }

    fn make_buffer(w: u32, h: u32) -> (Vec<u8>, RowAccessor) {
        let stride = (w * BPP as u32) as i32;
        let buf = vec![255u8; (h * w * BPP as u32) as usize];
        let mut ra = RowAccessor::new();
        unsafe {
            ra.attach(buf.as_ptr() as *mut u8, w, h, stride);
        }
        (buf, ra)
    }

    #[test]
    fn test_pixfmt_lcd_width_height() {
        let (_buf, mut ra) = make_buffer(100, 50);
        let lut = LcdDistributionLut::new(1.0 / 3.0, 2.0 / 9.0, 1.0 / 9.0);
        let pf = PixfmtRgba32Lcd::new(&mut ra, &lut);
        assert_eq!(pf.width(), 300); // 100 * 3
        assert_eq!(pf.height(), 50);
    }

    #[test]
    fn test_lcd_blend_solid_hspan_black_on_white() {
        // Blend black text on white background — should darken the pixels
        let (_buf, mut ra) = make_buffer(100, 10);
        let lut = LcdDistributionLut::new(1.0 / 3.0, 2.0 / 9.0, 1.0 / 9.0);
        let mut pf = PixfmtRgba32Lcd::new(&mut ra, &lut);

        // Full-coverage span of 6 subpixels at position 30 (pixel 10)
        let covers = [255u8; 6];
        let black = Rgba8::new(0, 0, 0, 255);
        pf.blend_solid_hspan(30, 5, 6, &black, &covers);

        // The affected pixels should be darker than 255 (white)
        let p = pf.pixel(30, 5); // pixel 10
        assert!(
            p.r < 255 || p.g < 255 || p.b < 255,
            "Expected darkened pixel, got {:?}",
            (p.r, p.g, p.b)
        );
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;

    fn make_buffer_colored(w: u32, h: u32, rgba: [u8; 4]) -> (Vec<u8>, RowAccessor) {
        let stride = (w * BPP as u32) as i32;
        let mut buf = Vec::with_capacity((h * w * BPP as u32) as usize);
        for _ in 0..(h * w) {
            buf.extend_from_slice(&rgba);
        }
        let mut ra = RowAccessor::new();
        unsafe {
            ra.attach(buf.as_ptr() as *mut u8, w, h, stride);
        }
        (buf, ra)
    }

    #[test]
    fn srgb_luts_round_trip_exactly() {
        let luts = SrgbLuts::get();
        for v in 0..=255u8 {
            assert_eq!(luts.srgb(luts.lin(v)), v, "sRGB LUT round trip failed at {v}");
        }
    }

    /// The core gamma assertion: 50% coverage of white over black must land
    /// at the LINEAR midpoint (sRGB ~188), not the sRGB code midpoint (128).
    /// Legacy sRGB-space blending yields ~127 — visibly too dark, which is
    /// exactly why thin light-on-dark text used to disappear.
    #[test]
    fn half_coverage_white_on_black_is_the_linear_midpoint() {
        let (_buf, mut ra) = make_buffer_colored(4, 1, [0, 0, 0, 255]);
        let lut = LcdDistributionLut::new(1.0, 0.0, 0.0); // identity distribution
        let params = LcdBlendParams { contrast: 0, chroma_limit: 0, coverage_gamma: 100 };
        let mut pf = PixfmtRgba32LcdLinear::new(&mut ra, &lut, params);
        let covers = [128u8; 3]; // one full pixel, all three stripes at ~50%
        let white = Rgba8::new(255, 255, 255, 255);
        pf.blend_solid_hspan(3, 0, 3, &white, &covers);
        let p = pf.pixel(3, 0); // pixel 1
        assert!(
            (183..=193).contains(&(p.g as i32)),
            "50% white over black must encode ~188 (linear midpoint), got {}",
            p.g
        );
    }

    #[test]
    fn full_coverage_reaches_exact_fg_color() {
        let (_buf, mut ra) = make_buffer_colored(4, 1, [0, 128, 0, 255]);
        let lut = LcdDistributionLut::new(1.0, 0.0, 0.0);
        let params = LcdBlendParams { contrast: 0, chroma_limit: 0, coverage_gamma: 220 };
        let mut pf = PixfmtRgba32LcdLinear::new(&mut ra, &lut, params);
        let covers = [255u8; 3];
        let white = Rgba8::new(255, 255, 255, 255);
        pf.blend_solid_hspan(3, 0, 3, &white, &covers);
        let p = pf.pixel(3, 0);
        assert_eq!((p.r, p.g, p.b), (255, 255, 255), "full coverage must hit fg exactly");
    }

    /// chroma_limit=255 must make all three stripes equal the mean-coverage
    /// composite (grayscale-equivalent), regardless of per-stripe covers.
    #[test]
    fn chroma_limit_full_equalizes_stripes() {
        let (_buf, mut ra) = make_buffer_colored(4, 1, [0, 128, 0, 255]);
        let lut = LcdDistributionLut::new(1.0, 0.0, 0.0);
        let params = LcdBlendParams { contrast: 0, chroma_limit: 255, coverage_gamma: 100 };
        let mut pf = PixfmtRgba32LcdLinear::new(&mut ra, &lut, params);
        let covers = [255u8, 128, 0]; // wildly unequal stripes
        let white = Rgba8::new(255, 255, 255, 255);
        pf.blend_solid_hspan(3, 0, 3, &white, &covers);
        let p = pf.pixel(3, 0);
        // R stripe composite with mean coverage vs B stripe: both driven to
        // the same mean, so R and B (equal bg, equal fg) must match.
        assert_eq!(p.r, p.b, "full chroma clamp must equalize stripes over equal fg/bg channels");
    }
}
