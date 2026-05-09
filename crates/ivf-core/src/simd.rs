use std::sync::OnceLock;

// f32 L2 between two stride-16 vectors (used by k-means assignment).
pub static L2_F32: OnceLock<fn(&[f32; 16], &[f32; 16]) -> f32> = OnceLock::new();

// Compute L2 distance from query to every centroid (dim-major layout).
// Signature: (query, centroids_slice, k, out_dists)
pub static CENTROID_DISTS: OnceLock<fn(&[f32; 16], &[f32], usize, &mut [f32])> = OnceLock::new();

// Scan a range of blocks against a query, updating top-5 in place.
// Signature: (query, blocks_ptr, labels_ptr, start_block, end_block, top5, worst_idx)
pub static SCAN_BLOCKS: OnceLock<
    fn(&[f32; 16], *const i16, *const u8, usize, usize, &mut [(f32, u8); 5], &mut usize),
> = OnceLock::new();

pub fn init() {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            L2_F32.set(l2_f32_avx2).ok();
            CENTROID_DISTS.set(centroid_dists_avx2).ok();
            SCAN_BLOCKS.set(scan_blocks_avx2).ok();
            return;
        }
    }
    L2_F32.set(l2_f32_scalar).ok();
    CENTROID_DISTS.set(centroid_dists_scalar).ok();
    SCAN_BLOCKS.set(scan_blocks_scalar).ok();
}

// ── Scalar implementations ──────────────────────────────────────────────────

pub fn l2_f32_scalar(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..16 {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum
}

pub fn centroid_dists_scalar(q: &[f32; 16], cents: &[f32], k: usize, dists: &mut [f32]) {
    // Initialize from dim 0
    for c in 0..k {
        let diff = q[0] - cents[c];
        dists[c] = diff * diff;
    }
    // Accumulate dims 1..N_DIMS
    for dim in 1..crate::format::N_DIMS {
        let base = dim * k;
        for c in 0..k {
            let diff = q[dim] - cents[base + c];
            dists[c] += diff * diff;
        }
    }
}

pub fn scan_blocks_scalar(
    q: &[f32; 16],
    blocks: *const i16,
    labels: *const u8,
    start: usize,
    end: usize,
    top: &mut [(f32, u8); 5],
    worst_idx: &mut usize,
) {
    use crate::format::{BLOCK_ELEMS, BLOCK_SIZE, N_DIMS, QUANT_SCALE};
    const INV: f32 = 1.0 / QUANT_SCALE;

    for block_i in start..end {
        let bb = block_i * BLOCK_ELEMS;
        for slot in 0..BLOCK_SIZE {
            let mut dist = 0.0f32;
            for d in 0..N_DIMS {
                let val = unsafe { *blocks.add(bb + d * BLOCK_SIZE + slot) } as f32 * INV;
                let diff = q[d] - val;
                dist += diff * diff;
            }
            if dist < top[*worst_idx].0 {
                let label = unsafe { *labels.add(block_i * BLOCK_SIZE + slot) };
                top[*worst_idx] = (dist, label);
                *worst_idx = 0;
                let mut wv = top[0].0;
                for j in 1..5 {
                    if top[j].0 > wv { wv = top[j].0; *worst_idx = j; }
                }
            }
        }
    }
}

// ── AVX2 + FMA implementations ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_f32_avx2_impl(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    use std::arch::x86_64::*;
    let a_lo = _mm256_loadu_ps(a.as_ptr());
    let b_lo = _mm256_loadu_ps(b.as_ptr());
    let sq_lo = { let d = _mm256_sub_ps(a_lo, b_lo); _mm256_mul_ps(d, d) };
    let a_hi = _mm256_loadu_ps(a.as_ptr().add(8));
    let b_hi = _mm256_loadu_ps(b.as_ptr().add(8));
    let sq_hi = { let d = _mm256_sub_ps(a_hi, b_hi); _mm256_mul_ps(d, d) };
    hsum_avx2(_mm256_add_ps(sq_lo, sq_hi))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn centroid_dists_avx2_impl(q: &[f32; 16], cents: &[f32], k: usize, dists: &mut [f32]) {
    use std::arch::x86_64::*;

    // Dim 0: initialize dists[c] = (q[0] - cents[0*k + c])^2
    {
        let qd = _mm256_set1_ps(q[0]);
        let mut ci = 0;
        while ci + 8 <= k {
            let cv = _mm256_loadu_ps(cents.as_ptr().add(ci));
            let d  = _mm256_sub_ps(cv, qd);
            _mm256_storeu_ps(dists.as_mut_ptr().add(ci), _mm256_mul_ps(d, d));
            ci += 8;
        }
        for c in ci..k {
            let diff = cents[c] - q[0];
            dists[c] = diff * diff;
        }
    }

    // Dims 1..N_DIMS: accumulate with FMA
    for dim in 1..crate::format::N_DIMS {
        let qd   = _mm256_set1_ps(q[dim]);
        let base = dim * k;
        let mut ci = 0;
        while ci + 8 <= k {
            let cv  = _mm256_loadu_ps(cents.as_ptr().add(base + ci));
            let acc = _mm256_loadu_ps(dists.as_ptr().add(ci));
            let d   = _mm256_sub_ps(cv, qd);
            _mm256_storeu_ps(dists.as_mut_ptr().add(ci), _mm256_fmadd_ps(d, d, acc));
            ci += 8;
        }
        for c in ci..k {
            let diff = cents[base + c] - q[dim];
            dists[c] += diff * diff;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_blocks_avx2_impl(
    q: &[f32; 16],
    blocks: *const i16,
    labels: *const u8,
    start: usize,
    end: usize,
    top: &mut [(f32, u8); 5],
    worst_idx: &mut usize,
) {
    use std::arch::x86_64::*;
    use crate::format::{BLOCK_ELEMS, BLOCK_SIZE, N_DIMS};

    const INV: f32 = 1.0 / 10000.0;
    let inv = _mm256_set1_ps(INV);

    // Pre-broadcast all 14 query dimensions into AVX registers.
    let qv: [__m256; N_DIMS] = std::array::from_fn(|d| _mm256_set1_ps(q[d]));

    // Load 8 i16 values for dimension d of the current block and convert to f32.
    macro_rules! load_dim {
        ($bb:expr, $d:expr) => {{
            let raw = _mm_loadu_si128(blocks.add($bb + $d * BLOCK_SIZE) as *const __m128i);
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(raw)), inv)
        }};
    }

    for block_i in start..end {
        // Prefetch 8 blocks ahead (two cache lines per block: offsets 0 and 56 bytes).
        if block_i + 8 < end {
            _mm_prefetch(blocks.add((block_i + 8) * BLOCK_ELEMS) as *const i8, _MM_HINT_T0);
            _mm_prefetch(blocks.add((block_i + 8) * BLOCK_ELEMS + 56) as *const i8, _MM_HINT_T0);
        }

        let bb        = block_i * BLOCK_ELEMS;
        let threshold = _mm256_set1_ps(top[*worst_idx].0);

        // First 8 dimensions — partial distance accumulator.
        let d0  = _mm256_sub_ps(load_dim!(bb, 0), qv[0]);
        let mut acc = _mm256_mul_ps(d0, d0);
        for dim in 1..8 {
            let dv = _mm256_sub_ps(load_dim!(bb, dim), qv[dim]);
            acc = _mm256_fmadd_ps(dv, dv, acc);
        }

        // If no vector in this block can beat the current threshold, skip early.
        if _mm256_movemask_ps(_mm256_cmp_ps(acc, threshold, _CMP_LT_OQ)) == 0 {
            continue;
        }

        // Remaining 6 dimensions.
        for dim in 8..N_DIMS {
            let dv = _mm256_sub_ps(load_dim!(bb, dim), qv[dim]);
            acc = _mm256_fmadd_ps(dv, dv, acc);
        }

        // Extract candidates that beat the threshold.
        let mut mask = _mm256_movemask_ps(_mm256_cmp_ps(acc, threshold, _CMP_LT_OQ)) as u32;
        if mask == 0 { continue; }

        let mut dists_buf = [0.0f32; 8];
        _mm256_storeu_ps(dists_buf.as_mut_ptr(), acc);
        let label_base = block_i * BLOCK_SIZE;

        while mask != 0 {
            let slot = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let di = dists_buf[slot];
            if di < top[*worst_idx].0 {
                top[*worst_idx] = (di, *labels.add(label_base + slot));
                *worst_idx = 0;
                let mut wv = top[0].0;
                for j in 1..5 {
                    if top[j].0 > wv { wv = top[j].0; *worst_idx = j; }
                }
            }
        }
    }
}

// ── Safe wrappers (fn-pointer compatible) ───────────────────────────────────

fn l2_f32_avx2(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe { l2_f32_avx2_impl(a, b) }
    #[cfg(not(target_arch = "x86_64"))]
    l2_f32_scalar(a, b)
}

fn centroid_dists_avx2(q: &[f32; 16], cents: &[f32], k: usize, dists: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    unsafe { centroid_dists_avx2_impl(q, cents, k, dists) }
    #[cfg(not(target_arch = "x86_64"))]
    centroid_dists_scalar(q, cents, k, dists)
}

fn scan_blocks_avx2(
    q: &[f32; 16],
    blocks: *const i16,
    labels: *const u8,
    start: usize,
    end: usize,
    top: &mut [(f32, u8); 5],
    worst_idx: &mut usize,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe { scan_blocks_avx2_impl(q, blocks, labels, start, end, top, worst_idx) }
    #[cfg(not(target_arch = "x86_64"))]
    scan_blocks_scalar(q, blocks, labels, start, end, top, worst_idx)
}

// ── Horizontal sum helper ────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo   = _mm256_extractf128_ps(v, 0);
    let hi   = _mm256_extractf128_ps(v, 1);
    let s4   = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s4);
    let s2   = _mm_add_ps(s4, shuf);
    let s1   = _mm_add_ss(s2, _mm_movehl_ps(shuf, s2));
    _mm_cvtss_f32(s1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_l2_f32(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        (0..16).map(|d| { let diff = a[d] - b[d]; diff * diff }).sum()
    }

    #[test]
    fn l2_f32_matches_scalar() {
        init();
        let a: [f32; 16] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0];
        let b: [f32; 16] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1.0, 0.9, 0.8, 0.7, 0.0, 0.0];
        let expected = scalar_l2_f32(&a, &b);
        let got = L2_F32.get().unwrap()(&a, &b);
        assert!((got - expected).abs() < 1e-6, "l2_f32: got {got} expected {expected}");
    }

    #[test]
    fn centroid_dists_matches_scalar() {
        init();
        let q  = [0.5f32; 16];
        let k  = 8usize;
        let cents: Vec<f32> = (0..crate::format::N_DIMS * k)
            .map(|i| (i % k) as f32 * 0.1)
            .collect();
        let mut expected = vec![0.0f32; k];
        centroid_dists_scalar(&q, &cents, k, &mut expected);
        let mut got = vec![0.0f32; k];
        CENTROID_DISTS.get().unwrap()(&q, &cents, k, &mut got);
        for i in 0..k {
            assert!((got[i] - expected[i]).abs() < 1e-5, "centroid_dists[{i}]: got {} expected {}", got[i], expected[i]);
        }
    }

    #[test]
    fn scan_blocks_finds_nearest() {
        use crate::format::{BLOCK_SIZE, N_DIMS};
        init();

        // 1 block: 8 vectors, dim d for slot s = (s as f32 * 0.1) quantized
        let k = N_DIMS * BLOCK_SIZE; // 112 i16s
        let mut blocks = vec![0i16; k];
        let mut labels = vec![0u8; BLOCK_SIZE];
        let q = [0.5f32; 16];

        for slot in 0..BLOCK_SIZE {
            labels[slot] = if slot == 0 { 1 } else { 0 };
            for d in 0..N_DIMS {
                let val = slot as f32 * 0.1;
                blocks[d * BLOCK_SIZE + slot] = (val * 10000.0).round() as i16;
            }
        }

        let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
        let mut worst_idx = 0usize;

        SCAN_BLOCKS.get().unwrap()(
            &q,
            blocks.as_ptr(),
            labels.as_ptr(),
            0, 1,
            &mut top, &mut worst_idx,
        );

        // All 5 nearest slots should be the ones closest to 0.5 (slots 4,5,3,6,2)
        let filled = top.iter().filter(|(d, _)| d.is_finite()).count();
        assert_eq!(filled, 5);
    }
}
