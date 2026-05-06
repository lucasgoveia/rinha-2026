use std::sync::OnceLock;

pub static ASYM_L2: OnceLock<fn(&[f32; 16], &[i16; 16]) -> f32> = OnceLock::new();
pub static L2_F32: OnceLock<fn(&[f32; 16], &[f32; 16]) -> f32> = OnceLock::new();
pub static BBOX_MIN_L2: OnceLock<fn(&[f32; 16], &[i16; 16], &[i16; 16]) -> f32> = OnceLock::new();

pub fn init() {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            ASYM_L2.set(asym_l2_avx2).ok();
            L2_F32.set(l2_f32_avx2).ok();
            BBOX_MIN_L2.set(bbox_min_l2_avx2).ok();
            return;
        }
        if is_x86_feature_detected!("avx2") {
            ASYM_L2.set(asym_l2_avx2).ok();
            L2_F32.set(l2_f32_avx2).ok();
            BBOX_MIN_L2.set(bbox_min_l2_avx2).ok();
            return;
        }
    }
    ASYM_L2.set(asym_l2_scalar).ok();
    L2_F32.set(l2_f32_scalar).ok();
    BBOX_MIN_L2.set(bbox_min_l2_scalar).ok();
}

pub fn asym_l2_scalar(q: &[f32; 16], db: &[i16; 16]) -> f32 {
    const INV: f32 = 1.0 / 10000.0;
    let mut sum = 0.0f32;
    for d in 0..16 {
        let diff = q[d] - db[d] as f32 * INV;
        sum += diff * diff;
    }
    sum
}

pub fn l2_f32_scalar(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..16 {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum
}

pub fn bbox_min_l2_scalar(q: &[f32; 16], mins: &[i16; 16], maxs: &[i16; 16]) -> f32 {
    const INV: f32 = 1.0 / 10000.0;
    let mut sum = 0.0f32;
    for d in 0..16 {
        let lo = mins[d] as f32 * INV;
        let hi = maxs[d] as f32 * INV;
        let clamped = q[d].clamp(lo, hi);
        let diff = q[d] - clamped;
        sum += diff * diff;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn asym_l2_avx2_impl(q: &[f32; 16], db: &[i16; 16]) -> f32 {
    use std::arch::x86_64::*;
    const INV: f32 = 1.0 / 10000.0;
    let inv = _mm256_set1_ps(INV);

    let db_lo_i16 = _mm_loadu_si128(db.as_ptr() as *const __m128i);
    let db_lo_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(db_lo_i16));
    let db_lo = _mm256_mul_ps(db_lo_f32, inv);
    let q_lo = _mm256_loadu_ps(q.as_ptr());
    let diff_lo = _mm256_sub_ps(q_lo, db_lo);
    let sq_lo = _mm256_mul_ps(diff_lo, diff_lo);

    let db_hi_i16 = _mm_loadu_si128(db.as_ptr().add(8) as *const __m128i);
    let db_hi_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(db_hi_i16));
    let db_hi = _mm256_mul_ps(db_hi_f32, inv);
    let q_hi = _mm256_loadu_ps(q.as_ptr().add(8));
    let diff_hi = _mm256_sub_ps(q_hi, db_hi);
    let sq_hi = _mm256_mul_ps(diff_hi, diff_hi);

    hsum_avx2(_mm256_add_ps(sq_lo, sq_hi))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_f32_avx2_impl(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    use std::arch::x86_64::*;
    let a_lo = _mm256_loadu_ps(a.as_ptr());
    let b_lo = _mm256_loadu_ps(b.as_ptr());
    let sq_lo = {
        let d = _mm256_sub_ps(a_lo, b_lo);
        _mm256_mul_ps(d, d)
    };
    let a_hi = _mm256_loadu_ps(a.as_ptr().add(8));
    let b_hi = _mm256_loadu_ps(b.as_ptr().add(8));
    let sq_hi = {
        let d = _mm256_sub_ps(a_hi, b_hi);
        _mm256_mul_ps(d, d)
    };
    hsum_avx2(_mm256_add_ps(sq_lo, sq_hi))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bbox_min_l2_avx2_impl(q: &[f32; 16], mins: &[i16; 16], maxs: &[i16; 16]) -> f32 {
    use std::arch::x86_64::*;
    const INV: f32 = 1.0 / 10000.0;
    let inv = _mm256_set1_ps(INV);

    macro_rules! load_i16_to_f32 {
        ($ptr:expr) => {{
            let raw = _mm_loadu_si128($ptr as *const __m128i);
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(raw)), inv)
        }};
    }

    let lo_mins = load_i16_to_f32!(mins.as_ptr());
    let lo_maxs = load_i16_to_f32!(maxs.as_ptr());
    let hi_mins = load_i16_to_f32!(mins.as_ptr().add(8));
    let hi_maxs = load_i16_to_f32!(maxs.as_ptr().add(8));

    let q_lo = _mm256_loadu_ps(q.as_ptr());
    let q_hi = _mm256_loadu_ps(q.as_ptr().add(8));

    let cl_lo = _mm256_min_ps(_mm256_max_ps(q_lo, lo_mins), lo_maxs);
    let cl_hi = _mm256_min_ps(_mm256_max_ps(q_hi, hi_mins), hi_maxs);

    let d_lo = _mm256_sub_ps(q_lo, cl_lo);
    let d_hi = _mm256_sub_ps(q_hi, cl_hi);
    hsum_avx2(_mm256_add_ps(
        _mm256_mul_ps(d_lo, d_lo),
        _mm256_mul_ps(d_hi, d_hi),
    ))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_extractf128_ps(v, 0);
    let hi = _mm256_extractf128_ps(v, 1);
    let s4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s4);
    let s2 = _mm_add_ps(s4, shuf);
    let s1 = _mm_add_ss(s2, _mm_movehl_ps(shuf, s2));
    _mm_cvtss_f32(s1)
}

// Safe fn-pointer-compatible wrappers for the OnceLock dispatch
fn asym_l2_avx2(q: &[f32; 16], db: &[i16; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        asym_l2_avx2_impl(q, db)
    }
    #[cfg(not(target_arch = "x86_64"))]
    asym_l2_scalar(q, db)
}

fn l2_f32_avx2(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        l2_f32_avx2_impl(a, b)
    }
    #[cfg(not(target_arch = "x86_64"))]
    l2_f32_scalar(a, b)
}

fn bbox_min_l2_avx2(q: &[f32; 16], mins: &[i16; 16], maxs: &[i16; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        bbox_min_l2_avx2_impl(q, mins, maxs)
    }
    #[cfg(not(target_arch = "x86_64"))]
    bbox_min_l2_scalar(q, mins, maxs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_asym_l2(q: &[f32; 16], db: &[i16; 16]) -> f32 {
        (0..16)
            .map(|d| {
                let diff = q[d] - db[d] as f32 / 10000.0;
                diff * diff
            })
            .sum()
    }

    fn scalar_l2_f32(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        (0..16)
            .map(|d| {
                let diff = a[d] - b[d];
                diff * diff
            })
            .sum()
    }

    #[test]
    fn asym_l2_matches_scalar() {
        init();
        let q: [f32; 16] = [
            0.5, 0.3, 0.1, 0.8, 0.2, -1.0, -1.0, 0.05, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006, 0.0,
            0.0,
        ];
        let db: [i16; 16] = [
            5000, 3000, 1000, 8000, 2000, -10000, -10000, 500, 1500, 0, 10000, 0, 1500, 60, 0, 0,
        ];
        let expected = scalar_asym_l2(&q, &db);
        let got = ASYM_L2.get().unwrap()(&q, &db);
        assert!(
            (got - expected).abs() < 1e-6,
            "asym_l2: got {got} expected {expected}"
        );
    }

    #[test]
    fn l2_f32_matches_scalar() {
        init();
        let a: [f32; 16] = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0,
        ];
        let b: [f32; 16] = [
            0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1.0, 0.9, 0.8, 0.7, 0.0, 0.0,
        ];
        let expected = scalar_l2_f32(&a, &b);
        let got = L2_F32.get().unwrap()(&a, &b);
        assert!(
            (got - expected).abs() < 1e-6,
            "l2_f32: got {got} expected {expected}"
        );
    }

    #[test]
    fn bbox_min_l2_zero_when_inside() {
        init();
        let q: [f32; 16] = [0.5; 16];
        let mins: [i16; 16] = [0; 16];
        let maxs: [i16; 16] = [10000; 16];
        let got = BBOX_MIN_L2.get().unwrap()(&q, &mins, &maxs);
        assert!(
            got.abs() < 1e-6,
            "should be 0 when query is inside bbox"
        );
    }

    #[test]
    fn bbox_min_l2_nonzero_when_outside() {
        init();
        let mut q = [0.5f32; 16];
        q[0] = 2.0;
        let mins: [i16; 16] = [0; 16];
        let maxs: [i16; 16] = [10000; 16];
        let got = BBOX_MIN_L2.get().unwrap()(&q, &mins, &maxs);
        assert!(got > 0.0, "should be nonzero when query is outside bbox");
        assert!((got - 1.0).abs() < 1e-5);
    }
}
