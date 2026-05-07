use rayon::prelude::*;
use crate::simd::L2_F32;

pub struct KMeansConfig {
    pub n_clusters: usize,
    pub max_iters: usize,
    pub change_tol: f64,
}

/// Returns (centroids: Vec<f32> of shape n_clusters×16, assignments: Vec<usize>)
pub fn fit(vectors: &[[f32; 16]], config: &KMeansConfig, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let n = vectors.len();
    let k = config.n_clusters;
    let mut rng = seed.wrapping_add(1);

    let sample_size = n.min(10_000);
    let step = n / sample_size.max(1);
    let subsample: Vec<&[f32; 16]> = (0..sample_size).map(|i| &vectors[i * step]).collect();

    let mut centroids = seed_plus_plus(&subsample, k, &mut rng);

    let mut assignments = vec![0usize; n];

    for iter in 0..config.max_iters {
        let new_assignments: Vec<usize> = vectors.par_iter().map(|v| {
            let l2 = L2_F32.get().expect("simd::init() not called");
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for c in 0..k {
                let centroid: &[f32; 16] = centroids[c * 16..(c + 1) * 16].try_into().unwrap();
                let d = l2(v, centroid);
                if d < best_dist { best_dist = d; best = c; }
            }
            best
        }).collect();

        let changes = new_assignments.iter().zip(&assignments).filter(|(a, b)| a != b).count();
        assignments = new_assignments;

        let mut sums = vec![0.0f64; k * 16];
        let mut counts = vec![0usize; k];
        for (v, &c) in vectors.iter().zip(&assignments) {
            for d in 0..16 { sums[c * 16 + d] += v[d] as f64; }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..16 {
                    centroids[c * 16 + d] = (sums[c * 16 + d] / counts[c] as f64) as f32;
                }
            }
        }

        let change_frac = changes as f64 / n as f64;
        eprintln!("kmeans iter {}/{}: {:.2}% changed", iter + 1, config.max_iters, change_frac * 100.0);
        if change_frac < config.change_tol { break; }
    }

    (centroids, assignments)
}

fn seed_plus_plus(subsample: &[&[f32; 16]], k: usize, rng: &mut u64) -> Vec<f32> {
    let n = subsample.len();
    let mut centroids = Vec::with_capacity(k * 16);

    let first = lcg_usize(rng, n);
    centroids.extend_from_slice(subsample[first]);

    let mut min_dists = vec![f32::MAX; n];

    for _ in 1..k {
        let n_existing = centroids.len() / 16;
        let new_c: &[f32; 16] = centroids[(n_existing - 1) * 16..n_existing * 16].try_into().unwrap();

        for (i, v) in subsample.iter().enumerate() {
            let d = scalar_l2(v, new_c);
            if d < min_dists[i] { min_dists[i] = d; }
        }

        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        let target = lcg_f64(rng) * total;
        let mut cumsum = 0.0f64;
        let mut chosen = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            cumsum += d as f64;
            if cumsum >= target { chosen = i; break; }
        }
        centroids.extend_from_slice(subsample[chosen]);
    }

    centroids
}

fn scalar_l2(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    (0..16).map(|d| { let diff = a[d] - b[d]; diff * diff }).sum()
}

fn lcg_usize(state: &mut u64, n: usize) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as usize % n
}

fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_separates_two_clusters() {
        crate::simd::init();
        let mut vectors: Vec<[f32; 16]> = Vec::new();
        for _ in 0..20 { vectors.push([0.05; 16]); }
        for _ in 0..20 { vectors.push([0.95; 16]); }

        let config = KMeansConfig { n_clusters: 2, max_iters: 20, change_tol: 1e-4 };
        let (_centroids, assignments) = fit(&vectors, &config, 42);

        let cluster_of_first = assignments[0];
        let cluster_of_last = assignments[20];
        assert_ne!(cluster_of_first, cluster_of_last);
        for i in 0..20 { assert_eq!(assignments[i], cluster_of_first); }
        for i in 20..40 { assert_eq!(assignments[i], cluster_of_last); }
    }

    #[test]
    fn fit_returns_correct_shapes() {
        crate::simd::init();
        let vectors: Vec<[f32; 16]> = (0..50).map(|i| [i as f32 / 50.0; 16]).collect();
        let config = KMeansConfig { n_clusters: 5, max_iters: 10, change_tol: 1e-4 };
        let (centroids, assignments) = fit(&vectors, &config, 0);
        assert_eq!(centroids.len(), 5 * 16);
        assert_eq!(assignments.len(), 50);
        assert!(assignments.iter().all(|&c| c < 5));
    }
}
