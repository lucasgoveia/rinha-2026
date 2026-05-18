use crate::format::IvfIndex;
use crate::simd::{CENTROID_DISTS, SCAN_BLOCKS};

pub const K: usize = 5;
pub const FRAUD_THRESHOLD: f32 = 0.6;
const FAST_NPROBE: usize = 8;
const FULL_NPROBE: usize = 24;

// Returns N indices of the smallest values in `dists`.
fn top_n<const N: usize>(dists: &[f32]) -> [usize; N] {
    let mut result   = [0usize; N];
    let mut top_d    = [f32::INFINITY; N];
    let mut worst_i  = 0usize;

    for (c, &d) in dists.iter().enumerate() {
        if d < top_d[worst_i] {
            top_d[worst_i]   = d;
            result[worst_i]  = c;
            worst_i = 0;
            let mut wv = top_d[0];
            for i in 1..N {
                if top_d[i] > wv { wv = top_d[i]; worst_i = i; }
            }
        }
    }
    result
}

fn top_n_dynamic(dists: &[f32], n: usize) -> Vec<usize> {
    let n = n.min(dists.len());
    if n == dists.len() {
        return (0..dists.len()).collect();
    }

    let mut ids: Vec<usize> = (0..dists.len()).collect();
    ids.select_nth_unstable_by(n - 1, |&a, &b| dists[a].total_cmp(&dists[b]));
    ids.truncate(n);
    ids
}

fn scan_probes(
    query: &[f32; 16],
    index: &IvfIndex,
    probes: &[usize],
) -> usize {
    let scan_fn = SCAN_BLOCKS.get().expect("simd::init() not called");
    let blocks_ptr = index.blocks.as_ptr();
    let labels_ptr = index.labels.as_ptr();

    let mut top: [(f32, u8); K] = [(f32::INFINITY, 0); K];
    let mut worst_idx = 0usize;

    for &cid in probes {
        let (start, n) = index.cluster_blocks(cid);
        if n == 0 { continue; }
        scan_fn(query, blocks_ptr, labels_ptr, start, start + n, &mut top, &mut worst_idx);
    }

    top.iter().filter(|(_, l)| *l == 1).count()
}

pub fn search_nprobe(query: &[f32; 16], index: &IvfIndex, nprobe: usize) -> usize {
    let k = index.n_clusters as usize;
    let cdists_fn = CENTROID_DISTS.get().expect("simd::init() not called");

    let mut cdists = vec![0.0f32; k];
    cdists_fn(query, &index.centroids, k, &mut cdists);

    let probes = top_n_dynamic(&cdists, nprobe.clamp(1, k));
    scan_probes(query, index, &probes)
}

pub fn search(query: &[f32; 16], index: &IvfIndex, _nprobe: usize) -> usize {
    let k = index.n_clusters as usize;
    let cdists_fn = CENTROID_DISTS.get().expect("simd::init() not called");

    let mut cdists = vec![0.0f32; k];
    cdists_fn(query, &index.centroids, k, &mut cdists);

    let fast_probes = top_n::<FAST_NPROBE>(&cdists);
    let fast_count  = scan_probes(query, index, &fast_probes);

    if fast_count == 2 || fast_count == 3 {
        let full_probes = top_n::<FULL_NPROBE>(&cdists);
        scan_probes(query, index, &full_probes)
    } else {
        fast_count
    }
}

pub fn fraud_score(fraud_count: usize) -> f32 {
    fraud_count as f32 / K as f32
}

pub fn is_approved(fraud_count: usize) -> bool {
    fraud_score(fraud_count) < FRAUD_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{IvfIndex, BLOCK_ELEMS, BLOCK_SIZE, N_DIMS, QUANT_SCALE};
    use aligned_vec::{AVec, ConstAlign};

    fn init() { crate::simd::init(); }

    fn make_test_index(label: u8, n_vecs: usize) -> IvfIndex {
        let k = 1usize;
        let n_blocks = (n_vecs + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let mut centroids: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32,N_DIMS * k);
        for _ in 0..(N_DIMS * k) { centroids.push(0.5f32); }

        let offsets = vec![0u32, n_blocks as u32];

        let mut labels = vec![0u8; n_blocks * BLOCK_SIZE];
        for i in 0..n_vecs { labels[i] = label; }

        // Blocks: all real vectors store 0.5 (= 5000 quantized), padding = i16::MAX
        let mut blocks: AVec<i16, ConstAlign<32>> = AVec::with_capacity(32,n_blocks * BLOCK_ELEMS);
        for _ in 0..(n_blocks * BLOCK_ELEMS) { blocks.push(i16::MAX); }
        for vi in 0..n_vecs {
            let b  = vi / BLOCK_SIZE;
            let s  = vi % BLOCK_SIZE;
            let bb = b * BLOCK_ELEMS;
            for d in 0..N_DIMS {
                blocks[bb + d * BLOCK_SIZE + s] = (0.5 * QUANT_SCALE).round() as i16;
            }
        }

        IvfIndex {
            n_vectors:  n_vecs as u32,
            n_dims:     N_DIMS as u16,
            n_clusters: k as u32,
            quant_scale: QUANT_SCALE,
            nprobe:     1,
            centroids,
            offsets,
            labels,
            blocks,
        }
    }

    #[test]
    fn search_all_fraud_returns_5() {
        init();
        let index = make_test_index(1, 10);
        let q: [f32; 16] = [0.5; 16];
        assert_eq!(search(&q, &index, 1), 5);
    }

    #[test]
    fn search_all_legit_returns_0() {
        init();
        let index = make_test_index(0, 10);
        let q: [f32; 16] = [0.5; 16];
        assert_eq!(search(&q, &index, 1), 0);
    }

    #[test]
    fn top_n_selects_smallest() {
        let dists = vec![0.9f32, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4];
        let top3 = top_n::<3>(&dists);
        let mut sorted = top3;
        sorted.sort_unstable();
        // Indices of 0.1, 0.2, 0.3 are 1, 5, 3
        let expected_vals: Vec<f32> = sorted.iter().map(|&i| dists[i]).collect();
        assert!(expected_vals.iter().all(|&v| v <= 0.3 + 1e-6));
    }
}
