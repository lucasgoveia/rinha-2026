use crate::format::IvfIndex;
use crate::simd::{ASYM_L2, BBOX_MIN_L2, L2_F32};

pub const K: usize = 5;
pub const FRAUD_THRESHOLD: f32 = 0.6;

pub struct NeighborHeap<const N: usize> {
    distances: [f32; N],
    labels: [u8; N],
    count: usize,
    max_idx: usize,
}

impl<const N: usize> NeighborHeap<N> {
    pub fn new() -> Self {
        Self { distances: [f32::MAX; N], labels: [0; N], count: 0, max_idx: 0 }
    }

    pub fn is_full(&self) -> bool { self.count == N }

    pub fn max_dist(&self) -> f32 {
        if self.count < N { f32::MAX } else { self.distances[self.max_idx] }
    }

    pub fn push(&mut self, dist: f32, label: u8) {
        if self.count < N {
            self.distances[self.count] = dist;
            self.labels[self.count] = label;
            self.count += 1;
            if self.count == N { self.find_max(); }
        } else if dist < self.distances[self.max_idx] {
            self.distances[self.max_idx] = dist;
            self.labels[self.max_idx] = label;
            self.find_max();
        }
    }

    fn find_max(&mut self) {
        self.max_idx = 0;
        for i in 1..N {
            if self.distances[i] > self.distances[self.max_idx] { self.max_idx = i; }
        }
    }

    pub fn fraud_count(&self) -> usize {
        (0..self.count).filter(|&i| self.labels[i] == 1).count()
    }
}

pub fn search(query: &[f32; 16], index: &IvfIndex, nprobe: usize) -> usize {
    let asym_l2 = ASYM_L2.get().expect("simd::init() not called");
    let l2_f32 = L2_F32.get().expect("simd::init() not called");
    let bbox_min_l2 = BBOX_MIN_L2.get().expect("simd::init() not called");

    let k = index.n_clusters as usize;
    let s = index.stride as usize;

    let mut cdists: Vec<(f32, usize)> = (0..k).map(|c| {
        let centroid: &[f32; 16] = index.centroid(c).try_into().unwrap();
        (l2_f32(query, centroid), c)
    }).collect();

    let nprobe = nprobe.min(k);
    cdists.select_nth_unstable_by(nprobe.saturating_sub(1), |a, b| a.0.total_cmp(&b.0));
    let candidates = &cdists[..nprobe];

    let mut heap = NeighborHeap::<K>::new();

    for &(_, cid) in candidates {
        let (mins_sl, maxs_sl) = index.bbox(cid);
        let mins: &[i16; 16] = mins_sl.try_into().unwrap();
        let maxs: &[i16; 16] = maxs_sl.try_into().unwrap();

        let lb = bbox_min_l2(query, mins, maxs);
        if heap.is_full() && lb >= heap.max_dist() { continue; }

        let (vecs, labs) = index.cluster(cid);
        for (chunk, &label) in vecs.chunks_exact(s).zip(labs) {
            let db: &[i16; 16] = chunk.try_into().unwrap();
            let dist = asym_l2(query, db);
            heap.push(dist, label);
        }
    }

    heap.fraud_count()
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
    use crate::format::IvfIndex;

    fn init() { crate::simd::init(); }

    #[test]
    fn neighbor_heap_tracks_top5() {
        init();
        let mut h = NeighborHeap::<5>::new();
        assert!(!h.is_full());
        assert_eq!(h.max_dist(), f32::MAX);

        h.push(0.9, 1);
        h.push(0.1, 0);
        h.push(0.5, 1);
        h.push(0.3, 0);
        h.push(0.7, 1);
        assert!(h.is_full());
        assert!((h.max_dist() - 0.9).abs() < 1e-6);

        h.push(0.2, 0); // replaces 0.9
        assert!((h.max_dist() - 0.7).abs() < 1e-6);
        assert_eq!(h.fraud_count(), 2); // 0.5→fraud, 0.7→fraud; 0.1,0.2,0.3→legit
    }

    #[test]
    fn search_all_fraud_returns_5() {
        init();
        let n_clusters = 1u32;
        let stride = 16usize;
        let n_vectors = 10u32;
        let query = [5000i16; 16];
        let vectors: Vec<i16> = query.iter().cloned().cycle().take(n_vectors as usize * stride).collect();

        let index = IvfIndex {
            n_vectors,
            n_dims: 14, stride: 16, quant_scale: 10000.0,
            n_clusters, nprobe: 1,
            centroids: vec![0.5f32; stride],
            bboxes: vec![0i16; stride * 2],
            offsets: vec![0],
            sizes: vec![n_vectors],
            labels: vec![1u8; n_vectors as usize],
            vectors,
        };

        let q_float: [f32; 16] = [0.5; 16];
        let fraud_count = search(&q_float, &index, 1);
        assert_eq!(fraud_count, 5);
    }

    #[test]
    fn search_all_legit_returns_0() {
        init();
        let n_clusters = 1u32;
        let stride = 16usize;
        let n_vectors = 10u32;
        let query = [5000i16; 16];
        let vectors: Vec<i16> = query.iter().cloned().cycle().take(n_vectors as usize * stride).collect();

        let index = IvfIndex {
            n_vectors,
            n_dims: 14, stride: 16, quant_scale: 10000.0,
            n_clusters, nprobe: 1,
            centroids: vec![0.5f32; stride],
            bboxes: vec![0i16; stride * 2],
            offsets: vec![0],
            sizes: vec![n_vectors],
            labels: vec![0u8; n_vectors as usize],
            vectors,
        };

        let q_float: [f32; 16] = [0.5; 16];
        let fraud_count = search(&q_float, &index, 1);
        assert_eq!(fraud_count, 0);
    }
}
