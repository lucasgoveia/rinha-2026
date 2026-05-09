use std::path::Path;
use aligned_vec::{AVec, ConstAlign};

pub const MAGIC: &[u8; 8] = b"IVFVEC02";
pub const VERSION: u8 = 2;
pub const QUANT_SCALE: f32 = 10000.0;
pub const N_DIMS: usize = 14;
pub const BLOCK_SIZE: usize = 8;
pub const BLOCK_ELEMS: usize = N_DIMS * BLOCK_SIZE; // 112 i16 per block

pub struct IvfIndex {
    pub n_vectors: u32,
    pub n_dims: u16,       // always 14
    pub n_clusters: u32,
    pub quant_scale: f32,  // 10000.0
    pub nprobe: u16,
    // Centroids in dimension-major layout: centroids[d * k + c]
    // shape: N_DIMS × n_clusters
    pub centroids: AVec<f32, ConstAlign<32>>,
    // k+1 block offsets (CSR): offsets[cid] = first block, offsets[cid+1] = end block
    pub offsets: Vec<u32>,
    // Padded labels: n_blocks * BLOCK_SIZE bytes
    pub labels: Vec<u8>,
    // Block-strided vectors: n_blocks * BLOCK_ELEMS i16s
    // layout: blocks[b * BLOCK_ELEMS + d * BLOCK_SIZE + slot]
    pub blocks: AVec<i16, ConstAlign<32>>,
}

impl IvfIndex {
    // Slice of all k centroid values for dimension d.
    pub fn centroid_dim(&self, d: usize) -> &[f32] {
        let k = self.n_clusters as usize;
        &self.centroids[d * k..(d + 1) * k]
    }

    // Returns (start_block, n_blocks) for cluster cid.
    pub fn cluster_blocks(&self, cid: usize) -> (usize, usize) {
        let start = self.offsets[cid] as usize;
        let end   = self.offsets[cid + 1] as usize;
        (start, end - start)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{BufWriter, Write};
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);

        let k = self.n_clusters as usize;
        let n_blocks = self.offsets[k] as u32;

        w.write_all(MAGIC)?;
        w.write_all(&[VERSION])?;
        w.write_all(&self.n_vectors.to_le_bytes())?;
        w.write_all(&self.n_dims.to_le_bytes())?;
        w.write_all(&self.n_clusters.to_le_bytes())?;
        w.write_all(&self.nprobe.to_le_bytes())?;
        w.write_all(&self.quant_scale.to_bits().to_le_bytes())?;
        w.write_all(&n_blocks.to_le_bytes())?;

        for &v in self.centroids.iter() { w.write_all(&v.to_bits().to_le_bytes())?; }
        for &v in &self.offsets        { w.write_all(&v.to_le_bytes())?; }
        w.write_all(&self.labels)?;
        for &v in self.blocks.iter()   { w.write_all(&(v as u16).to_le_bytes())?; }

        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if data.len() < 29 { return Err("file too short".into()); }
        if &data[0..8] != MAGIC { return Err(format!("bad magic: {:?}", &data[0..8]).into()); }
        if data[8] != VERSION   { return Err(format!("unsupported version {}", data[8]).into()); }

        let mut pos = 9;
        macro_rules! read_u16 { () => {{ let v = u16::from_le_bytes(data[pos..pos+2].try_into()?); pos += 2; v }}; }
        macro_rules! read_u32 { () => {{ let v = u32::from_le_bytes(data[pos..pos+4].try_into()?); pos += 4; v }}; }
        macro_rules! read_f32 { () => {{ let v = f32::from_bits(u32::from_le_bytes(data[pos..pos+4].try_into()?)); pos += 4; v }}; }

        let n_vectors   = read_u32!();
        let n_dims      = read_u16!();
        let n_clusters  = read_u32!();
        let nprobe      = read_u16!();
        let quant_scale = read_f32!();
        let n_blocks    = read_u32!() as usize;

        let k = n_clusters as usize;
        let d = n_dims as usize;

        let centroids = read_f32_avec(&data, &mut pos, d * k)?;
        let offsets   = read_u32_vec(&data, &mut pos, k + 1)?;
        let padded_n  = n_blocks * BLOCK_SIZE;
        let labels    = data[pos..pos + padded_n].to_vec(); pos += padded_n;
        let blocks    = read_i16_avec(&data, &mut pos, n_blocks * BLOCK_ELEMS)?;

        Ok(IvfIndex { n_vectors, n_dims, n_clusters, quant_scale, nprobe,
                      centroids, offsets, labels, blocks })
    }
}

pub fn quantize(v: f32) -> i16 {
    (v * QUANT_SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn read_f32_avec(data: &[u8], pos: &mut usize, count: usize) -> Result<AVec<f32, ConstAlign<32>>, Box<dyn std::error::Error>> {
    let bytes = count * 4;
    if *pos + bytes > data.len() { return Err("truncated f32 section".into()); }
    let mut v = AVec::with_capacity(32, count);
    for i in 0..count {
        let p = *pos + i * 4;
        v.push(f32::from_bits(u32::from_le_bytes(data[p..p+4].try_into().unwrap())));
    }
    *pos += bytes;
    Ok(v)
}

fn read_i16_avec(data: &[u8], pos: &mut usize, count: usize) -> Result<AVec<i16, ConstAlign<32>>, Box<dyn std::error::Error>> {
    let bytes = count * 2;
    if *pos + bytes > data.len() { return Err("truncated i16 section".into()); }
    let mut v = AVec::with_capacity(32, count);
    for i in 0..count {
        let p = *pos + i * 2;
        v.push(i16::from_le_bytes(data[p..p+2].try_into().unwrap()));
    }
    *pos += bytes;
    Ok(v)
}

fn read_u32_vec(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let bytes = count * 4;
    if *pos + bytes > data.len() { return Err("truncated u32 section".into()); }
    let v = (0..count).map(|i| {
        let p = *pos + i * 4;
        u32::from_le_bytes(data[p..p+4].try_into().unwrap())
    }).collect();
    *pos += bytes;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_small_index() -> IvfIndex {
        // 4 clusters, 2 real vectors each → 1 block per cluster (slots 0-1 real, 2-7 padding)
        let k = 4usize;
        let n_vectors = 8u32;
        let n_blocks = 4usize; // 1 block per cluster

        let mut centroids: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32, N_DIMS * k);
        for _ in 0..(N_DIMS * k) { centroids.push(0.5f32); }

        // offsets: [0, 1, 2, 3, 4]
        let offsets: Vec<u32> = (0..=k as u32).collect();

        let mut labels = vec![0u8; n_blocks * BLOCK_SIZE];
        // Cluster 0: slots 0=legit,1=fraud; cluster 1: slots 0=legit,1=fraud; etc.
        labels[0] = 0; labels[1] = 1; // cluster 0
        labels[8] = 0; labels[9] = 1; // cluster 1
        labels[16] = 1; labels[17] = 0; // cluster 2
        labels[24] = 0; labels[25] = 1; // cluster 3

        let mut blocks: AVec<i16, ConstAlign<32>> = AVec::with_capacity(32,n_blocks * BLOCK_ELEMS);
        for _ in 0..(n_blocks * BLOCK_ELEMS) { blocks.push(1000i16); }

        IvfIndex {
            n_vectors,
            n_dims: N_DIMS as u16,
            n_clusters: k as u32,
            quant_scale: QUANT_SCALE,
            nprobe: 2,
            centroids,
            offsets,
            labels,
            blocks,
        }
    }

    #[test]
    fn round_trip_write_load() {
        let original = make_small_index();
        let tmp = NamedTempFile::new().unwrap();
        original.write(tmp.path()).unwrap();
        let loaded = IvfIndex::load(tmp.path()).unwrap();

        assert_eq!(loaded.n_vectors, original.n_vectors);
        assert_eq!(loaded.n_clusters, original.n_clusters);
        assert_eq!(loaded.nprobe, original.nprobe);
        assert_eq!(loaded.quant_scale, original.quant_scale);
        assert_eq!(loaded.centroids.as_slice(), original.centroids.as_slice());
        assert_eq!(loaded.offsets, original.offsets);
        assert_eq!(loaded.labels, original.labels);
        assert_eq!(loaded.blocks.as_slice(), original.blocks.as_slice());
    }

    #[test]
    fn cluster_blocks_returns_correct_range() {
        let idx = make_small_index();
        let (start, n) = idx.cluster_blocks(1);
        assert_eq!(start, 1);
        assert_eq!(n, 1);
    }

    #[test]
    fn centroid_dim_returns_correct_slice() {
        let idx = make_small_index();
        let k = idx.n_clusters as usize;
        let dim0 = idx.centroid_dim(0);
        assert_eq!(dim0.len(), k);
        assert!((dim0[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut data = vec![0u8; 200];
        data[0..8].copy_from_slice(b"GARBAGE!");
        assert!(IvfIndex::from_bytes(&data).is_err());
    }
}
