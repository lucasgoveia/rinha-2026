use std::path::Path;

pub const MAGIC: &[u8; 8] = b"IVFVEC01";
pub const VERSION: u8 = 1;
pub const QUANT_SCALE: f32 = 10000.0;

#[derive(Debug)]
pub struct IvfIndex {
    pub n_vectors: u32,
    pub n_dims: u16,
    pub stride: u16,
    pub quant_scale: f32,
    pub n_clusters: u32,
    pub nprobe: u16,
    pub centroids: Vec<f32>,
    pub bboxes: Vec<i16>,
    pub offsets: Vec<u32>,
    pub sizes: Vec<u32>,
    pub labels: Vec<u8>,
    pub vectors: Vec<i16>,
}

impl IvfIndex {
    pub fn cluster(&self, cid: usize) -> (&[i16], &[u8]) {
        let s = self.stride as usize;
        let off = self.offsets[cid] as usize;
        let sz = self.sizes[cid] as usize;
        (&self.vectors[off * s..(off + sz) * s], &self.labels[off..off + sz])
    }

    pub fn centroid(&self, cid: usize) -> &[f32] {
        let s = self.stride as usize;
        &self.centroids[cid * s..(cid + 1) * s]
    }

    pub fn bbox(&self, cid: usize) -> (&[i16], &[i16]) {
        let s = self.stride as usize;
        let base = cid * s * 2;
        (&self.bboxes[base..base + s], &self.bboxes[base + s..base + 2 * s])
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{BufWriter, Write};
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);

        w.write_all(MAGIC)?;
        w.write_all(&[VERSION])?;
        w.write_all(&self.n_vectors.to_le_bytes())?;
        w.write_all(&self.n_dims.to_le_bytes())?;
        w.write_all(&self.stride.to_le_bytes())?;
        w.write_all(&self.quant_scale.to_bits().to_le_bytes())?;
        w.write_all(&self.n_clusters.to_le_bytes())?;
        w.write_all(&self.nprobe.to_le_bytes())?;

        for &v in &self.centroids { w.write_all(&v.to_bits().to_le_bytes())?; }
        for &v in &self.bboxes    { w.write_all(&(v as u16).to_le_bytes())?; }
        for &v in &self.offsets   { w.write_all(&v.to_le_bytes())?; }
        for &v in &self.sizes     { w.write_all(&v.to_le_bytes())?; }
        w.write_all(&self.labels)?;
        for &v in &self.vectors   { w.write_all(&(v as u16).to_le_bytes())?; }

        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if data.len() < 26 { return Err("file too short".into()); }
        if &data[0..8] != MAGIC { return Err(format!("bad magic: {:?}", &data[0..8]).into()); }
        if data[8] != VERSION { return Err(format!("unsupported version {}", data[8]).into()); }

        let mut pos = 9;
        macro_rules! read_u16 { () => {{ let v = u16::from_le_bytes(data[pos..pos+2].try_into()?); pos += 2; v }}; }
        macro_rules! read_u32 { () => {{ let v = u32::from_le_bytes(data[pos..pos+4].try_into()?); pos += 4; v }}; }
        macro_rules! read_f32 { () => {{ let v = f32::from_bits(u32::from_le_bytes(data[pos..pos+4].try_into()?)); pos += 4; v }}; }

        let n_vectors  = read_u32!();
        let n_dims     = read_u16!();
        let stride     = read_u16!();
        let quant_scale = read_f32!();
        let n_clusters = read_u32!();
        let nprobe     = read_u16!();

        let k = n_clusters as usize;
        let s = stride as usize;
        let n = n_vectors as usize;

        let centroids = read_f32_vec(&data, &mut pos, k * s)?;
        let bboxes    = read_i16_vec(&data, &mut pos, k * s * 2)?;
        let offsets   = read_u32_vec(&data, &mut pos, k)?;
        let sizes     = read_u32_vec(&data, &mut pos, k)?;
        let labels    = data[pos..pos + n].to_vec(); pos += n;
        let vectors   = read_i16_vec(&data, &mut pos, n * s)?;

        Ok(IvfIndex { n_vectors, n_dims, stride, quant_scale, n_clusters, nprobe,
                      centroids, bboxes, offsets, sizes, labels, vectors })
    }
}

pub fn quantize(v: f32) -> i16 {
    (v * QUANT_SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn read_f32_vec(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = count * 4;
    if *pos + bytes > data.len() { return Err("truncated f32 section".into()); }
    let v = (0..count).map(|i| {
        let p = *pos + i * 4;
        f32::from_bits(u32::from_le_bytes(data[p..p+4].try_into().unwrap()))
    }).collect();
    *pos += bytes;
    Ok(v)
}

fn read_i16_vec(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    let bytes = count * 2;
    if *pos + bytes > data.len() { return Err("truncated i16 section".into()); }
    let v = (0..count).map(|i| {
        let p = *pos + i * 2;
        i16::from_le_bytes(data[p..p+2].try_into().unwrap())
    }).collect();
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
        let n_clusters = 4u32;
        let stride = 16usize;
        let n_vectors = 8u32;
        IvfIndex {
            n_vectors,
            n_dims: 14,
            stride: stride as u16,
            quant_scale: 10000.0,
            n_clusters,
            nprobe: 2,
            centroids: vec![0.5f32; n_clusters as usize * stride],
            bboxes: vec![0i16; n_clusters as usize * stride * 2],
            offsets: vec![0, 2, 4, 6],
            sizes: vec![2, 2, 2, 2],
            labels: vec![0, 1, 0, 1, 1, 0, 0, 1],
            vectors: vec![1000i16; n_vectors as usize * stride],
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
        assert_eq!(loaded.centroids, original.centroids);
        assert_eq!(loaded.bboxes, original.bboxes);
        assert_eq!(loaded.offsets, original.offsets);
        assert_eq!(loaded.sizes, original.sizes);
        assert_eq!(loaded.labels, original.labels);
        assert_eq!(loaded.vectors, original.vectors);
    }

    #[test]
    fn cluster_returns_correct_slice() {
        let idx = make_small_index();
        let (vecs, labs) = idx.cluster(1);
        assert_eq!(labs, &[0u8, 1u8]);
        assert_eq!(vecs.len(), 2 * 16);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut data = vec![0u8; 200];
        data[0..8].copy_from_slice(b"GARBAGE!");
        assert!(IvfIndex::from_bytes(&data).is_err());
    }
}
