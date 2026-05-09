use clap::Parser;
use flate2::read::GzDecoder;
use ivf_core::{
    format::{quantize, IvfIndex, BLOCK_ELEMS, BLOCK_SIZE, N_DIMS, QUANT_SCALE},
    kmeans::{fit, KMeansConfig},
    simd,
};
use aligned_vec::{AVec, ConstAlign};
use serde::Deserialize;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "resources/references.json.gz")]
    refs: PathBuf,
    #[arg(long, default_value = "resources/normalization.json")]
    norm: PathBuf,
    #[arg(long, default_value = "resources/mcc_risk.json")]
    mcc: PathBuf,
    #[arg(long, default_value = "resources/references.ivfvec")]
    out: PathBuf,
    #[arg(long, default_value_t = 4096)]
    clusters: usize,
    #[arg(long, default_value_t = 16)]
    nprobe: u16,
    #[arg(long, default_value_t = 100)]
    iters: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: Vec<f32>,
    label: String,
}

fn main() {
    let args = Args::parse();
    simd::init();

    eprintln!("reading {}...", args.refs.display());
    let file = std::fs::File::open(&args.refs).expect("open refs");
    let gz   = GzDecoder::new(BufReader::new(file));
    let records: Vec<ReferenceRecord> = serde_json::from_reader(gz).expect("parse refs JSON");
    eprintln!("loaded {} records", records.len());

    // Load into stride-16 f32 vectors for k-means (14 dims + 2 padding zeros).
    let mut vectors: Vec<[f32; 16]> = Vec::with_capacity(records.len());
    let mut labels: Vec<u8> = Vec::with_capacity(records.len());
    for r in &records {
        let mut v = [0.0f32; 16];
        for (i, &x) in r.vector.iter().enumerate().take(N_DIMS) { v[i] = x; }
        vectors.push(v);
        labels.push(if r.label == "fraud" { 1 } else { 0 });
    }

    let k = args.clusters;
    let config = KMeansConfig { n_clusters: k, max_iters: args.iters };
    eprintln!("running k-means++ (k={k}, max_iters={})...", args.iters);
    let (centroids_flat, assignments) = fit(&vectors, &config, args.seed);
    eprintln!("k-means done");

    // ── Build cluster posting lists ─────────────────────────────────────────
    let mut cluster_vecs: Vec<Vec<usize>> = vec![vec![]; k];
    for (vi, &ci) in assignments.iter().enumerate() {
        cluster_vecs[ci].push(vi);
    }

    // ── Compute block offsets (CSR, k+1 entries) ────────────────────────────
    let mut offsets = vec![0u32; k + 1];
    for ci in 0..k {
        let n_blocks = (cluster_vecs[ci].len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        offsets[ci + 1] = offsets[ci] + n_blocks as u32;
    }
    let total_blocks = offsets[k] as usize;
    let padded_n     = total_blocks * BLOCK_SIZE;

    // ── Pack vectors into block-strided layout ──────────────────────────────
    // Layout: blocks[b * BLOCK_ELEMS + d * BLOCK_SIZE + slot]
    // Padding slots get i16::MAX (far from any real query distance).
    let mut out_labels: Vec<u8> = vec![0u8; padded_n];
    let mut out_blocks: AVec<i16, ConstAlign<32>> = {
        let mut v = AVec::with_capacity(32, total_blocks * BLOCK_ELEMS);
        for _ in 0..(total_blocks * BLOCK_ELEMS) { v.push(i16::MAX); }
        v
    };

    for ci in 0..k {
        let block_start = offsets[ci] as usize;
        let vecs        = &cluster_vecs[ci];

        for (local_i, &vi) in vecs.iter().enumerate() {
            let b    = local_i / BLOCK_SIZE;
            let slot = local_i % BLOCK_SIZE;
            let bb   = (block_start + b) * BLOCK_ELEMS;

            for d in 0..N_DIMS {
                out_blocks[bb + d * BLOCK_SIZE + slot] = quantize(vectors[vi][d]);
            }
            out_labels[(block_start + b) * BLOCK_SIZE + slot] = labels[vi];
        }
    }

    // ── Transpose centroids to dimension-major layout ───────────────────────
    // centroids_flat: centroid-major [c * 16 + d]
    // centroids_t:    dimension-major [d * k + c]
    let mut centroids_t: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32,N_DIMS * k);
    for _ in 0..(N_DIMS * k) { centroids_t.push(0.0f32); }
    for c in 0..k {
        for d in 0..N_DIMS {
            centroids_t[d * k + c] = centroids_flat[c * 16 + d];
        }
    }

    let index = IvfIndex {
        n_vectors:   records.len() as u32,
        n_dims:      N_DIMS as u16,
        n_clusters:  k as u32,
        quant_scale: QUANT_SCALE,
        nprobe:      args.nprobe,
        centroids:   centroids_t,
        offsets,
        labels:      out_labels,
        blocks:      out_blocks,
    };

    eprintln!("writing {}...", args.out.display());
    index.write(&args.out).expect("write ivfvec");
    eprintln!("done — {} vectors, {} clusters, {} blocks", index.n_vectors, index.n_clusters, total_blocks);
}
