use clap::Parser;
use flate2::read::GzDecoder;
use ivf_core::{
    format::{quantize, IvfIndex},
    kmeans::{fit, KMeansConfig},
    simd,
};
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
    #[arg(long, default_value_t = 2048)]
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
    let gz = GzDecoder::new(BufReader::new(file));
    let records: Vec<ReferenceRecord> = serde_json::from_reader(gz).expect("parse refs JSON");
    eprintln!("loaded {} records", records.len());

    let mut vectors: Vec<[f32; 16]> = Vec::with_capacity(records.len());
    let mut labels: Vec<u8> = Vec::with_capacity(records.len());
    for r in &records {
        let mut v = [0.0f32; 16];
        for (i, &x) in r.vector.iter().enumerate().take(14) { v[i] = x; }
        vectors.push(v);
        labels.push(if r.label == "fraud" { 1 } else { 0 });
    }

    let config = KMeansConfig {
        n_clusters: args.clusters,
        max_iters: args.iters,
        change_tol: 1e-4,
    };
    eprintln!("running k-means++ (k={}, max_iters={})...", args.clusters, args.iters);
    let (centroids_flat, assignments) = fit(&vectors, &config, args.seed);
    eprintln!("k-means done");

    let mut indexed: Vec<(usize, [f32; 16], u8)> = assignments.iter()
        .zip(vectors.iter())
        .zip(labels.iter())
        .map(|((&c, &v), &l)| (c, v, l))
        .collect();
    indexed.sort_unstable_by_key(|&(c, _, _)| c);

    let k = args.clusters;
    let mut offsets = vec![0u32; k];
    let mut sizes = vec![0u32; k];
    for &(c, _, _) in &indexed { sizes[c] += 1; }
    for i in 1..k { offsets[i] = offsets[i - 1] + sizes[i - 1]; }

    let stride = 16usize;
    let mut qvectors: Vec<i16> = Vec::with_capacity(indexed.len() * stride);
    let mut qlabels: Vec<u8>   = Vec::with_capacity(indexed.len());
    // bbox layout: [all_mins: k*stride i16, all_maxs: k*stride i16] — reordered to planar later
    let mut bboxes_mins: Vec<i16> = vec![i16::MAX; k * stride];
    let mut bboxes_maxs: Vec<i16> = vec![i16::MIN; k * stride];

    for (c, v, l) in &indexed {
        let qv: [i16; 16] = std::array::from_fn(|d| quantize(v[d]));
        for d in 0..stride {
            if qv[d] < bboxes_mins[c * stride + d] { bboxes_mins[c * stride + d] = qv[d]; }
            if qv[d] > bboxes_maxs[c * stride + d] { bboxes_maxs[c * stride + d] = qv[d]; }
        }
        qvectors.extend_from_slice(&qv);
        qlabels.push(*l);
    }

    // Reorder to per-cluster planar: [cluster0_mins[16], cluster0_maxs[16], cluster1_mins[16], ...]
    let mut bboxes_planar: Vec<i16> = Vec::with_capacity(k * stride * 2);
    for c in 0..k {
        bboxes_planar.extend_from_slice(&bboxes_mins[c * stride..(c + 1) * stride]);
        bboxes_planar.extend_from_slice(&bboxes_maxs[c * stride..(c + 1) * stride]);
    }

    let index = IvfIndex {
        n_vectors: indexed.len() as u32,
        n_dims: 14,
        stride: 16,
        quant_scale: 10000.0,
        n_clusters: k as u32,
        nprobe: args.nprobe,
        centroids: centroids_flat,
        bboxes: bboxes_planar,
        offsets,
        sizes,
        labels: qlabels,
        vectors: qvectors,
    };

    eprintln!("writing {}...", args.out.display());
    index.write(&args.out).expect("write ivfvec");
    eprintln!("done — {} vectors, {} clusters", index.n_vectors, index.n_clusters);
}
