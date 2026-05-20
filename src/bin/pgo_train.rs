// PGO training binary. Exercises the hot query() loop with representative
// vectors so llvm-profdata captures branch/inlining patterns matching real
// fraud-score traffic.

use api_lib::ivf;
use flate2::read::GzDecoder;
use serde::Deserialize;

#[derive(Deserialize)]
struct RefEntry {
    vector: [f32; 14],
    #[allow(dead_code)]
    label: String,
}

const SCALE: f32 = 10000.0;
const REPEATS: usize = 4;
const SAMPLE_STRIDE: usize = 5;

fn quantize(v: &[f32; 14]) -> [i16; 14] {
    v.map(|x| (x * SCALE).round().clamp(-SCALE, SCALE) as i16)
}

fn main() {
    let refs_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/resources/references.json.gz".to_string());
    let index_path = std::env::args().nth(2)
        .unwrap_or_else(|| "/tmp/index.bin".to_string());

    eprintln!("[pgo_train] loading references from {}", refs_path);
    let file = std::fs::File::open(&refs_path).expect("open refs");
    let gz = GzDecoder::new(file);
    let entries: Vec<RefEntry> = serde_json::from_reader(gz).expect("parse refs");
    let queries: Vec<[i16; 14]> = entries.iter()
        .step_by(SAMPLE_STRIDE)
        .map(|e| quantize(&e.vector))
        .collect();
    eprintln!("[pgo_train] {} training queries", queries.len());

    eprintln!("[pgo_train] loading index from {}", index_path);
    let idx = ivf::IvfIndex::load(&index_path);

    let mut sum: u64 = 0;
    for r in 0..REPEATS {
        for q in &queries {
            sum = sum.wrapping_add(idx.query(q) as u64);
        }
        eprintln!("[pgo_train] pass {}/{} done", r + 1, REPEATS);
    }
    eprintln!("[pgo_train] done (checksum {})", sum);
}
