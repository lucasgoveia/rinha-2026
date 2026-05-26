use api_lib::{ivf, vectorize};
use serde::Deserialize;

#[derive(Deserialize)]
struct TestData {
    entries: Vec<TestEntry>,
}

#[derive(Deserialize)]
struct TestEntry {
    request: serde_json::Value,
    expected_approved: bool,
}

fn main() {
    let test_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../challenge/test/test-data.json".to_string());
    let index_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/rinha-index.bin".to_string());
    let resources_dir = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "../challenge/resources".to_string());

    vectorize::init(
        &format!("{}/normalization.json", resources_dir),
        &format!("{}/mcc_risk.json", resources_dir),
    );

    eprintln!("[eval] loading test data from {}", test_path);
    let bytes = std::fs::read(&test_path).expect("read test data");
    let data: TestData = serde_json::from_slice(&bytes).expect("parse test data");

    eprintln!("[eval] vectorizing {} entries", data.entries.len());
    let queries: Vec<([i16; 14], bool)> = data
        .entries
        .iter()
        .map(|entry| {
            let body = serde_json::to_vec(&entry.request).expect("serialize request");
            (vectorize::parse_body(&body), entry.expected_approved)
        })
        .collect();

    eprintln!("[eval] loading index from {}", index_path);
    let idx = ivf::IvfIndex::load(&index_path);

    let total_fraud = queries.iter().filter(|(_, approved)| !*approved).count();
    println!("nprobe,total,fraud_expected,false_negative,fn_rate,false_positive,fp_rate,wrong");

    for nprobe in [4usize, 6, 8, 10, 12, 16, 24] {
        let mut false_negative = 0usize;
        let mut false_positive = 0usize;
        let mut wrong = 0usize;

        for (q, expected_approved) in &queries {
            let (fraud_count, _) = idx.query_probe(q, nprobe);
            let approved = fraud_count < 3;

            if approved != *expected_approved {
                wrong += 1;
                if !*expected_approved && approved {
                    false_negative += 1;
                } else if *expected_approved && !approved {
                    false_positive += 1;
                }
            }
        }

        let fn_rate = false_negative as f64 / total_fraud as f64;
        let legit_total = queries.len() - total_fraud;
        let fp_rate = false_positive as f64 / legit_total as f64;
        println!(
            "{},{},{},{},{:.8},{},{:.8},{}",
            nprobe,
            queries.len(),
            total_fraud,
            false_negative,
            fn_rate,
            false_positive,
            fp_rate,
            wrong
        );
    }
}
