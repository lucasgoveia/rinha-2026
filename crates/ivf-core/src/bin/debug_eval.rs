use ivf_core::{
    engine::{search, search_nprobe},
    format::IvfIndex,
    norm::{MerchantRiskConfig, NormalizationConfig},
    simd,
    vector::{vectorize, FraudScoreRequest, STRIDE},
};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct TestData {
    entries: Vec<TestEntry>,
}

#[derive(Deserialize)]
struct TestEntry {
    request: FraudScoreRequest,
    info: TestInfo,
}

#[derive(Deserialize)]
struct TestInfo {
    vector: [f32; 14],
    expected_response: ExpectedResponse,
}

#[derive(Deserialize)]
struct ExpectedResponse {
    approved: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    simd::init();

    let args: Vec<String> = std::env::args().collect();
    let test_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("test/test-data.json"));
    let resources = args.get(2).map(String::as_str).unwrap_or("resources");
    let mode = args.get(3).map(String::as_str).unwrap_or("heuristic");

    let norm = NormalizationConfig::load(format!("{resources}/normalization.json"))?;
    let merchs = MerchantRiskConfig::load(format!("{resources}/mcc_risk.json"))?;
    let index = IvfIndex::load(format!("{resources}/references.ivfvec"))?;
    let data: TestData = serde_json::from_slice(&std::fs::read(test_path)?)?;

    let mut vector_mismatches = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut total = 0usize;
    let mut printed_errors = 0usize;

    for entry in &data.entries {
        total += 1;
        let got_vec = vectorize(&entry.request, &norm, &merchs)?;
        if !same_vector(&got_vec, &entry.info.vector) {
            vector_mismatches += 1;
            if vector_mismatches <= 5 {
                eprintln!(
                    "vector mismatch id={} got={:?} expected={:?}",
                    entry.request.id,
                    &got_vec[..14],
                    entry.info.vector
                );
            }
        }

        let fraud_count = match mode.parse::<usize>() {
            Ok(nprobe) => search_nprobe(&got_vec, &index, nprobe),
            Err(_) => search(&got_vec, &index, 0),
        };
        let approved = fraud_count < 3;
        match (approved, entry.info.expected_response.approved) {
            (false, true) => {
                fp += 1;
                printed_errors = print_error_sample(
                    printed_errors,
                    "FP",
                    &entry.request.id,
                    fraud_count,
                    &got_vec,
                    &index,
                );
            }
            (true, false) => {
                fn_ += 1;
                printed_errors = print_error_sample(
                    printed_errors,
                    "FN",
                    &entry.request.id,
                    fraud_count,
                    &got_vec,
                    &index,
                );
            }
            _ => {}
        }
    }

    println!("total={total}");
    println!("vector_mismatches={vector_mismatches}");
    println!("false_positive={fp}");
    println!("false_negative={fn_}");
    println!("errors={}", fp + fn_);
    Ok(())
}

fn print_error_sample(
    printed_errors: usize,
    kind: &str,
    id: &str,
    fraud_count: usize,
    query: &[f32; STRIDE],
    index: &IvfIndex,
) -> usize {
    if printed_errors >= 5 {
        return printed_errors;
    }
    let exact_count = search_nprobe(query, index, index.n_clusters as usize);
    eprintln!(
        "{kind} id={id} heuristic_count={fraud_count} all_clusters_count={exact_count}"
    );
    printed_errors + 1
}

fn same_vector(got: &[f32; STRIDE], expected: &[f32; 14]) -> bool {
    got.iter()
        .take(14)
        .zip(expected)
        .all(|(a, b)| (a - b).abs() <= 0.00011)
}
