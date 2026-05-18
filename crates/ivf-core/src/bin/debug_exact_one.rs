use flate2::read::GzDecoder;
use serde::Deserialize;
use std::io::BufReader;

#[derive(Deserialize)]
struct TestData {
    entries: Vec<TestEntry>,
}

#[derive(Deserialize)]
struct TestEntry {
    request: TestRequest,
    info: TestInfo,
}

#[derive(Deserialize)]
struct TestRequest {
    id: String,
}

#[derive(Deserialize)]
struct TestInfo {
    vector: [f32; 14],
    expected_response: ExpectedResponse,
}

#[derive(Deserialize)]
struct ExpectedResponse {
    approved: bool,
    fraud_score: f32,
}

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: Vec<f32>,
    label: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let id = std::env::args().nth(1).expect("usage: debug_exact_one <tx-id>");
    let test_path = std::env::args().nth(2).unwrap_or_else(|| "test/test-data.json".into());
    let refs_path = std::env::args().nth(3).unwrap_or_else(|| "resources/references.json.gz".into());

    let data: TestData = serde_json::from_slice(&std::fs::read(test_path)?)?;
    let entry = data
        .entries
        .iter()
        .find(|entry| entry.request.id == id)
        .ok_or("transaction id not found")?;

    let file = std::fs::File::open(refs_path)?;
    let gz = GzDecoder::new(BufReader::new(file));
    let records: Vec<ReferenceRecord> = serde_json::from_reader(gz)?;

    let mut top: [(f32, &str); 5] = [(f32::INFINITY, "legit"); 5];
    let mut worst_idx = 0usize;

    for record in &records {
        let mut dist = 0.0f32;
        for d in 0..14 {
            let diff = entry.info.vector[d] - record.vector[d];
            dist += diff * diff;
        }
        if dist < top[worst_idx].0 {
            top[worst_idx] = (dist, record.label.as_str());
            worst_idx = 0;
            let mut worst = top[0].0;
            for i in 1..5 {
                if top[i].0 > worst {
                    worst = top[i].0;
                    worst_idx = i;
                }
            }
        }
    }

    top.sort_by(|a, b| a.0.total_cmp(&b.0));
    let fraud_count = top.iter().filter(|(_, label)| *label == "fraud").count();
    println!("id={id}");
    println!(
        "expected approved={} fraud_score={}",
        entry.info.expected_response.approved,
        entry.info.expected_response.fraud_score
    );
    println!("exact fraud_count={fraud_count}");
    for (dist, label) in top {
        println!("{dist:.8} {label}");
    }

    Ok(())
}
