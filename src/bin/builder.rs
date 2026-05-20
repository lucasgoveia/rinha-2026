use api_lib::ivf;

fn main() {
    let refs_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/resources/references.json.gz".to_string());
    let out_path = std::env::args().nth(2)
        .unwrap_or_else(|| "/tmp/index.bin".to_string());

    let built = ivf::IvfIndex::build(&refs_path);
    built.save(&out_path);

    let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("[builder] index written to {} ({} MB)", out_path, size / 1_048_576);
}
