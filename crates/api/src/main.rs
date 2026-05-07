mod http;

use http::AppState;
use ivf_core::{
    format::IvfIndex,
    norm::{MerchantRiskConfig, NormalizationConfig},
    simd,
};
use std::sync::Arc;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    simd::init();

    let resources = std::env::var("RESOURCES_PATH").unwrap_or_else(|_| "resources".into());
    let nprobe: usize = std::env::var("NPROBE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let port = std::env::var("PORT").unwrap_or_else(|_| "9999".into());

    let norm = NormalizationConfig::load(format!("{resources}/normalization.json"))
        .expect("load normalization.json");
    let merchs = MerchantRiskConfig::load(format!("{resources}/mcc_risk.json"))
        .expect("load mcc_risk.json");

    eprintln!("loading {resources}/references.ivfvec...");
    let index = IvfIndex::load(format!("{resources}/references.ivfvec"))
        .expect("load references.ivfvec");
    eprintln!("loaded {} vectors, {} clusters", index.n_vectors, index.n_clusters);

    let state = Arc::new(AppState { index, norm, merchs, nprobe });

    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
    eprintln!("listening on :{port}");

    loop {
        let (stream, _) = listener.accept().await.unwrap();
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = http::handle(stream, state).await {
                eprintln!("connection error: {e}");
            }
        });
    }
}
