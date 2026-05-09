mod http;

use http::AppState;
use ivf_core::{
    engine,
    format::IvfIndex,
    norm::{MerchantRiskConfig, NormalizationConfig},
    simd,
};
use mimalloc::MiMalloc;
use std::sync::Arc;
use tokio::net::UnixListener;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    simd::init();

    let resources = std::env::var("RESOURCES_PATH").unwrap_or_else(|_| "resources".into());
    let nprobe: usize = std::env::var("NPROBE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let sock_path = std::env::var("SOCK").expect("SOCK env required");

    let norm = NormalizationConfig::load(format!("{resources}/normalization.json"))
        .expect("load normalization.json");
    let merchs = MerchantRiskConfig::load(format!("{resources}/mcc_risk.json"))
        .expect("load mcc_risk.json");

    eprintln!("loading {resources}/references.ivfvec...");
    let index = IvfIndex::load(format!("{resources}/references.ivfvec"))
        .expect("load references.ivfvec");
    eprintln!("loaded {} vectors, {} clusters", index.n_vectors, index.n_clusters);

    // Warmup: random queries prime L1/L2/TLB and branch predictor.
    {
        let mut state = 0x12345678u32;
        for _ in 0..1000 {
            let mut q = [0.0f32; 16];
            for v in q.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = (state >> 8) as f32 / (1u32 << 24) as f32;
            }
            let _ = engine::search(&q, &index, 0);
        }
        eprintln!("warmup done");
    }

    let state = Arc::new(AppState { index, norm, merchs, nprobe });

    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).expect("bind unix socket");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o666))
            .expect("chmod socket");
    }
    eprintln!("listening on {sock_path}");

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
