mod http;

use http::AppState;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use ivf_core::{
    format::IvfIndex,
    norm::{MerchantRiskConfig, NormalizationConfig},
    simd,
};
use mimalloc::MiMalloc;
use std::sync::Arc;
use tokio::net::UnixListener;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
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
        let io = TokioIo::new(stream);
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .keep_alive(true)
                .half_close(false)
                .writev(true)
                .max_buf_size(16 * 1024)
                .preserve_header_case(false)
                .title_case_headers(false)
                .serve_connection(io, service_fn(move |req| http::handle_request(req, Arc::clone(&state))))
                .await
            {
                eprintln!("connection error: {e}");
            }
        });
    }
}
