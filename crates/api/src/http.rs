use ivf_core::{
    engine::search,
    format::IvfIndex,
    norm::{MerchantRiskConfig, NormalizationConfig},
    response,
    vector::{vectorize, FraudScoreRequest},
};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

pub struct AppState {
    pub index: IvfIndex,
    pub norm: NormalizationConfig,
    pub merchs: MerchantRiskConfig,
    pub nprobe: usize,
}

const BUF_SIZE: usize = 65536;

pub async fn handle(mut stream: TcpStream, state: Arc<AppState>) -> std::io::Result<()> {
    let mut buf = vec![0u8; BUF_SIZE];

    loop {
        let n = stream.read(&mut buf).await?;
        if n == 0 { return Ok(()); }

        let mut headers = [httparse::EMPTY_HEADER; 32];
        let mut req = httparse::Request::new(&mut headers);

        let body_start = match req.parse(&buf[..n]) {
            Ok(httparse::Status::Complete(offset)) => offset,
            _ => {
                stream.write_all(response::BAD_REQUEST).await?;
                return Ok(());
            }
        };

        let body = &buf[body_start..n];

        let resp: &[u8] = match (req.method, req.path) {
            (Some("GET"), Some("/ready")) => response::READY_RESPONSE,
            (Some("POST"), Some("/fraud-score")) => handle_fraud_score(body, &state),
            _ => response::NOT_FOUND,
        };

        stream.write_all(resp).await?;

        let close = req.headers.iter().any(|h| {
            h.name.eq_ignore_ascii_case("connection")
                && std::str::from_utf8(h.value)
                    .unwrap_or("")
                    .eq_ignore_ascii_case("close")
        });
        if close { return Ok(()); }
    }
}

fn handle_fraud_score(body: &[u8], state: &AppState) -> &'static [u8] {
    let req: FraudScoreRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(_) => return response::BAD_REQUEST,
    };
    let query = match vectorize(&req, &state.norm, &state.merchs) {
        Ok(v) => v,
        Err(_) => return response::BAD_REQUEST,
    };
    let fraud_count = search(&query, &state.index, state.nprobe);
    response::RESPONSES[fraud_count]
}
