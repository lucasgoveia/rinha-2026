use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{header, header::HeaderValue, Method, Request, Response, StatusCode};
use hyper::body::Incoming;
use ivf_core::{
    engine::search,
    format::IvfIndex,
    norm::{MerchantRiskConfig, NormalizationConfig},
    vector::{vectorize, FraudScoreRequest},
};
use std::convert::Infallible;
use std::sync::{Arc, LazyLock};

pub struct AppState {
    pub index: IvfIndex,
    pub norm: NormalizationConfig,
    pub merchs: MerchantRiskConfig,
    pub nprobe: usize,
}

static FRAUD_BODIES: LazyLock<[Bytes; 6]> = LazyLock::new(|| [
    Bytes::from_static(b"{\"approved\":true,\"fraud_score\":0.0}"),
    Bytes::from_static(b"{\"approved\":true,\"fraud_score\":0.2}"),
    Bytes::from_static(b"{\"approved\":true,\"fraud_score\":0.4}"),
    Bytes::from_static(b"{\"approved\":false,\"fraud_score\":0.6}"),
    Bytes::from_static(b"{\"approved\":false,\"fraud_score\":0.8}"),
    Bytes::from_static(b"{\"approved\":false,\"fraud_score\":1.0}"),
]);

static JSON_CT: LazyLock<HeaderValue> =
    LazyLock::new(|| HeaderValue::from_static("application/json"));

pub async fn handle_request(
    req: Request<Incoming>,
    state: Arc<AppState>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let resp = match (req.method(), req.uri().path()) {
        (&Method::GET, "/ready") => ready_response(),
        (&Method::POST, "/fraud-score") => handle_fraud_score(req, &state).await,
        _ => not_found_response(),
    };
    Ok(resp)
}

async fn handle_fraud_score(req: Request<Incoming>, state: &AppState) -> Response<Full<Bytes>> {
    let body = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => return bad_request_response(),
    };
    let fraud_req: FraudScoreRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(_) => return bad_request_response(),
    };
    let query = match vectorize(&fraud_req, &state.norm, &state.merchs) {
        Ok(v) => v,
        Err(_) => return bad_request_response(),
    };
    let fraud_count = search(&query, &state.index, state.nprobe);
    fraud_score_response(fraud_count)
}

fn fraud_score_response(n: usize) -> Response<Full<Bytes>> {
    let mut resp = Response::new(Full::new(FRAUD_BODIES[n].clone()));
    resp.headers_mut().insert(header::CONTENT_TYPE, JSON_CT.clone());
    resp
}

fn ready_response() -> Response<Full<Bytes>> {
    Response::new(Full::new(Bytes::new()))
}

fn not_found_response() -> Response<Full<Bytes>> {
    let mut resp = Response::new(Full::new(Bytes::new()));
    *resp.status_mut() = StatusCode::NOT_FOUND;
    resp
}

fn bad_request_response() -> Response<Full<Bytes>> {
    let mut resp = Response::new(Full::new(Bytes::new()));
    *resp.status_mut() = StatusCode::BAD_REQUEST;
    resp
}
