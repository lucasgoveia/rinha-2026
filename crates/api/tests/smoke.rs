// Integration test — requires a running api server at 127.0.0.1:9999
// Run with: cargo test -p api --test smoke -- --ignored

use std::net::TcpStream;
use std::io::{Read, Write};

fn send_request(addr: &str, method: &str, path: &str, body: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    let req = if body.is_empty() {
        format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
    } else {
        format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
    };
    stream.write_all(req.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}

#[test]
#[ignore]
fn ready_returns_200() {
    let resp = send_request("127.0.0.1:9999", "GET", "/ready", "");
    assert!(resp.starts_with("HTTP/1.1 200"));
}

#[test]
#[ignore]
fn fraud_score_returns_valid_response() {
    let body = r#"{
        "id":"test-1",
        "transaction":{"amount":100.0,"installments":1,"requested_at":"2026-03-11T12:00:00Z"},
        "customer":{"avg_amount":100.0,"tx_count_24h":2,"known_merchants":["M1"]},
        "merchant":{"id":"M1","mcc":"5411","avg_amount":90.0},
        "terminal":{"is_online":false,"card_present":true,"km_from_home":5.0},
        "last_transaction":null
    }"#;
    let resp = send_request("127.0.0.1:9999", "POST", "/fraud-score", body);
    assert!(resp.starts_with("HTTP/1.1 200"));
    assert!(resp.contains("approved"));
    assert!(resp.contains("fraud_score"));
}
