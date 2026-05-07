// 6 complete HTTP/1.1 responses — indexed by fraud_count (0..=5)
// approved responses: 35-byte body; denied responses: 36-byte body
pub static RESPONSES: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

pub static READY_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n";

pub static NOT_FOUND: &[u8] =
    b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n";

pub static BAD_REQUEST: &[u8] =
    b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\n\r\n";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_cover_all_fraud_counts() {
        assert_eq!(RESPONSES.len(), 6);
    }

    #[test]
    fn response_bodies_valid_json_with_correct_fields() {
        fn body_of(resp: &[u8]) -> &[u8] {
            let sep = b"\r\n\r\n";
            let pos = resp.windows(4).position(|w| w == sep).unwrap();
            &resp[pos + 4..]
        }

        let b0 = body_of(RESPONSES[0]);
        assert!(b0.starts_with(b"{\"approved\":true"));
        assert!(b0.ends_with(b"0.0}"));

        let b3 = body_of(RESPONSES[3]);
        assert!(b3.starts_with(b"{\"approved\":false"));
        assert!(b3.ends_with(b"0.6}"));

        let b5 = body_of(RESPONSES[5]);
        assert!(b5.ends_with(b"1.0}"));
    }

    #[test]
    fn content_length_matches_body() {
        for resp in RESPONSES.iter() {
            let resp_str = std::str::from_utf8(resp).unwrap();
            let header_end = resp_str.find("\r\n\r\n").unwrap();
            let body = &resp[header_end + 4..];
            let cl_line = resp_str[..header_end]
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                .unwrap();
            let cl: usize = cl_line.split(':').nth(1).unwrap().trim().parse().unwrap();
            assert_eq!(cl, body.len(), "content-length mismatch");
        }
    }
}
