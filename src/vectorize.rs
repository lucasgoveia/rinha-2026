use std::sync::OnceLock;
use serde::Deserialize;

#[derive(Deserialize)]
struct NormJson {
    max_amount: f32,
    max_installments: f32,
    amount_vs_avg_ratio: f32,
    max_minutes: f32,
    max_km: f32,
    max_tx_count_24h: f32,
    max_merchant_avg_amount: f32,
}

struct Norm {
    max_amount: f32,
    max_installments: f32,
    amount_vs_avg_ratio: f32,
    max_minutes: f32,
    max_km: f32,
    max_tx_24h: f32,
    max_merchant_avg: f32,
}

static NORM: OnceLock<Norm> = OnceLock::new();
static MCC: OnceLock<Vec<(u16, f32)>> = OnceLock::new();

pub fn init(norm_path: &str, mcc_path: &str) {
    let norm_json = std::fs::read_to_string(norm_path)
        .unwrap_or_else(|_| panic!("failed to read {}", norm_path));
    let n: NormJson = serde_json::from_str(&norm_json)
        .expect("failed to parse normalization.json");
    NORM.set(Norm {
        max_amount: n.max_amount,
        max_installments: n.max_installments,
        amount_vs_avg_ratio: n.amount_vs_avg_ratio,
        max_minutes: n.max_minutes,
        max_km: n.max_km,
        max_tx_24h: n.max_tx_count_24h,
        max_merchant_avg: n.max_merchant_avg_amount,
    }).ok();

    let mcc_json = std::fs::read_to_string(mcc_path)
        .unwrap_or_else(|_| panic!("failed to read {}", mcc_path));
    let map: std::collections::HashMap<String, f32> = serde_json::from_str(&mcc_json)
        .expect("failed to parse mcc_risk.json");
    let mut table: Vec<(u16, f32)> = map.iter()
        .filter_map(|(k, &v)| k.parse::<u16>().ok().map(|code| (code, v)))
        .collect();
    table.sort_unstable_by_key(|&(code, _)| code);
    MCC.set(table).ok();
}

fn mcc_risk(mcc: u16) -> f32 {
    let table = MCC.get().expect("MCC table not initialized");
    match table.binary_search_by_key(&mcc, |&(m, _)| m) {
        Ok(i) => table[i].1,
        Err(_) => 0.5,
    }
}

#[inline(always)]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline(always)]
fn quantize(v: f32) -> i16 {
    (v * 10000.0).round().clamp(-10000.0, 10000.0) as i16
}

fn find_key(buf: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    let end = buf.len().saturating_sub(needle.len());
    for i in from..=end {
        if buf[i..i + needle.len()] == *needle {
            return Some(i + needle.len());
        }
    }
    None
}

fn skip_ws(buf: &[u8], pos: usize) -> usize {
    let mut i = pos;
    while i < buf.len() && matches!(buf[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

fn parse_f32(buf: &[u8], pos: usize) -> f32 {
    let start = skip_ws(buf, pos);
    let end = buf[start..]
        .iter()
        .position(|&b| !matches!(b, b'0'..=b'9' | b'.' | b'-' | b'e' | b'E' | b'+'))
        .map(|p| start + p)
        .unwrap_or(buf.len());
    std::str::from_utf8(&buf[start..end])
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0)
}

fn parse_i32(buf: &[u8], pos: usize) -> i32 {
    let start = skip_ws(buf, pos);
    let end = buf[start..]
        .iter()
        .position(|&b| !matches!(b, b'0'..=b'9' | b'-'))
        .map(|p| start + p)
        .unwrap_or(buf.len());
    std::str::from_utf8(&buf[start..end])
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

// pos points to char just before the opening quote (or at whitespace before it)
fn parse_quoted(buf: &[u8], pos: usize) -> (&[u8], usize) {
    let start = skip_ws(buf, pos);
    let content_start = if start < buf.len() && buf[start] == b'"' { start + 1 } else { start };
    let end = buf[content_start..]
        .iter()
        .position(|&b| b == b'"')
        .map(|p| content_start + p)
        .unwrap_or(buf.len());
    (&buf[content_start..end], end + 1)
}

// Parse "YYYY-MM-DDTHH:MM:SSZ" — pos points at the opening `"`.
fn parse_timestamp(buf: &[u8], pos: usize) -> (u8, u8, i64) {
    let start = skip_ws(buf, pos);
    let s = if start < buf.len() && buf[start] == b'"' { start + 1 } else { start };
    if s + 19 > buf.len() { return (0, 0, 0); }
    let b = &buf[s..];
    let year = parse_digits4(b, 0);
    let month = parse_digits2(b, 5) as u32;
    let day = parse_digits2(b, 8) as u32;
    let hour = parse_digits2(b, 11);
    let minute = parse_digits2(b, 14);
    let second = parse_digits2(b, 17);
    let weekday = day_of_week(year as i32, month, day);
    let epoch = to_epoch(year as i32, month, day, hour as i32, minute as i32, second as i32);
    (hour, weekday, epoch)
}

fn parse_digits4(b: &[u8], pos: usize) -> u32 {
    ((b[pos] - b'0') as u32) * 1000
        + ((b[pos+1] - b'0') as u32) * 100
        + ((b[pos+2] - b'0') as u32) * 10
        + (b[pos+3] - b'0') as u32
}

fn parse_digits2(b: &[u8], pos: usize) -> u8 {
    (b[pos] - b'0') * 10 + (b[pos+1] - b'0')
}

// Tomohiko Sakamoto: 0=Sun,1=Mon,...,6=Sat → remap to Mon=0..Sun=6
fn day_of_week(y: i32, m: u32, d: u32) -> u8 {
    static T: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    let dow = ((y + y/4 - y/100 + y/400 + T[(m-1) as usize] + d as i32).rem_euclid(7)) as u8;
    if dow == 0 { 6 } else { dow - 1 }
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(m: u32, leap: bool) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if leap { 29 } else { 28 },
        _ => 30,
    }
}

fn to_epoch(y: i32, m: u32, d: u32, h: i32, min: i32, sec: i32) -> i64 {
    let y0 = y - 1970;
    let leap_days = (y0 + 3) / 4 - (y0 + 99) / 100 + (y0 + 399) / 400;
    let mut days = y0 as i64 * 365 + leap_days as i64;
    let leap = is_leap(y);
    for mo in 1..m { days += days_in_month(mo, leap) as i64; }
    days += d as i64 - 1;
    days * 86400 + h as i64 * 3600 + min as i64 * 60 + sec as i64
}

fn fnv1a(s: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in s {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000000001b3);
    }
    h
}

pub fn parse_body(body: &[u8]) -> [i16; 14] {
    let norm = NORM.get().expect("norm not initialized");

    let mut amount = 0.0f32;
    let mut installments = 0i32;
    let mut requested_at_hour = 0u8;
    let mut requested_at_dow = 0u8;
    let mut requested_at_epoch = 0i64;
    let mut customer_avg_amount = 1.0f32;
    let mut tx_count_24h = 0i32;
    let mut known_merchant_hashes = [0u64; 16];
    let mut known_merchant_count = 0usize;
    let mut merchant_id_hash = 0u64;
    let mut merchant_mcc = 0u16;
    let mut merchant_avg_amount = 0.0f32;
    let mut is_online = false;
    let mut card_present = false;
    let mut km_from_home = 0.0f32;
    let mut last_tx_null = true;
    let mut km_from_current = 0.0f32;
    let mut last_tx_epoch = 0i64;

    if let Some(p) = find_key(body, 0, b"\"amount\":") {
        amount = parse_f32(body, p);
    }
    if let Some(p) = find_key(body, 0, b"\"installments\":") {
        installments = parse_i32(body, p);
    }
    if let Some(p) = find_key(body, 0, b"\"requested_at\":") {
        let p2 = skip_ws(body, p);
        let (h, dow, epoch) = parse_timestamp(body, p2);
        requested_at_hour = h;
        requested_at_dow = dow;
        requested_at_epoch = epoch;
    }
    // First "avg_amount" = customer
    if let Some(p) = find_key(body, 0, b"\"avg_amount\":") {
        let v = parse_f32(body, p);
        if v > 0.0 { customer_avg_amount = v; }
    }
    if let Some(p) = find_key(body, 0, b"\"tx_count_24h\":") {
        tx_count_24h = parse_i32(body, p);
    }
    if let Some(arr_start) = find_key(body, 0, b"\"known_merchants\":") {
        let bracket = skip_ws(body, arr_start);
        if bracket < body.len() && body[bracket] == b'[' {
            let mut p = bracket + 1;
            while p < body.len() && known_merchant_count < 16 {
                let p2 = skip_ws(body, p);
                if p2 >= body.len() || body[p2] == b']' { break; }
                if body[p2] == b'"' {
                    let (s, after) = parse_quoted(body, p2 + 1);
                    known_merchant_hashes[known_merchant_count] = fnv1a(s);
                    known_merchant_count += 1;
                    p = after;
                } else {
                    p = p2 + 1;
                }
            }
        }
    }
    if let Some(merchant_pos) = find_key(body, 0, b"\"merchant\":") {
        if let Some(p) = find_key(body, merchant_pos, b"\"id\":") {
            let p2 = skip_ws(body, p);
            let (s, _) = parse_quoted(body, p2 + 1);
            merchant_id_hash = fnv1a(s);
        }
        if let Some(p) = find_key(body, merchant_pos, b"\"mcc\":") {
            let p2 = skip_ws(body, p);
            let (s, _) = parse_quoted(body, p2 + 1);
            let mut code: u16 = 0;
            for &b in s {
                if b.is_ascii_digit() { code = code * 10 + (b - b'0') as u16; }
            }
            merchant_mcc = code;
        }
        // Second "avg_amount" = merchant
        if let Some(p) = find_key(body, merchant_pos, b"\"avg_amount\":") {
            merchant_avg_amount = parse_f32(body, p);
        }
    }
    if let Some(terminal_pos) = find_key(body, 0, b"\"terminal\":") {
        if let Some(p) = find_key(body, terminal_pos, b"\"is_online\":") {
            let p2 = skip_ws(body, p);
            if p2 < body.len() { is_online = body[p2] == b't'; }
        }
        if let Some(p) = find_key(body, terminal_pos, b"\"card_present\":") {
            let p2 = skip_ws(body, p);
            if p2 < body.len() { card_present = body[p2] == b't'; }
        }
        if let Some(p) = find_key(body, terminal_pos, b"\"km_from_home\":") {
            km_from_home = parse_f32(body, p);
        }
    }
    if let Some(p) = find_key(body, 0, b"\"last_transaction\":") {
        let p2 = skip_ws(body, p);
        if p2 < body.len() && body[p2] != b'n' {
            last_tx_null = false;
            if let Some(tp) = find_key(body, p2, b"\"timestamp\":") {
                let tp2 = skip_ws(body, tp);
                let (_, _, epoch) = parse_timestamp(body, tp2);
                last_tx_epoch = epoch;
            }
            if let Some(kp) = find_key(body, p2, b"\"km_from_current\":") {
                km_from_current = parse_f32(body, kp);
            }
        }
    }

    let unknown_merchant = if merchant_id_hash != 0 && known_merchant_count > 0 {
        let found = known_merchant_hashes[..known_merchant_count].iter().any(|&h| h == merchant_id_hash);
        if found { 0.0f32 } else { 1.0 }
    } else {
        1.0
    };

    let (dim5, dim6) = if last_tx_null {
        (-1.0f32, -1.0f32)
    } else {
        let minutes = (requested_at_epoch - last_tx_epoch).max(0) as f32 / 60.0;
        (clamp01(minutes / norm.max_minutes), clamp01(km_from_current / norm.max_km))
    };

    let raw: [f32; 14] = [
        clamp01(amount / norm.max_amount),
        clamp01(installments as f32 / norm.max_installments),
        clamp01((amount / customer_avg_amount) / norm.amount_vs_avg_ratio),
        requested_at_hour as f32 / 23.0,
        requested_at_dow as f32 / 6.0,
        dim5,
        dim6,
        clamp01(km_from_home / norm.max_km),
        clamp01(tx_count_24h as f32 / norm.max_tx_24h),
        if is_online { 1.0 } else { 0.0 },
        if card_present { 1.0 } else { 0.0 },
        unknown_merchant,
        mcc_risk(merchant_mcc),
        clamp01(merchant_avg_amount / norm.max_merchant_avg),
    ];

    raw.map(quantize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_test() {
        let _ = NORM.set(Norm {
            max_amount: 10000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1440.0,
            max_km: 1000.0,
            max_tx_24h: 20.0,
            max_merchant_avg: 10000.0,
        });
        let table = vec![
            (4511u16, 0.35f32), (5311, 0.25), (5411, 0.15),
            (5812, 0.30), (5912, 0.20), (5944, 0.45),
            (5999, 0.50), (7801, 0.80), (7802, 0.75), (7995, 0.85),
        ];
        let _ = MCC.set(table);
    }

    #[test]
    fn test_legit_example() {
        init_test();
        let body = br#"{
          "id": "tx-1329056812",
          "transaction":      { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
          "customer":         { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
          "merchant":         { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
          "terminal":         { "is_online": false, "card_present": true, "km_from_home": 29.23 },
          "last_transaction": null
        }"#;
        let v = parse_body(body);
        let f: Vec<f32> = v.iter().map(|&x| x as f32 / 10000.0).collect();
        assert!((f[0] - 0.0041).abs() < 0.001, "dim0 amount: {}", f[0]);
        assert!((f[1] - 0.1667).abs() < 0.002, "dim1 installments: {}", f[1]);
        assert!((f[2] - 0.05).abs() < 0.001, "dim2 amount_vs_avg: {}", f[2]);
        assert!((f[3] - 0.7826).abs() < 0.002, "dim3 hour: {}", f[3]);
        assert!((f[4] - 0.3333).abs() < 0.002, "dim4 dow: {}", f[4]);
        assert_eq!(v[5], quantize(-1.0), "dim5 null");
        assert_eq!(v[6], quantize(-1.0), "dim6 null");
        assert!((f[7] - 0.0292).abs() < 0.001, "dim7 km_from_home: {}", f[7]);
        assert!((f[8] - 0.15).abs() < 0.002, "dim8 tx_count: {}", f[8]);
        assert_eq!(v[9], 0, "dim9 is_online");
        assert_eq!(v[10], quantize(1.0), "dim10 card_present");
        assert_eq!(v[11], 0, "dim11 unknown_merchant");
        assert!((f[12] - 0.15).abs() < 0.001, "dim12 mcc_risk: {}", f[12]);
        assert!((f[13] - 0.006).abs() < 0.001, "dim13 merchant_avg: {}", f[13]);
    }

    #[test]
    fn test_fraud_example() {
        init_test();
        let body = br#"{
          "id": "tx-3330991687",
          "transaction":      { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
          "customer":         { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
          "merchant":         { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
          "terminal":         { "is_online": false, "card_present": true, "km_from_home": 952.27 },
          "last_transaction": null
        }"#;
        let v = parse_body(body);
        let f: Vec<f32> = v.iter().map(|&x| x as f32 / 10000.0).collect();
        assert!((f[0] - 0.9506).abs() < 0.001, "dim0: {}", f[0]);
        assert!((f[1] - 0.8333).abs() < 0.002, "dim1: {}", f[1]);
        assert!(f[2] >= 1.0 - 0.001, "dim2 clamped: {}", f[2]);
        assert!((f[3] - 0.2174).abs() < 0.002, "dim3 hour: {}", f[3]);
        assert_eq!(v[5], quantize(-1.0), "dim5 null");
        assert_eq!(v[6], quantize(-1.0), "dim6 null");
        assert!((f[7] - 0.9523).abs() < 0.001, "dim7: {}", f[7]);
        assert_eq!(v[9], 0, "dim9");
        assert_eq!(v[10], quantize(1.0), "dim10");
        assert_eq!(v[11], quantize(1.0), "dim11 unknown");
        assert!((f[12] - 0.75).abs() < 0.001, "dim12 mcc 7802: {}", f[12]);
        assert!((f[13] - 0.0055).abs() < 0.001, "dim13: {}", f[13]);
    }
}
