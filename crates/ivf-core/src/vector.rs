use crate::norm::{MerchantRiskConfig, NormalizationConfig};
use serde::Deserialize;

pub const N_DIMS: usize = 14;
pub const STRIDE: usize = 16;

#[derive(Debug, Deserialize)]
pub struct FraudScoreRequest {
    pub id: String,
    pub transaction: TransactionReq,
    pub customer: CustomerReq,
    pub merchant: MerchantReq,
    pub terminal: TerminalReq,
    pub last_transaction: Option<LastTransactionReq>,
}

#[derive(Debug, Deserialize)]
pub struct TransactionReq {
    pub amount: f32,
    pub installments: i32,
    pub requested_at: String,
}

#[derive(Debug, Deserialize)]
pub struct CustomerReq {
    pub avg_amount: f32,
    pub tx_count_24h: i32,
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct MerchantReq {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Debug, Deserialize)]
pub struct TerminalReq {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Debug, Deserialize)]
pub struct LastTransactionReq {
    pub timestamp: String,
    pub km_from_current: f32,
}

#[derive(Debug)]
pub struct VectorizeError(pub String);
impl std::fmt::Display for VectorizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self.0) }
}
impl std::error::Error for VectorizeError {}

pub fn vectorize(
    req: &FraudScoreRequest,
    norm: &NormalizationConfig,
    merchs: &MerchantRiskConfig,
) -> Result<[f32; STRIDE], VectorizeError> {
    let requested_at = parse_rfc3339(&req.transaction.requested_at)
        .map_err(|e| VectorizeError(format!("parse requested_at: {e}")))?;

    let (minutes_since_last, km_from_last) = match &req.last_transaction {
        None => (-1.0f32, -1.0f32),
        Some(lt) => {
            let ts = parse_rfc3339(&lt.timestamp)
                .map_err(|e| VectorizeError(format!("parse last_transaction.timestamp: {e}")))?;
            let minutes = requested_at.duration_since(ts).unwrap_or_default().as_secs_f32() / 60.0;
            (clamp_ratio(minutes, norm.max_minutes), clamp_ratio(lt.km_from_current, norm.max_km))
        }
    };

    let unknown_merchant = if req.customer.known_merchants.iter().any(|m| m == &req.merchant.id) {
        0.0f32
    } else {
        1.0
    };

    let mcc_risk = merchs.risk(&req.merchant.mcc);

    let (hour, weekday) = hour_weekday(requested_at);

    let mut v = [0.0f32; STRIDE];
    v[0]  = clamp_ratio(req.transaction.amount, norm.max_amount);
    v[1]  = clamp_ratio(req.transaction.installments as f32, norm.max_installments);
    v[2]  = normalize_amount_vs_avg(req.transaction.amount, req.customer.avg_amount, norm.amount_vs_avg_ratio);
    v[3]  = hour as f32 / 23.0;
    v[4]  = weekday as f32 / 6.0;
    v[5]  = minutes_since_last;
    v[6]  = km_from_last;
    v[7]  = clamp_ratio(req.terminal.km_from_home, norm.max_km);
    v[8]  = clamp_ratio(req.customer.tx_count_24h as f32, norm.max_tx_count_24h);
    v[9]  = if req.terminal.is_online { 1.0 } else { 0.0 };
    v[10] = if req.terminal.card_present { 1.0 } else { 0.0 };
    v[11] = unknown_merchant;
    v[12] = mcc_risk;
    v[13] = clamp_ratio(req.merchant.avg_amount, norm.max_merchant_avg_amount);
    // v[14], v[15] stay 0.0 (stride padding)

    Ok(v)
}

fn clamp_ratio(value: f32, max: f32) -> f32 {
    if max == 0.0 { return 0.0; }
    clamp_unit(value / max)
}

fn clamp_unit(v: f32) -> f32 { v.clamp(0.0, 1.0) }

fn normalize_amount_vs_avg(amount: f32, avg: f32, max_ratio: f32) -> f32 {
    if avg == 0.0 || max_ratio == 0.0 { return 0.0; }
    clamp_unit((amount / avg) / max_ratio)
}

// Returns (hour: u8, weekday: u8) from SystemTime
// weekday: Monday=0 .. Sunday=6  (matches Go's (Weekday()+6)%7)
fn hour_weekday(ts: std::time::SystemTime) -> (u8, u8) {
    use std::time::UNIX_EPOCH;
    let secs = ts.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let hour = ((secs % 86400) / 3600) as u8;
    let days = secs / 86400;
    let go_weekday = (days + 4) % 7;
    let weekday = (go_weekday + 6) % 7;
    (hour, weekday as u8)
}

fn parse_rfc3339(s: &str) -> Result<std::time::SystemTime, Box<dyn std::error::Error>> {
    let s = s.trim_end_matches('Z').trim_end_matches("+00:00");
    let parts: Vec<&str> = s.split('T').collect();
    if parts.len() != 2 { return Err("invalid timestamp format".into()); }
    let date: Vec<u32> = parts[0].split('-').map(|x| x.parse()).collect::<Result<_,_>>()?;
    let time: Vec<u32> = parts[1].split(':').map(|x| x.parse()).collect::<Result<_,_>>()?;
    if date.len() != 3 || time.len() != 3 { return Err("invalid date/time parts".into()); }
    let secs = days_since_epoch(date[0], date[1], date[2]) as u64 * 86400
        + time[0] as u64 * 3600 + time[1] as u64 * 60 + time[2] as u64;
    Ok(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
}

fn days_since_epoch(y: u32, m: u32, d: u32) -> u32 {
    let y = y as i32;
    let m = m as i32;
    let d = d as i32;
    let a = (14 - m) / 12;
    let y2 = y + 4800 - a;
    let m2 = m + 12 * a - 3;
    let jdn = d + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045;
    (jdn - 2440588) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::norm::{MerchantRiskConfig, NormalizationConfig};

    fn test_norm() -> NormalizationConfig {
        NormalizationConfig {
            max_amount: 10000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1440.0,
            max_km: 1000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10000.0,
        }
    }

    fn test_merchs() -> MerchantRiskConfig {
        MerchantRiskConfig::load("../../resources/mcc_risk.json").unwrap()
    }

    #[test]
    fn vectorize_known_payload() {
        let req = FraudScoreRequest {
            id: "tx-1329056812".into(),
            transaction: TransactionReq {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".into(),
            },
            customer: CustomerReq {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".into(), "MERC-016".into()],
            },
            merchant: MerchantReq {
                id: "MERC-016".into(),
                mcc: "5411".into(),
                avg_amount: 60.25,
            },
            terminal: TerminalReq {
                is_online: false,
                card_present: true,
                km_from_home: 29.2331036248,
            },
            last_transaction: None,
        };

        let norm = test_norm();
        let merchs = test_merchs();
        let v = vectorize(&req, &norm, &merchs).unwrap();

        fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 0.0001 }

        assert!(approx(v[0], 0.004112));   // amount / max_amount
        assert!(approx(v[1], 0.16667));    // installments / max_installments
        assert!(approx(v[2], 0.05));       // amount_vs_avg normalized
        assert!(approx(v[3], 0.78261));    // hour 18 / 23
        assert!(approx(v[4], 0.33333));    // Wednesday → (3+6)%7=2, 2/6
        assert_eq!(v[5], -1.0);            // null last_tx → sentinel
        assert_eq!(v[6], -1.0);            // null last_tx → sentinel
        assert!(approx(v[7], 0.02923));    // km_from_home / max_km
        assert!(approx(v[8], 0.15));       // tx_count_24h / max_tx_count_24h
        assert_eq!(v[9], 0.0);             // is_online=false
        assert_eq!(v[10], 1.0);            // card_present=true
        assert_eq!(v[11], 0.0);            // known merchant
        assert!(approx(v[12], 0.15));      // mcc 5411 risk
        assert!(approx(v[13], 0.006025));  // merchant avg_amount / max
        assert_eq!(v[14], 0.0);            // padding
        assert_eq!(v[15], 0.0);            // padding
    }

    #[test]
    fn vectorize_clamps_out_of_range() {
        let norm = test_norm();
        let merchs = test_merchs();
        let req = FraudScoreRequest {
            id: "x".into(),
            transaction: TransactionReq {
                amount: 99999.0,
                installments: 0,
                requested_at: "2026-01-01T00:00:00Z".into(),
            },
            customer: CustomerReq { avg_amount: 100.0, tx_count_24h: 0, known_merchants: vec![] },
            merchant: MerchantReq { id: "X".into(), mcc: "0000".into(), avg_amount: 0.0 },
            terminal: TerminalReq { is_online: false, card_present: false, km_from_home: 0.0 },
            last_transaction: None,
        };
        let v = vectorize(&req, &norm, &merchs).unwrap();
        assert_eq!(v[0], 1.0); // clamped to 1.0
    }
}
