# Rust IVF Fraud Detection Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Go fraud detection API in Rust using a Cargo workspace with three crates: `ivf-core` (IVF engine library), `refvec-builder` (index builder binary), and `api` (tokio HTTP server binary).

**Architecture:** IVF index with 2048 k-means clusters, asymmetric L2 distance (query=f32, DB=i16 with scale 10000), bbox cluster pruning, fixed-size NeighborHeap<5> for top-K tracking. Six preallocated `&'static [u8]` full HTTP/1.1 responses cover all possible outcomes.

**Tech Stack:** Rust 2021 edition, tokio 1.x, httparse 1.x, serde_json 1.x, rayon 1.x, flate2 1.x, clap 4.x, std::arch (AVX2/AVX-512 intrinsics)

**Spec:** `docs/superpowers/specs/2026-05-06-rust-ivf-rewrite-design.md`

---

## File Map

```
Cargo.toml                               new — workspace root
crates/ivf-core/Cargo.toml               new
crates/ivf-core/src/lib.rs               new — pub re-exports
crates/ivf-core/src/norm.rs              new — NormalizationConfig, MerchantRiskConfig
crates/ivf-core/src/vector.rs            new — FraudScoreRequest types, vectorize()
crates/ivf-core/src/format.rs            new — IvfIndex struct, binary read/write
crates/ivf-core/src/simd.rs              new — asym_l2, l2_f32, bbox_min_l2 (AVX2/512)
crates/ivf-core/src/engine.rs            new — NeighborHeap<N>, IvfIndex::search()
crates/ivf-core/src/response.rs          new — 6 static HTTP responses + READY/NOT_FOUND/BAD_REQUEST
crates/ivf-core/src/kmeans.rs            new — k-means++ builder (rayon-parallel)
crates/refvec-builder/Cargo.toml         new
crates/refvec-builder/src/main.rs        new — CLI pipeline: read .json.gz → cluster → write .ivfvec
crates/api/Cargo.toml                    new
crates/api/src/main.rs                   new — tokio TcpListener + accept loop
crates/api/src/http.rs                   new — httparse request parsing, connection handler
Dockerfile                               modify — replace Zig/Go build with Rust
```

---

## Task 1: Cargo Workspace Scaffold

**Files:**
- Create: `Cargo.toml`
- Create: `crates/ivf-core/Cargo.toml`
- Create: `crates/ivf-core/src/lib.rs`
- Create: `crates/refvec-builder/Cargo.toml`
- Create: `crates/refvec-builder/src/main.rs`
- Create: `crates/api/Cargo.toml`
- Create: `crates/api/src/main.rs`

- [ ] **Step 1: Create workspace Cargo.toml**

```toml
[workspace]
members = ["crates/ivf-core", "crates/refvec-builder", "crates/api"]
resolver = "2"
```

- [ ] **Step 2: Create crates/ivf-core/Cargo.toml**

```toml
[package]
name = "ivf-core"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rayon = "1"
flate2 = "1"
```

- [ ] **Step 3: Create crates/ivf-core/src/lib.rs**

```rust
pub mod engine;
pub mod format;
pub mod kmeans;
pub mod norm;
pub mod response;
pub mod simd;
pub mod vector;
```

- [ ] **Step 4: Create crates/refvec-builder/Cargo.toml**

```toml
[package]
name = "refvec-builder"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "refvec-builder"
path = "src/main.rs"

[dependencies]
ivf-core = { path = "../ivf-core" }
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 5: Create crates/refvec-builder/src/main.rs (stub)**

```rust
fn main() {
    println!("refvec-builder stub");
}
```

- [ ] **Step 6: Create crates/api/Cargo.toml**

```toml
[package]
name = "api"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "api"
path = "src/main.rs"

[dependencies]
ivf-core = { path = "../ivf-core" }
tokio = { version = "1", features = ["net", "io-util", "rt-multi-thread", "macros"] }
httparse = "1"
serde_json = "1"

[features]
default = []
otel = [
    "dep:opentelemetry",
    "dep:opentelemetry-otlp",
    "dep:opentelemetry_sdk",
    "dep:tracing",
    "dep:tracing-opentelemetry",
    "dep:tracing-subscriber",
]

[dependencies.opentelemetry]
version = "0.27"
optional = true

[dependencies.opentelemetry-otlp]
version = "0.27"
optional = true

[dependencies.opentelemetry_sdk]
version = "0.27"
optional = true

[dependencies.tracing]
version = "0.1"
optional = true

[dependencies.tracing-opentelemetry]
version = "0.28"
optional = true

[dependencies.tracing-subscriber]
version = "0.3"
optional = true
```

- [ ] **Step 7: Create crates/api/src/main.rs (stub)**

```rust
fn main() {
    println!("api stub");
}
```

- [ ] **Step 8: Verify workspace compiles**

```
cargo build --workspace
```

Expected: compiles with zero errors (two stub binaries).

- [ ] **Step 9: Commit**

```
git add Cargo.toml crates/
git commit -m "feat: scaffold Rust workspace with ivf-core, refvec-builder, api crates"
```

---

## Task 2: `norm.rs` — NormalizationConfig + MerchantRiskConfig

**Files:**
- Create: `crates/ivf-core/src/norm.rs`

- [ ] **Step 1: Write failing test**

```rust
// Bottom of crates/ivf-core/src/norm.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_normalization_config() {
        let config = NormalizationConfig::load("../../resources/normalization.json").unwrap();
        assert_eq!(config.max_amount, 10000.0);
        assert_eq!(config.max_installments, 12.0);
        assert_eq!(config.amount_vs_avg_ratio, 10.0);
        assert_eq!(config.max_minutes, 1440.0);
        assert_eq!(config.max_km, 1000.0);
        assert_eq!(config.max_tx_count_24h, 20.0);
        assert_eq!(config.max_merchant_avg_amount, 10000.0);
    }

    #[test]
    fn load_merchant_risk_config() {
        let config = MerchantRiskConfig::load("../../resources/mcc_risk.json").unwrap();
        assert_eq!(config.risk("5411"), 0.15);
        assert_eq!(config.risk("7995"), 0.85);
        assert_eq!(config.risk("9999"), 0.5); // unknown → default 0.5
    }
}
```

- [ ] **Step 2: Run test — verify it fails**

```
cargo test -p ivf-core norm
```

Expected: compile error (types not defined yet).

- [ ] **Step 3: Implement norm.rs**

```rust
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct NormalizationConfig {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

impl NormalizationConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct MerchantRiskConfig(HashMap<String, f32>);

impl MerchantRiskConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn risk(&self, mcc: &str) -> f32 {
        *self.0.get(mcc).unwrap_or(&0.5)
    }
}
```

- [ ] **Step 4: Run tests — verify pass**

```
cargo test -p ivf-core norm
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```
git add crates/ivf-core/src/norm.rs
git commit -m "feat(ivf-core): add NormalizationConfig and MerchantRiskConfig"
```

---

## Task 3: `vector.rs` — FraudScoreRequest + vectorize()

**Files:**
- Create: `crates/ivf-core/src/vector.rs`

- [ ] **Step 1: Write failing tests**

Port the exact logic from Go's `vector.go`. Test with the known example payload.

```rust
// Bottom of crates/ivf-core/src/vector.rs

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
        // tx-1329056812 from resources/example-payloads.json
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
        let mut req = FraudScoreRequest {
            id: "x".into(),
            transaction: TransactionReq {
                amount: 99999.0, // above max
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
```

- [ ] **Step 2: Run test — verify it fails**

```
cargo test -p ivf-core vector
```

Expected: compile error (types not defined).

- [ ] **Step 3: Implement vector.rs**

```rust
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
            let minutes = (requested_at - ts).as_secs_f32() / 60.0;
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

// Returns (hour: u8, weekday: u8) from Unix timestamp
// weekday: Monday=0 .. Sunday=6  (matches Go's (Weekday()+6)%7)
fn hour_weekday(ts: std::time::SystemTime) -> (u8, u8) {
    use std::time::UNIX_EPOCH;
    let secs = ts.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let hour = ((secs % 86400) / 3600) as u8;
    // Days since epoch; epoch was Thursday (Go: 4, our adjusted: (4+6)%7=3 → Monday=0 mapping)
    // Go: Sunday=0, Monday=1, ...; adjusted = (go_weekday + 6) % 7 → Monday=0, Sunday=6
    let days = secs / 86400;
    let go_weekday = (days + 4) % 7; // epoch was Thursday, Go's Thursday=4
    let weekday = (go_weekday + 6) % 7;
    (hour, weekday as u8)
}

fn parse_rfc3339(s: &str) -> Result<std::time::SystemTime, Box<dyn std::error::Error>> {
    // Parse RFC 3339 / ISO 8601 UTC timestamp without external deps
    // Expected format: "2026-03-11T18:45:53Z" or with +00:00 offset
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
    // Days from 1970-01-01 to y-m-d (Gregorian)
    let y = y as i32;
    let m = m as i32;
    let d = d as i32;
    let a = (14 - m) / 12;
    let y2 = y + 4800 - a;
    let m2 = m + 12 * a - 3;
    let jdn = d + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045;
    (jdn - 2440588) as u32 // 2440588 = JDN of 1970-01-01
}
```

- [ ] **Step 4: Run tests — verify pass**

```
cargo test -p ivf-core vector
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```
git add crates/ivf-core/src/vector.rs
git commit -m "feat(ivf-core): add FraudScoreRequest types and vectorize()"
```

---

## Task 4: `format.rs` — IvfIndex Binary Read/Write

**Files:**
- Create: `crates/ivf-core/src/format.rs`

- [ ] **Step 1: Write failing test**

```rust
// Bottom of crates/ivf-core/src/format.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile; // add to [dev-dependencies]: tempfile = "3"

    fn make_small_index() -> IvfIndex {
        let n_clusters = 4u32;
        let stride = 16usize;
        let n_vectors = 8u32;
        IvfIndex {
            n_vectors,
            n_dims: 14,
            stride: stride as u16,
            quant_scale: 10000.0,
            n_clusters,
            nprobe: 2,
            centroids: vec![0.5f32; n_clusters as usize * stride],
            bboxes: vec![0i16; n_clusters as usize * stride * 2],
            offsets: vec![0, 2, 4, 6],
            sizes: vec![2, 2, 2, 2],
            labels: vec![0, 1, 0, 1, 1, 0, 0, 1],
            vectors: vec![1000i16; n_vectors as usize * stride],
        }
    }

    #[test]
    fn round_trip_write_load() {
        let original = make_small_index();
        let mut tmp = NamedTempFile::new().unwrap();
        original.write(tmp.path()).unwrap();
        let loaded = IvfIndex::load(tmp.path()).unwrap();

        assert_eq!(loaded.n_vectors, original.n_vectors);
        assert_eq!(loaded.n_clusters, original.n_clusters);
        assert_eq!(loaded.nprobe, original.nprobe);
        assert_eq!(loaded.quant_scale, original.quant_scale);
        assert_eq!(loaded.centroids, original.centroids);
        assert_eq!(loaded.bboxes, original.bboxes);
        assert_eq!(loaded.offsets, original.offsets);
        assert_eq!(loaded.sizes, original.sizes);
        assert_eq!(loaded.labels, original.labels);
        assert_eq!(loaded.vectors, original.vectors);
    }

    #[test]
    fn cluster_returns_correct_slice() {
        let idx = make_small_index();
        let (vecs, labs) = idx.cluster(1);
        assert_eq!(labs, &[0u8, 1u8]);
        assert_eq!(vecs.len(), 2 * 16);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut data = vec![0u8; 200];
        data[0..8].copy_from_slice(b"GARBAGE!");
        assert!(IvfIndex::from_bytes(&data).is_err());
    }
}
```

Add to `crates/ivf-core/Cargo.toml`:
```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Run test — verify it fails**

```
cargo test -p ivf-core format
```

Expected: compile error.

- [ ] **Step 3: Implement format.rs**

```rust
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"IVFVEC01";
pub const VERSION: u8 = 1;
pub const QUANT_SCALE: f32 = 10000.0;

#[derive(Debug)]
pub struct IvfIndex {
    pub n_vectors: u32,
    pub n_dims: u16,
    pub stride: u16,
    pub quant_scale: f32,
    pub n_clusters: u32,
    pub nprobe: u16,
    // centroids: n_clusters × stride f32, contiguous
    pub centroids: Vec<f32>,
    // bboxes: n_clusters × stride × 2 i16, layout: [mins[stride], maxs[stride]] per cluster
    pub bboxes: Vec<i16>,
    pub offsets: Vec<u32>,
    pub sizes: Vec<u32>,
    pub labels: Vec<u8>,
    // vectors: n_vectors × stride i16, cluster-ordered
    pub vectors: Vec<i16>,
}

impl IvfIndex {
    pub fn cluster(&self, cid: usize) -> (&[i16], &[u8]) {
        let s = self.stride as usize;
        let off = self.offsets[cid] as usize;
        let sz = self.sizes[cid] as usize;
        (&self.vectors[off * s..(off + sz) * s], &self.labels[off..off + sz])
    }

    pub fn centroid(&self, cid: usize) -> &[f32] {
        let s = self.stride as usize;
        &self.centroids[cid * s..(cid + 1) * s]
    }

    pub fn bbox(&self, cid: usize) -> (&[i16], &[i16]) {
        let s = self.stride as usize;
        let base = cid * s * 2;
        (&self.bboxes[base..base + s], &self.bboxes[base + s..base + 2 * s])
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::{BufWriter, Write};
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);

        w.write_all(MAGIC)?;
        w.write_all(&[VERSION])?;
        w.write_all(&self.n_vectors.to_le_bytes())?;
        w.write_all(&self.n_dims.to_le_bytes())?;
        w.write_all(&self.stride.to_le_bytes())?;
        w.write_all(&self.quant_scale.to_bits().to_le_bytes())?;
        w.write_all(&self.n_clusters.to_le_bytes())?;
        w.write_all(&self.nprobe.to_le_bytes())?;

        for &v in &self.centroids { w.write_all(&v.to_bits().to_le_bytes())?; }
        for &v in &self.bboxes    { w.write_all(&(v as u16).to_le_bytes())?; }
        for &v in &self.offsets   { w.write_all(&v.to_le_bytes())?; }
        for &v in &self.sizes     { w.write_all(&v.to_le_bytes())?; }
        w.write_all(&self.labels)?;
        for &v in &self.vectors   { w.write_all(&(v as u16).to_le_bytes())?; }

        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if data.len() < 26 { return Err("file too short".into()); }
        if &data[0..8] != MAGIC { return Err(format!("bad magic: {:?}", &data[0..8]).into()); }
        if data[8] != VERSION { return Err(format!("unsupported version {}", data[8]).into()); }

        let mut pos = 9;
        macro_rules! read_u16 { () => {{ let v = u16::from_le_bytes(data[pos..pos+2].try_into()?); pos += 2; v }}; }
        macro_rules! read_u32 { () => {{ let v = u32::from_le_bytes(data[pos..pos+4].try_into()?); pos += 4; v }}; }
        macro_rules! read_f32 { () => {{ let v = f32::from_bits(u32::from_le_bytes(data[pos..pos+4].try_into()?)); pos += 4; v }}; }

        let n_vectors  = read_u32!();
        let n_dims     = read_u16!();
        let stride     = read_u16!();
        let quant_scale = read_f32!();
        let n_clusters = read_u32!();
        let nprobe     = read_u16!();

        let k = n_clusters as usize;
        let s = stride as usize;
        let n = n_vectors as usize;

        let centroids = read_f32_vec(&data, &mut pos, k * s)?;
        let bboxes    = read_i16_vec(&data, &mut pos, k * s * 2)?;
        let offsets   = read_u32_vec(&data, &mut pos, k)?;
        let sizes     = read_u32_vec(&data, &mut pos, k)?;
        let labels    = data[pos..pos + n].to_vec(); pos += n;
        let vectors   = read_i16_vec(&data, &mut pos, n * s)?;

        Ok(IvfIndex { n_vectors, n_dims, stride, quant_scale, n_clusters, nprobe,
                      centroids, bboxes, offsets, sizes, labels, vectors })
    }
}

pub fn quantize(v: f32) -> i16 {
    (v * QUANT_SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn read_f32_vec(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = count * 4;
    if *pos + bytes > data.len() { return Err("truncated f32 section".into()); }
    let v = (0..count).map(|i| {
        let p = *pos + i * 4;
        f32::from_bits(u32::from_le_bytes(data[p..p+4].try_into().unwrap()))
    }).collect();
    *pos += bytes;
    Ok(v)
}

fn read_i16_vec(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    let bytes = count * 2;
    if *pos + bytes > data.len() { return Err("truncated i16 section".into()); }
    let v = (0..count).map(|i| {
        let p = *pos + i * 2;
        i16::from_le_bytes(data[p..p+2].try_into().unwrap())
    }).collect();
    *pos += bytes;
    Ok(v)
}

fn read_u32_vec(data: &[u8], pos: &mut usize, count: usize) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let bytes = count * 4;
    if *pos + bytes > data.len() { return Err("truncated u32 section".into()); }
    let v = (0..count).map(|i| {
        let p = *pos + i * 4;
        u32::from_le_bytes(data[p..p+4].try_into().unwrap())
    }).collect();
    *pos += bytes;
    Ok(v)
}
```

- [ ] **Step 4: Run tests — verify pass**

```
cargo test -p ivf-core format
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```
git add crates/ivf-core/src/format.rs crates/ivf-core/Cargo.toml
git commit -m "feat(ivf-core): add IvfIndex binary read/write format"
```

---

## Task 5: `simd.rs` — Distance Functions (AVX2/512 dispatch)

**Files:**
- Create: `crates/ivf-core/src/simd.rs`

- [ ] **Step 1: Write failing tests**

```rust
// Bottom of crates/ivf-core/src/simd.rs

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_asym_l2(q: &[f32; 16], db: &[i16; 16]) -> f32 {
        (0..16).map(|d| { let diff = q[d] - db[d] as f32 / 10000.0; diff * diff }).sum()
    }

    fn scalar_l2_f32(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        (0..16).map(|d| { let diff = a[d] - b[d]; diff * diff }).sum()
    }

    #[test]
    fn asym_l2_matches_scalar() {
        let q: [f32; 16] = [0.5, 0.3, 0.1, 0.8, 0.2, -1.0, -1.0, 0.05, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006, 0.0, 0.0];
        let db: [i16; 16] = [5000, 3000, 1000, 8000, 2000, -10000, -10000, 500, 1500, 0, 10000, 0, 1500, 60, 0, 0];
        let expected = scalar_asym_l2(&q, &db);
        let got = ASYM_L2.get().unwrap()(&q, &db);
        assert!((got - expected).abs() < 1e-6, "asym_l2: got {got} expected {expected}");
    }

    #[test]
    fn l2_f32_matches_scalar() {
        let a: [f32; 16] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0];
        let b: [f32; 16] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1.0, 0.9, 0.8, 0.7, 0.0, 0.0];
        let expected = scalar_l2_f32(&a, &b);
        let got = L2_F32.get().unwrap()(&a, &b);
        assert!((got - expected).abs() < 1e-6, "l2_f32: got {got} expected {expected}");
    }

    #[test]
    fn bbox_min_l2_zero_when_inside() {
        let q: [f32; 16] = [0.5; 16];
        let mins: [i16; 16] = [0; 16];  // 0.0 after dequant
        let maxs: [i16; 16] = [10000; 16]; // 1.0 after dequant
        let got = BBOX_MIN_L2.get().unwrap()(&q, &mins, &maxs);
        assert!(got.abs() < 1e-6, "should be 0 when query is inside bbox");
    }

    #[test]
    fn bbox_min_l2_nonzero_when_outside() {
        let mut q = [0.5f32; 16];
        q[0] = 2.0; // outside [0, 1]
        let mins: [i16; 16] = [0; 16];
        let maxs: [i16; 16] = [10000; 16];
        let got = BBOX_MIN_L2.get().unwrap()(&q, &mins, &maxs);
        assert!(got > 0.0, "should be nonzero when query is outside bbox");
        // dist = (2.0 - 1.0)^2 = 1.0 for dim 0
        assert!((got - 1.0).abs() < 1e-5);
    }
}
```

- [ ] **Step 2: Run test — verify it fails**

```
cargo test -p ivf-core simd
```

Expected: compile error.

- [ ] **Step 3: Implement simd.rs**

```rust
use std::sync::OnceLock;

pub static ASYM_L2: OnceLock<fn(&[f32; 16], &[i16; 16]) -> f32> = OnceLock::new();
pub static L2_F32: OnceLock<fn(&[f32; 16], &[f32; 16]) -> f32> = OnceLock::new();
pub static BBOX_MIN_L2: OnceLock<fn(&[f32; 16], &[i16; 16], &[i16; 16]) -> f32> = OnceLock::new();

pub fn init() {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            ASYM_L2.set(asym_l2_avx2).ok(); // use avx2 for now; avx512 is additive
            L2_F32.set(l2_f32_avx2).ok();
            BBOX_MIN_L2.set(bbox_min_l2_avx2).ok();
            return;
        }
        if is_x86_feature_detected!("avx2") {
            ASYM_L2.set(asym_l2_avx2).ok();
            L2_F32.set(l2_f32_avx2).ok();
            BBOX_MIN_L2.set(bbox_min_l2_avx2).ok();
            return;
        }
    }
    ASYM_L2.set(asym_l2_scalar).ok();
    L2_F32.set(l2_f32_scalar).ok();
    BBOX_MIN_L2.set(bbox_min_l2_scalar).ok();
}

pub fn asym_l2_scalar(q: &[f32; 16], db: &[i16; 16]) -> f32 {
    const INV: f32 = 1.0 / 10000.0;
    let mut sum = 0.0f32;
    for d in 0..16 {
        let diff = q[d] - db[d] as f32 * INV;
        sum += diff * diff;
    }
    sum
}

pub fn l2_f32_scalar(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..16 { let diff = a[d] - b[d]; sum += diff * diff; }
    sum
}

pub fn bbox_min_l2_scalar(q: &[f32; 16], mins: &[i16; 16], maxs: &[i16; 16]) -> f32 {
    const INV: f32 = 1.0 / 10000.0;
    let mut sum = 0.0f32;
    for d in 0..16 {
        let lo = mins[d] as f32 * INV;
        let hi = maxs[d] as f32 * INV;
        let clamped = q[d].clamp(lo, hi);
        let diff = q[d] - clamped;
        sum += diff * diff;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn asym_l2_avx2(q: &[f32; 16], db: &[i16; 16]) -> f32 {
    use std::arch::x86_64::*;
    const INV: f32 = 1.0 / 10000.0;
    let inv = _mm256_set1_ps(INV);

    // dims 0-7
    let db_lo_i16 = _mm_loadu_si128(db.as_ptr() as *const __m128i);
    let db_lo_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(db_lo_i16));
    let db_lo = _mm256_mul_ps(db_lo_f32, inv);
    let q_lo = _mm256_loadu_ps(q.as_ptr());
    let diff_lo = _mm256_sub_ps(q_lo, db_lo);
    let sq_lo = _mm256_mul_ps(diff_lo, diff_lo);

    // dims 8-15
    let db_hi_i16 = _mm_loadu_si128(db.as_ptr().add(8) as *const __m128i);
    let db_hi_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(db_hi_i16));
    let db_hi = _mm256_mul_ps(db_hi_f32, inv);
    let q_hi = _mm256_loadu_ps(q.as_ptr().add(8));
    let diff_hi = _mm256_sub_ps(q_hi, db_hi);
    let sq_hi = _mm256_mul_ps(diff_hi, diff_hi);

    hsum_avx2(_mm256_add_ps(sq_lo, sq_hi))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_f32_avx2(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    use std::arch::x86_64::*;
    let a_lo = _mm256_loadu_ps(a.as_ptr());
    let b_lo = _mm256_loadu_ps(b.as_ptr());
    let sq_lo = { let d = _mm256_sub_ps(a_lo, b_lo); _mm256_mul_ps(d, d) };
    let a_hi = _mm256_loadu_ps(a.as_ptr().add(8));
    let b_hi = _mm256_loadu_ps(b.as_ptr().add(8));
    let sq_hi = { let d = _mm256_sub_ps(a_hi, b_hi); _mm256_mul_ps(d, d) };
    hsum_avx2(_mm256_add_ps(sq_lo, sq_hi))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bbox_min_l2_avx2(q: &[f32; 16], mins: &[i16; 16], maxs: &[i16; 16]) -> f32 {
    use std::arch::x86_64::*;
    const INV: f32 = 1.0 / 10000.0;
    let inv = _mm256_set1_ps(INV);

    macro_rules! load_i16_to_f32 { ($ptr:expr) => {{
        let raw = _mm_loadu_si128($ptr as *const __m128i);
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(raw)), inv)
    }}; }

    let lo_mins = load_i16_to_f32!(mins.as_ptr());
    let lo_maxs = load_i16_to_f32!(maxs.as_ptr());
    let hi_mins = load_i16_to_f32!(mins.as_ptr().add(8));
    let hi_maxs = load_i16_to_f32!(maxs.as_ptr().add(8));

    let q_lo = _mm256_loadu_ps(q.as_ptr());
    let q_hi = _mm256_loadu_ps(q.as_ptr().add(8));

    let cl_lo = _mm256_min_ps(_mm256_max_ps(q_lo, lo_mins), lo_maxs);
    let cl_hi = _mm256_min_ps(_mm256_max_ps(q_hi, hi_mins), hi_maxs);

    let d_lo = _mm256_sub_ps(q_lo, cl_lo);
    let d_hi = _mm256_sub_ps(q_hi, cl_hi);
    hsum_avx2(_mm256_add_ps(_mm256_mul_ps(d_lo, d_lo), _mm256_mul_ps(d_hi, d_hi)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_extractf128_ps(v, 0);
    let hi = _mm256_extractf128_ps(v, 1);
    let s4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s4);
    let s2 = _mm_add_ps(s4, shuf);
    let s1 = _mm_add_ss(s2, _mm_movehl_ps(shuf, s2));
    _mm_cvtss_f32(s1)
}

// Safe wrappers for dispatch — call these from engine.rs
fn asym_l2_avx2(q: &[f32; 16], db: &[i16; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe { self::asym_l2_avx2(q, db) }
    #[cfg(not(target_arch = "x86_64"))]
    asym_l2_scalar(q, db)
}

fn l2_f32_avx2(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe { self::l2_f32_avx2(a, b) }
    #[cfg(not(target_arch = "x86_64"))]
    l2_f32_scalar(a, b)
}

fn bbox_min_l2_avx2(q: &[f32; 16], mins: &[i16; 16], maxs: &[i16; 16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe { self::bbox_min_l2_avx2(q, mins, maxs) }
    #[cfg(not(target_arch = "x86_64"))]
    bbox_min_l2_scalar(q, mins, maxs)
}
```

- [ ] **Step 4: Run tests — verify pass**

```
cargo test -p ivf-core simd
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```
git add crates/ivf-core/src/simd.rs
git commit -m "feat(ivf-core): add SIMD distance functions with AVX2 dispatch"
```

---

## Task 6: `engine.rs` — NeighborHeap + IVF Search

**Files:**
- Create: `crates/ivf-core/src/engine.rs`

- [ ] **Step 1: Write failing tests**

```rust
// Bottom of crates/ivf-core/src/engine.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::IvfIndex;

    fn init() { crate::simd::init(); }

    #[test]
    fn neighbor_heap_tracks_top5() {
        init();
        let mut h = NeighborHeap::<5>::new();
        assert!(!h.is_full());
        assert_eq!(h.max_dist(), f32::MAX);

        h.push(0.9, 1);
        h.push(0.1, 0);
        h.push(0.5, 1);
        h.push(0.3, 0);
        h.push(0.7, 1);
        assert!(h.is_full());
        assert!((h.max_dist() - 0.9).abs() < 1e-6);

        h.push(0.2, 0); // replaces 0.9
        assert!((h.max_dist() - 0.7).abs() < 1e-6);
        assert_eq!(h.fraud_count(), 2); // 0.5→fraud, 0.7→fraud; 0.1,0.2,0.3→legit
    }

    #[test]
    fn search_all_fraud_returns_5() {
        init();
        let n_clusters = 1u32;
        let stride = 16usize;
        let n_vectors = 10u32;
        // All vectors identical to query, all labeled fraud
        let query = [5000i16; 16];
        let vectors: Vec<i16> = query.iter().cloned().cycle().take(n_vectors as usize * stride).collect();

        let index = IvfIndex {
            n_vectors,
            n_dims: 14, stride: 16, quant_scale: 10000.0,
            n_clusters, nprobe: 1,
            centroids: vec![0.5f32; stride],
            bboxes: vec![0i16; stride * 2],
            offsets: vec![0],
            sizes: vec![n_vectors],
            labels: vec![1u8; n_vectors as usize], // all fraud
            vectors,
        };

        let q_float: [f32; 16] = [0.5; 16]; // matches all DB vectors exactly
        let fraud_count = search(&q_float, &index, 1);
        assert_eq!(fraud_count, 5);
    }

    #[test]
    fn search_all_legit_returns_0() {
        init();
        let n_clusters = 1u32;
        let stride = 16usize;
        let n_vectors = 10u32;
        let query = [5000i16; 16];
        let vectors: Vec<i16> = query.iter().cloned().cycle().take(n_vectors as usize * stride).collect();

        let index = IvfIndex {
            n_vectors,
            n_dims: 14, stride: 16, quant_scale: 10000.0,
            n_clusters, nprobe: 1,
            centroids: vec![0.5f32; stride],
            bboxes: vec![0i16; stride * 2],
            offsets: vec![0],
            sizes: vec![n_vectors],
            labels: vec![0u8; n_vectors as usize], // all legit
            vectors,
        };

        let q_float: [f32; 16] = [0.5; 16];
        let fraud_count = search(&q_float, &index, 1);
        assert_eq!(fraud_count, 0);
    }
}
```

- [ ] **Step 2: Run test — verify it fails**

```
cargo test -p ivf-core engine
```

Expected: compile error.

- [ ] **Step 3: Implement engine.rs**

```rust
use crate::format::IvfIndex;
use crate::simd::{ASYM_L2, BBOX_MIN_L2, L2_F32};

pub const K: usize = 5;
pub const FRAUD_THRESHOLD: f32 = 0.6;

pub struct NeighborHeap<const N: usize> {
    distances: [f32; N],
    labels: [u8; N],
    count: usize,
    max_idx: usize,
}

impl<const N: usize> NeighborHeap<N> {
    pub fn new() -> Self {
        Self { distances: [f32::MAX; N], labels: [0; N], count: 0, max_idx: 0 }
    }

    pub fn is_full(&self) -> bool { self.count == N }

    pub fn max_dist(&self) -> f32 {
        if self.count < N { f32::MAX } else { self.distances[self.max_idx] }
    }

    pub fn push(&mut self, dist: f32, label: u8) {
        if self.count < N {
            self.distances[self.count] = dist;
            self.labels[self.count] = label;
            self.count += 1;
            if self.count == N { self.find_max(); }
        } else if dist < self.distances[self.max_idx] {
            self.distances[self.max_idx] = dist;
            self.labels[self.max_idx] = label;
            self.find_max();
        }
    }

    fn find_max(&mut self) {
        self.max_idx = 0;
        for i in 1..N {
            if self.distances[i] > self.distances[self.max_idx] { self.max_idx = i; }
        }
    }

    pub fn fraud_count(&self) -> usize {
        (0..self.count).filter(|&i| self.labels[i] == 1).count()
    }
}

pub fn search(query: &[f32; 16], index: &IvfIndex, nprobe: usize) -> usize {
    let asym_l2 = ASYM_L2.get().expect("simd::init() not called");
    let l2_f32 = L2_F32.get().expect("simd::init() not called");
    let bbox_min_l2 = BBOX_MIN_L2.get().expect("simd::init() not called");

    let k = index.n_clusters as usize;
    let s = index.stride as usize;

    // 1. Centroid distances (stack-allocated if nprobe ≤ 2048)
    let mut cdists: Vec<(f32, usize)> = (0..k).map(|c| {
        let centroid: &[f32; 16] = index.centroid(c).try_into().unwrap();
        (l2_f32(query, centroid), c)
    }).collect();

    // 2. Partial sort: top nprobe
    let nprobe = nprobe.min(k);
    cdists.select_nth_unstable_by(nprobe.saturating_sub(1), |a, b| a.0.total_cmp(&b.0));
    let candidates = &cdists[..nprobe];

    // 3. Bbox pruning + cluster scan
    let mut heap = NeighborHeap::<K>::new();

    for &(_, cid) in candidates {
        let (mins_sl, maxs_sl) = index.bbox(cid);
        let mins: &[i16; 16] = mins_sl.try_into().unwrap();
        let maxs: &[i16; 16] = maxs_sl.try_into().unwrap();

        // Lower-bound distance to cluster: if ≥ current worst, no vector here can enter top-5
        let lb = bbox_min_l2(query, mins, maxs);
        if heap.is_full() && lb >= heap.max_dist() { continue; }

        let (vecs, labs) = index.cluster(cid);
        for (chunk, &label) in vecs.chunks_exact(s).zip(labs) {
            let db: &[i16; 16] = chunk.try_into().unwrap();
            let dist = asym_l2(query, db);
            heap.push(dist, label);
        }
    }

    heap.fraud_count()
}

pub fn fraud_score(fraud_count: usize) -> f32 {
    fraud_count as f32 / K as f32
}

pub fn is_approved(fraud_count: usize) -> bool {
    fraud_score(fraud_count) < FRAUD_THRESHOLD
}
```

- [ ] **Step 4: Run tests — verify pass**

```
cargo test -p ivf-core engine
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```
git add crates/ivf-core/src/engine.rs
git commit -m "feat(ivf-core): add NeighborHeap and IVF search with bbox pruning"
```

---

## Task 7: `response.rs` — Static HTTP Responses

**Files:**
- Create: `crates/ivf-core/src/response.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_cover_all_fraud_counts() {
        assert_eq!(RESPONSES.len(), 6);
    }

    #[test]
    fn response_bodies_valid_json_with_correct_fields() {
        // Extract body after \r\n\r\n separator
        let body_of = |resp: &[u8]| -> &[u8] {
            let sep = b"\r\n\r\n";
            let pos = resp.windows(4).position(|w| w == sep).unwrap();
            &resp[pos + 4..]
        };

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
```

- [ ] **Step 2: Run test — verify it fails**

```
cargo test -p ivf-core response
```

Expected: compile error.

- [ ] **Step 3: Implement response.rs**

```rust
// 6 complete HTTP/1.1 responses — indexed by fraud_count (0..=5)
// true responses: 35-byte body; false responses: 36-byte body
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
```

- [ ] **Step 4: Run tests — verify pass**

```
cargo test -p ivf-core response
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```
git add crates/ivf-core/src/response.rs
git commit -m "feat(ivf-core): add preallocated static HTTP response bytes"
```

---

## Task 8: `kmeans.rs` — K-Means++ Builder

**Files:**
- Create: `crates/ivf-core/src/kmeans.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_separates_two_clusters() {
        crate::simd::init();
        // 20 vectors near [0,0,...] and 20 near [1,1,...]
        let mut vectors: Vec<[f32; 16]> = Vec::new();
        for _ in 0..20 { vectors.push([0.05; 16]); }
        for _ in 0..20 { vectors.push([0.95; 16]); }

        let config = KMeansConfig { n_clusters: 2, max_iters: 20, change_tol: 1e-4 };
        let (centroids, assignments) = fit(&vectors, &config, 42);

        // All first 20 should be in one cluster, all last 20 in the other
        let cluster_of_first = assignments[0];
        let cluster_of_last = assignments[20];
        assert_ne!(cluster_of_first, cluster_of_last);
        for i in 0..20 { assert_eq!(assignments[i], cluster_of_first); }
        for i in 20..40 { assert_eq!(assignments[i], cluster_of_last); }
    }

    #[test]
    fn fit_returns_correct_shapes() {
        crate::simd::init();
        let vectors: Vec<[f32; 16]> = (0..50).map(|i| [i as f32 / 50.0; 16]).collect();
        let config = KMeansConfig { n_clusters: 5, max_iters: 10, change_tol: 1e-4 };
        let (centroids, assignments) = fit(&vectors, &config, 0);
        assert_eq!(centroids.len(), 5 * 16);
        assert_eq!(assignments.len(), 50);
        assert!(assignments.iter().all(|&c| c < 5));
    }
}
```

- [ ] **Step 2: Run test — verify it fails**

```
cargo test -p ivf-core kmeans
```

Expected: compile error.

- [ ] **Step 3: Implement kmeans.rs**

```rust
use rayon::prelude::*;
use crate::simd::L2_F32;

pub struct KMeansConfig {
    pub n_clusters: usize,
    pub max_iters: usize,
    pub change_tol: f64,
}

/// Returns (centroids: Vec<f32> of shape n_clusters×16, assignments: Vec<usize>)
/// seed: random seed for reproducible builds
pub fn fit(vectors: &[[f32; 16]], config: &KMeansConfig, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let n = vectors.len();
    let k = config.n_clusters;
    let mut rng = seed.wrapping_add(1);

    // Subsample for k-means++ seeding (max 10_000 vectors for speed)
    let sample_size = n.min(10_000);
    let step = n / sample_size;
    let subsample: Vec<&[f32; 16]> = (0..sample_size).map(|i| &vectors[i * step]).collect();

    let mut centroids = seed_plus_plus(&subsample, k, &mut rng);

    let mut assignments = vec![0usize; n];

    for iter in 0..config.max_iters {
        let new_assignments: Vec<usize> = vectors.par_iter().map(|v| {
            let l2 = L2_F32.get().expect("simd::init() not called");
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for c in 0..k {
                let centroid: &[f32; 16] = centroids[c * 16..(c + 1) * 16].try_into().unwrap();
                let d = l2(v, centroid);
                if d < best_dist { best_dist = d; best = c; }
            }
            best
        }).collect();

        let changes = new_assignments.iter().zip(&assignments).filter(|(a, b)| a != b).count();
        assignments = new_assignments;

        // Recompute centroids
        let mut sums = vec![0.0f64; k * 16];
        let mut counts = vec![0usize; k];
        for (v, &c) in vectors.iter().zip(&assignments) {
            for d in 0..16 { sums[c * 16 + d] += v[d] as f64; }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..16 {
                    centroids[c * 16 + d] = (sums[c * 16 + d] / counts[c] as f64) as f32;
                }
            }
        }

        let change_frac = changes as f64 / n as f64;
        eprintln!("kmeans iter {}/{}: {:.2}% changed", iter + 1, config.max_iters, change_frac * 100.0);
        if change_frac < config.change_tol { break; }
    }

    (centroids, assignments)
}

fn seed_plus_plus(subsample: &[&[f32; 16]], k: usize, rng: &mut u64) -> Vec<f32> {
    let n = subsample.len();
    let mut centroids = Vec::with_capacity(k * 16);

    let first = lcg_usize(rng, n);
    centroids.extend_from_slice(subsample[first]);

    let mut min_dists = vec![f32::MAX; n];

    for _ in 1..k {
        let n_existing = centroids.len() / 16;
        let new_c: &[f32; 16] = centroids[(n_existing - 1) * 16..n_existing * 16].try_into().unwrap();

        // Update min_dists with distance to the most recently added centroid only
        for (i, v) in subsample.iter().enumerate() {
            let d = scalar_l2(v, new_c);
            if d < min_dists[i] { min_dists[i] = d; }
        }

        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        let target = lcg_f64(rng) * total;
        let mut cumsum = 0.0f64;
        let mut chosen = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            cumsum += d as f64;
            if cumsum >= target { chosen = i; break; }
        }
        centroids.extend_from_slice(subsample[chosen]);
    }

    centroids
}

fn scalar_l2(a: &[f32; 16], b: &[f32]) -> f32 {
    (0..16).map(|d| { let diff = a[d] - b[d]; diff * diff }).sum()
}

fn lcg_usize(state: &mut u64, n: usize) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as usize % n
}

fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
```

- [ ] **Step 4: Run tests — verify pass**

```
cargo test -p ivf-core kmeans
```

Expected: 2 tests pass.

- [ ] **Step 5: Run full ivf-core test suite**

```
cargo test -p ivf-core
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```
git add crates/ivf-core/src/kmeans.rs
git commit -m "feat(ivf-core): add k-means++ builder with rayon-parallel assignment"
```

---

## Task 9: `refvec-builder` — Full Pipeline

**Files:**
- Modify: `crates/refvec-builder/src/main.rs`

- [ ] **Step 1: Implement refvec-builder/src/main.rs**

```rust
use clap::Parser;
use flate2::read::GzDecoder;
use ivf_core::{
    format::{quantize, IvfIndex},
    kmeans::{fit, KMeansConfig},
    simd,
};
use serde::Deserialize;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "resources/references.json.gz")]
    refs: PathBuf,
    #[arg(long, default_value = "resources/normalization.json")]
    norm: PathBuf,
    #[arg(long, default_value = "resources/mcc_risk.json")]
    mcc: PathBuf,
    #[arg(long, default_value = "resources/references.ivfvec")]
    out: PathBuf,
    #[arg(long, default_value_t = 2048)]
    clusters: usize,
    #[arg(long, default_value_t = 16)]
    nprobe: u16,
    #[arg(long, default_value_t = 100)]
    iters: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: Vec<f32>, // 14 elements
    label: String,    // "legit" | "fraud"
}

fn main() {
    let args = Args::parse();
    simd::init();

    // Step 1: Read pre-vectorized references from .json.gz
    eprintln!("reading {}...", args.refs.display());
    let file = std::fs::File::open(&args.refs).expect("open refs");
    let gz = GzDecoder::new(BufReader::new(file));
    let records: Vec<ReferenceRecord> = serde_json::from_reader(gz).expect("parse refs JSON");
    eprintln!("loaded {} records", records.len());

    // Step 2: Build float[16] vectors (pad 14 → 16) + labels
    let mut vectors: Vec<[f32; 16]> = Vec::with_capacity(records.len());
    let mut labels: Vec<u8> = Vec::with_capacity(records.len());
    for r in &records {
        let mut v = [0.0f32; 16];
        for (i, &x) in r.vector.iter().enumerate().take(14) { v[i] = x; }
        vectors.push(v);
        labels.push(if r.label == "fraud" { 1 } else { 0 });
    }

    // Step 3: K-means++ clustering
    let config = KMeansConfig {
        n_clusters: args.clusters,
        max_iters: args.iters,
        change_tol: 1e-4,
    };
    eprintln!("running k-means++ (k={}, max_iters={})...", args.clusters, args.iters);
    let (centroids_flat, assignments) = fit(&vectors, &config, args.seed);
    eprintln!("k-means done");

    // Step 4: Sort vectors + labels by cluster assignment
    let mut indexed: Vec<(usize, [f32; 16], u8)> = assignments.iter()
        .zip(vectors.iter())
        .zip(labels.iter())
        .map(|((&c, &v), &l)| (c, v, l))
        .collect();
    indexed.sort_unstable_by_key(|&(c, _, _)| c);

    // Step 5: Compute offsets + sizes
    let k = args.clusters;
    let mut offsets = vec![0u32; k];
    let mut sizes = vec![0u32; k];
    for &(c, _, _) in &indexed { sizes[c] += 1; }
    for i in 1..k { offsets[i] = offsets[i - 1] + sizes[i - 1]; }

    // Step 6: Quantize vectors + compute bboxes
    let stride = 16usize;
    let mut qvectors: Vec<i16> = Vec::with_capacity(indexed.len() * stride);
    let mut qlabels: Vec<u8>   = Vec::with_capacity(indexed.len());
    // bbox layout: [cluster0_mins[16], cluster0_maxs[16], cluster1_mins[16], ...]
    let mut bboxes: Vec<i16>   = vec![i16::MAX; k * stride]; // mins
    bboxes.extend(vec![i16::MIN; k * stride]);                // maxs

    for (c, v, l) in &indexed {
        let qv: [i16; 16] = std::array::from_fn(|d| quantize(v[d]));
        for d in 0..stride {
            if qv[d] < bboxes[c * stride + d]           { bboxes[c * stride + d] = qv[d]; }
            if qv[d] > bboxes[k * stride + c * stride + d] { bboxes[k * stride + c * stride + d] = qv[d]; }
        }
        qvectors.extend_from_slice(&qv);
        qlabels.push(*l);
    }

    // Reorder bboxes from [all_mins, all_maxs] to per-cluster planar [mins, maxs, mins, maxs, ...]
    // Current layout: mins[0..k*s], maxs[k*s..2*k*s]
    // Target layout:  [cluster0_mins[s], cluster0_maxs[s], cluster1_mins[s], ...]
    let mut bboxes_planar: Vec<i16> = Vec::with_capacity(k * stride * 2);
    for c in 0..k {
        bboxes_planar.extend_from_slice(&bboxes[c * stride..(c + 1) * stride]);         // mins
        bboxes_planar.extend_from_slice(&bboxes[k * stride + c * stride..k * stride + (c + 1) * stride]); // maxs
    }

    // Step 7: Write .ivfvec
    let index = IvfIndex {
        n_vectors: indexed.len() as u32,
        n_dims: 14,
        stride: 16,
        quant_scale: 10000.0,
        n_clusters: k as u32,
        nprobe: args.nprobe,
        centroids: centroids_flat,
        bboxes: bboxes_planar,
        offsets,
        sizes,
        labels: qlabels,
        vectors: qvectors,
    };

    eprintln!("writing {}...", args.out.display());
    index.write(&args.out).expect("write ivfvec");
    eprintln!("done — {} vectors, {} clusters", index.n_vectors, index.n_clusters);
}
```

- [ ] **Step 2: Build refvec-builder**

```
cargo build -p refvec-builder --release
```

Expected: compiles with no errors.

- [ ] **Step 3: Run builder on real data (smoke test)**

```
cargo run -p refvec-builder --release -- \
  --refs resources/references.json.gz \
  --out resources/references.ivfvec \
  --clusters 2048 --nprobe 16 --iters 100
```

Expected: ~30–90s. Outputs file size ~99 MB.

- [ ] **Step 4: Commit**

```
git add crates/refvec-builder/src/main.rs
git commit -m "feat(refvec-builder): full k-means++ IVF index build pipeline"
```

---

## Task 10: `api` — HTTP Server

**Files:**
- Create: `crates/api/src/http.rs`
- Modify: `crates/api/src/main.rs`

- [ ] **Step 1: Write integration test (smoke)**

Create `crates/api/tests/smoke.rs`:

```rust
// Integration test — requires a real .ivfvec file at resources/
// Run with: cargo test -p api --test smoke

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
#[ignore] // run manually: cargo test -p api --test smoke -- --ignored
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
```

- [ ] **Step 2: Create crates/api/src/http.rs**

```rust
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

        // HTTP/1.1: keep-alive by default; close if client requests it
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
```

- [ ] **Step 3: Implement crates/api/src/main.rs**

```rust
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
```

- [ ] **Step 4: Build api**

```
cargo build -p api --release
```

Expected: compiles with no errors.

- [ ] **Step 5: Smoke test manually**

```
RESOURCES_PATH=resources cargo run -p api --release
```

In another terminal:
```
curl -s http://localhost:9999/ready
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'content-type: application/json' \
  -d '{"id":"t1","transaction":{"amount":100,"installments":1,"requested_at":"2026-03-11T12:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":2,"known_merchants":["M1"]},"merchant":{"id":"M1","mcc":"5411","avg_amount":90},"terminal":{"is_online":false,"card_present":true,"km_from_home":5},"last_transaction":null}'
```

Expected: `200 OK` for `/ready`; JSON with `approved` and `fraud_score` for `/fraud-score`.

- [ ] **Step 6: Commit**

```
git add crates/api/src/ crates/api/tests/
git commit -m "feat(api): tokio HTTP server with httparse, zero-alloc fraud scoring"
```

---

## Task 11: Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Update Dockerfile for Rust**

```dockerfile
FROM rust:1.87-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN cargo build -p api --release

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/target/release/api .
EXPOSE 9999
CMD ["./api"]
```

- [ ] **Step 2: Build Docker image (verify)**

```
docker build -t rinha-api .
```

Expected: builds successfully.

- [ ] **Step 3: Smoke test Docker image**

```
docker run --rm -v $(pwd)/resources:/app/resources -p 9999:9999 rinha-api
```

In another terminal:
```
curl -s http://localhost:9999/ready
```

Expected: `HTTP/1.1 200 OK`.

- [ ] **Step 4: Commit**

```
git add Dockerfile
git commit -m "feat: update Dockerfile for Rust api binary"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Workspace with 3 crates (Task 1)
- ✅ IVF binary format IVFVEC01 (Task 4)
- ✅ K-means++ builder (Task 8/9)
- ✅ AVX2/512 SIMD dispatch (Task 5)
- ✅ Bbox cluster pruning (Task 6)
- ✅ NeighborHeap<5> with score-aware reranking via exact asymmetric L2 (Task 6)
- ✅ 6 preallocated full HTTP responses (Task 7)
- ✅ tokio + httparse, no hyper (Task 10)
- ✅ OTel feature flag in api/Cargo.toml (Task 1)
- ✅ NPROBE env var override (Task 10)
- ✅ Dockerfile (Task 11)
- ⚠️ OTel instrumentation body (Task 1 adds the feature flag + deps; actual span instrumentation in http.rs is left for a follow-up since it's dev-only and doesn't affect correctness)

**Type consistency:**
- `IvfIndex::cluster(cid)` → `(&[i16], &[u8])` — used identically in engine.rs and builder
- `IvfIndex::bbox(cid)` → `(&[i16], &[i16])` (mins, maxs) — matches bbox_min_l2 signature
- `search(&[f32;16], &IvfIndex, usize) -> usize` — consistent across engine.rs and http.rs
- `RESPONSES[fraud_count]` indexed by `usize` in range 0..=5 — matches engine's `fraud_count()` return
