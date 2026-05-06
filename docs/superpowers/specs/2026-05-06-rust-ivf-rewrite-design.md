# Rust IVF Fraud Detection Rewrite — Design Spec
Date: 2026-05-06

## Overview

Rewrite the Go/Echo fraud detection API in Rust. Replace flat KNN search over 3M vectors with IVF (Inverted File Index) approximate nearest-neighbor search using Asymmetric distance computation, bbox cluster pruning, and score-aware reranking. Two executables: index builder and HTTP API server.

**Runtime:** tokio async + httparse for HTTP parsing. No framework.
**Indexing:** IVF with 2048 clusters, k-means++ initialization, nprobe=16 default.
**Distance:** Asymmetric L2 — query stays f32, DB vectors stay i16, dequantized on-the-fly via SIMD.
**Quant scale:** 10000 (values map to [0, 10000] as i16).
**Responses:** 6 pre-built static `&'static [u8]` full HTTP/1.1 responses, zero alloc on hot path.
**OTel:** feature-flagged, enabled in dev only.

---

## 1. Workspace & Crate Layout

```
rinha-2026/
├── Cargo.toml                    (workspace)
├── crates/
│   ├── ivf-core/                 (lib)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── format.rs         (IvfIndex binary read/write, magic "IVFVEC01")
│   │   │   ├── engine.rs         (IVF search: centroid dist, bbox pruning, ALT scan)
│   │   │   ├── kmeans.rs         (k-means++ builder, rayon-parallel)
│   │   │   ├── vector.rs         (FraudScoreRequest → [f32; 16])
│   │   │   ├── norm.rs           (NormalizationConfig, MerchantRiskConfig)
│   │   │   └── response.rs       (6 preallocated static HTTP responses)
│   │   └── Cargo.toml
│   ├── refvec-builder/           (bin)
│   │   ├── src/main.rs
│   │   └── Cargo.toml            (ivf-core, serde_json, flate2, rayon, clap)
│   └── api/                      (bin)
│       ├── src/main.rs
│       └── Cargo.toml            (ivf-core, tokio, httparse, serde_json; feature: otel)
└── resources/
    ├── normalization.json
    ├── mcc_risk.json
    ├── references.json.gz        (builder input)
    └── references.ivfvec         (builder output, api input)
```

Dependency direction: `api` → `ivf-core` ← `refvec-builder`. No cross-dependency between bins.

Go files remain until Rust port is verified.

---

## 2. IVF Binary Format (`references.ivfvec`)

Little-endian throughout.

```
[magic]       8 bytes   "IVFVEC01"
[version]     1 byte    = 1
[n_vectors]   4 bytes   u32   ~3_000_000
[n_dims]      2 bytes   u16   = 14
[stride]      2 bytes   u16   = 16
[quant_scale] 4 bytes   f32   = 10000.0
[n_clusters]  4 bytes   u32   = 2048
[nprobe]      2 bytes   u16   = 16 (default; overridable via env NPROBE)
──────────────────────────────────────────────────────────────
[centroids]   n_clusters × stride × f32        ~131 KB   float, not quantized
[bboxes]      n_clusters × stride × 2 × i16   ~131 KB   [min, max] per dim per cluster
[offsets]     n_clusters × u32                  ~8 KB   start index in flat array
[sizes]       n_clusters × u32                  ~8 KB   vector count per cluster
[labels]      n_vectors × u8                    ~3 MB   0=legit, 1=fraud
[vectors]     n_vectors × stride × i16         ~96 MB   cluster-ordered, contiguous per cluster
```

Total: ~99 MB. Vectors stored contiguous per cluster — cluster access is a zero-copy subslice.
Bbox stores `[min_i16, max_i16]` per dim across all vectors in the cluster.

---

## 3. `refvec-builder` Executable

### CLI
```
refvec-builder \
  --refs   resources/references.json.gz \
  --norm   resources/normalization.json \
  --mcc    resources/mcc_risk.json \
  --out    resources/references.ivfvec \
  [--clusters 2048] \
  [--nprobe 16] \
  [--iters 100]
```

### Pipeline

**Step 1 — Read & vectorize**
- Decompress `.json.gz` with `flate2`
- Parse JSON array of reference records with `serde_json`
- Apply `ivf_core::vector::vectorize()` to each record → `[f32; 16]`
- Retain label per vector (`legit` → 0u8, `fraud` → 1u8)

**Step 2 — K-means++ (rayon-parallel)**
- Init: pick first centroid randomly; each subsequent centroid selected with probability ∝ min-squared-distance to existing centroids (k-means++ seeding)
- Iterate up to `--iters`:
  - Assign each vector to nearest centroid (parallel over vectors with rayon)
  - Recompute centroids as mean of assigned vectors (parallel over clusters)
  - Early exit if assignment change < 0.01%
- Output: `centroids[2048][16]` as f32

**Step 3 — Assign & sort**
- Assign each vector to nearest centroid → `cluster_id[]`
- Stable-sort vectors + labels by `cluster_id` (in-place)
- Compute `offsets[]` and `sizes[]` from sorted assignment

**Step 4 — Compute bboxes**
- Per cluster: min and max per dim across all assigned vectors (in i16 after quantization)

**Step 5 — Quantize**
- f32 → i16: `clamp(v, 0.0, 1.0) * 10000.0` as i16

**Step 6 — Write `.ivfvec`**
- Serialize header + sections in format order (Section 2), little-endian

**Build time estimate:** ~30–90s on modern CPU (one-shot; output ships with Docker image).

---

## 4. `api` Executable — Server & Search

### Startup
1. Parse env: `RESOURCES_PATH`, `NPROBE` (default 16), `PORT` (default 9999)
2. Load `references.ivfvec` → `Arc<IvfIndex>` via mmap (lazy page faults for vector data)
3. Load `normalization.json` + `mcc_risk.json` → `OnceLock<T>` (hot path read-only)
4. Detect CPU SIMD capabilities → store function pointers in `OnceLock`
5. Init OTel if `otel` feature enabled
6. `TcpListener::bind` → accept loop, spawn task per connection

### Request path (`POST /fraud-score`)
```
1. Read TCP bytes into stack buffer
2. httparse::parse_request → extract body slice (zero-copy)
3. serde_json::from_slice(&body) → FraudScoreRequest
4. ivf_core::vector::vectorize(req, norm, merchs) → [f32; 16]  (stack-only)
5. ivf_core::engine::search(&query, &index)                    (stack-only, see below)
6. stream.write_all(RESPONSES[fraud_count]).await              (zero alloc)
```

### IVF Search (`engine.rs`)

```
fn search(query: &[f32; 16], index: &IvfIndex, nprobe: usize) -> usize /* fraud_count */ {

  // 1. Centroid distances — stack array
  let mut cdists: [(f32, u16); 2048] = uninit;
  for (i, centroid) in index.centroids.chunks(16).enumerate() {
      cdists[i] = (l2_f32(query, centroid), i as u16);
  }

  // 2. Partial sort: top nprobe (no full sort needed)
  cdists.select_nth_unstable_by(nprobe, |a, b| a.0.total_cmp(&b.0));
  let candidates = &cdists[..nprobe];

  // 3. Bbox pruning + ALT cluster scan
  let mut heap = NeighborHeap::<5>::new();  // stack, fixed-size max-heap

  for &(_, cid) in candidates {
      // bbox_min_l2 = lower bound on L2 dist from query to any vector in cluster
      // if lower bound >= current 5th-best, no vector here can enter top-5 → skip
      let bbox_dist = bbox_min_l2(query, index.bbox(cid));
      if heap.is_full() && bbox_dist >= heap.max_dist() { continue; }

      let (vecs, labels) = index.cluster(cid);  // zero-copy subslices
      for (vec, &label) in vecs.chunks(16).zip(labels) {
          let dist = ASYM_L2_FN.get()(query, vec);
          heap.push(dist, label);
      }
  }

  heap.fraud_count()
}
```

**`NeighborHeap<5>`:** fixed-size array-based max-heap on stack (5-element array + count). `push` replaces root if new dist < max. `max_dist` returns root. Same role as Go's `nearestNeighbors` struct. No stdlib heap — pure array for cache locality.

**Fraud score:** `fraud_count as f32 / 5.0`. Threshold 0.6 → approved if `fraud_count < 3`.
Response selection is keyed off `fraud_count` (integer 0..=5) directly: `RESPONSES[fraud_count]`. With K=5 fixed, only 6 discrete scores are possible — no intermediate values.

---

## 5. SIMD Strategy

### Runtime dispatch (startup)

```rust
static ASYM_L2_FN: OnceLock<fn(&[f32; 16], &[i16; 16]) -> f32> = OnceLock::new();

fn init_simd() {
    let f = if is_x86_feature_detected!("avx512f") { asym_l2_avx512 }
            else if is_x86_feature_detected!("avx2") { asym_l2_avx2 }
            else { asym_l2_scalar };
    ASYM_L2_FN.set(f).unwrap();
}
```

Same pattern for `l2_f32` (centroid distances, both f32).

### `asym_l2` — AVX2 path (baseline)
- 2 passes of 8 f32 each (16 dims total)
- Per pass: load 8×i16 → `_mm256_cvtepi16_epi32` → `_mm256_cvtepi32_ps` → multiply by `1.0/10000.0` → subtract query → square → accumulate
- Horizontal reduce at end

### `asym_l2` — AVX-512 path (fast)
- 1 pass of 16 f32
- 16×i16 = 256 bits → load with `_mm256_loadu_si256` (__m256i), widen to 16×i32 (512-bit) via `_mm512_cvtepi16_epi32(__m256i)`, then `_mm512_cvtepi32_ps` → scale → subtract → `_mm512_reduce_add_ps`
- Note: 256-bit load is correct here — 16 i16 values fit exactly in 256 bits; widening to 512-bit happens during conversion

### Centroid scan (2048 × 16 f32)
- `l2_f32` with same AVX2/512 dispatch
- Loop body unrolled 4× to hide FP latency

### `bbox_min_l2`
- Clamp query per dim to `[bbox_min[d], bbox_max[d]]` via SIMD min/max, then `l2_f32`

---

## 6. Preallocated Responses

All 6 fraud outcomes + `/ready` + 404 pre-built at compile time as `&'static [u8]`:

```rust
// ivf-core::response
pub static RESPONSES: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 34\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 34\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 34\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 35\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 35\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 35\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

pub static READY_RESPONSE: &[u8] = b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n";
pub static NOT_FOUND: &[u8]      = b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n";
```

Response path: `stream.write_all(RESPONSES[fraud_count]).await` — zero heap allocation.

---

## 7. Memory Layout

| Region | Size | Strategy |
|---|---|---|
| IvfIndex vectors | ~96 MB | mmap read-only; OS pages fault lazily on first access |
| IvfIndex labels | ~3 MB | mmap, same file |
| Centroids + bboxes | ~265 KB | eagerly copied into `Vec<f32>` at startup (hot path) |
| Norm + MCC config | ~1 KB | `OnceLock<T>`, loaded once at startup |
| Per-request stack | ~0 heap | `[f32;16]` query, `[(f32,u16);2048]` cdists, `NeighborHeap<5>` |

`Arc<IvfIndex>` shared across tokio tasks, cloned cheaply per connection.

---

## 8. OTel Feature Flag

```toml
# api/Cargo.toml
[features]
default = []
otel = ["opentelemetry", "opentelemetry-otlp", "tracing-opentelemetry", "tracing"]
```

Dev build: `cargo build --features otel`
Prod/competition build: `cargo build --release` (no otel, smaller binary, no trace overhead)

Instrumentation gated with `#[cfg(feature = "otel")]` in `api/src/main.rs` only. `ivf-core` has no OTel dependency.

---

## 9. Error Handling & Operational Notes

- `.ivfvec` load failure (bad magic, truncated, version mismatch): `panic!` — fail fast at startup, not in flight
- Request parse failure (bad JSON, missing fields): return `400 Bad Request` (pre-built static response)
- Max request body: 64 KB (httparse header + body; competition payloads are ~2 KB max)
- Connection limit: none explicit; tokio task per connection; OS backlog handles burst
- `NPROBE` env var overrides file default at runtime without rebuilding index

## 10. Key Invariants

- `n_dims = 14`, `stride = 16` — always 2 padding dims; SIMD always processes 16
- `quant_scale = 10000` — all features normalized [0,1] before quantize
- `fraud_threshold = 0.6` — approved if `fraud_count < 3` (K=5)
- `n_clusters = 2048` — matches sqrt(3M) ≈ 1732, rounded to power of 2
- `nprobe = 16` default — scans ~23K vectors/query before bbox pruning
- Vectors within each cluster stored contiguously — zero-copy cluster slice
