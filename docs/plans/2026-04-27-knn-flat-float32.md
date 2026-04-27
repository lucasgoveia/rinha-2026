# k-NN Flat float32 Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the k-NN search pipeline to use a contiguous flat `[]float32` layout with stride 16, remove `math.Sqrt`, add early-exit pruning, and make the request pipeline explicit: bind → vectorize → search → result.

**Architecture:** References are stored as two parallel slices: `Flat []float32` (features only, stride=16: indices 0–13 are the 14 features, indices 14–15 are zero padding) and `Labels []bool` (one bool per vector: `true` = fraud, `false` = legit), plus `N int` as the explicit vector count. The query vector is `[16]float32` with the same stride. `topK` uses fixed-size `[k]float32` arrays — stays on the stack with zero heap allocation in the hot path. Squared Euclidean distance replaces `math.Sqrt`.

**Tech Stack:** Go 1.25, standard library only.

---

### Task 1: Update `References` and `LoadReferences`

**Files:**
- Modify: `internal/fraud/references.go`

**Rewrite:**

```go
package fraud

import (
	"compress/gzip"
	"encoding/json"
	"os"
)

// stride is the number of float32 slots per reference vector.
// Indices 0–13: features. Indices 14–15: zero padding for 16-float alignment.
const stride = 16

type referenceEntry struct {
	Vector [14]float64 `json:"vector"`
	Label  string      `json:"label"`
}

// References holds the training set as a contiguous flat []float32 of feature vectors
// and a parallel []bool of labels (true = fraud, false = legit).
// Vector i occupies Flat[i*stride : i*stride+14]; slots 14–15 are zero padding.
type References struct {
	Flat   []float32 // len = N * stride
	Labels []bool    // len = N; Labels[i] == true means vector i is fraud
	N      int
}

func LoadReferences(path string) (*References, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var entries []referenceEntry
	if err := json.NewDecoder(gz).Decode(&entries); err != nil {
		return nil, err
	}

	n := len(entries)
	refs := &References{
		Flat:   make([]float32, n*stride),
		Labels: make([]bool, n),
		N:      n,
	}
	for i, e := range entries {
		base := i * stride
		for j := 0; j < 14; j++ {
			refs.Flat[base+j] = float32(e.Vector[j])
		}
		// refs.Flat[base+14] and [base+15] remain 0 (padding)
		refs.Labels[i] = e.Label == "fraud"
	}
	return refs, nil
}
```

**Build check:**

```bash
go build ./...
```

Expected: errors in `knn.go` referencing the old `Vectors` and `IsFraud` fields. Fixed in Task 3.

**Commit:**

```bash
git add internal/fraud/references.go
git commit -m "feat: references as flat []float32 stride-16 with parallel Labels bool slice"
```

---

### Task 2: Update `Vectorize` to return `[16]float32`

**Files:**
- Modify: `internal/fraud/vector.go`

**Rewrite:**

```go
package fraud

import "time"

type NormConstants struct {
	MaxAmount            float64 `json:"max_amount"`
	MaxInstallments      float64 `json:"max_installments"`
	AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float64 `json:"max_minutes"`
	MaxKm                float64 `json:"max_km"`
	MaxTxCount24h        float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

type TransactionReq struct {
	Amount       float64 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type CustomerReq struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type MerchantReq struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type TerminalReq struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

type LastTransactionReq struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

type FraudScoreRequest struct {
	ID              string              `json:"id"`
	Transaction     TransactionReq      `json:"transaction"`
	Customer        CustomerReq         `json:"customer"`
	Merchant        MerchantReq         `json:"merchant"`
	Terminal        TerminalReq         `json:"terminal"`
	LastTransaction *LastTransactionReq `json:"last_transaction"`
}

func clamp(x float64) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return float32(x)
}

func boolF32(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

// Vectorize converts a request into a [16]float32 query vector (stride-16 layout).
// Indices 0–13: features. Indices 14–15: zero padding (not features).
func Vectorize(req *FraudScoreRequest, mccRisk map[string]float64, norm NormConstants) [16]float32 {
	t, _ := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	hour := float64(t.UTC().Hour())
	wdRaw := int(t.UTC().Weekday()+6) % 7
	dow := float64(wdRaw) / 6

	mccRiskVal, ok := mccRisk[req.Merchant.MCC]
	if !ok {
		mccRiskVal = 0.5
	}

	unknownMerchant := float32(1)
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			unknownMerchant = 0
			break
		}
	}

	var minutesSinceLast, kmFromLast float32
	if req.LastTransaction == nil {
		minutesSinceLast = -1
		kmFromLast = -1
	} else {
		lastT, _ := time.Parse(time.RFC3339, req.LastTransaction.Timestamp)
		diff := t.Sub(lastT).Minutes()
		minutesSinceLast = clamp(diff / norm.MaxMinutes)
		kmFromLast = clamp(req.LastTransaction.KmFromCurrent / norm.MaxKm)
	}

	return [16]float32{
		clamp(req.Transaction.Amount / norm.MaxAmount),
		clamp(float64(req.Transaction.Installments) / norm.MaxInstallments),
		clamp((req.Transaction.Amount / req.Customer.AvgAmount) / norm.AmountVsAvgRatio),
		float32(hour / 23),
		float32(dow),
		minutesSinceLast,
		kmFromLast,
		clamp(req.Terminal.KmFromHome / norm.MaxKm),
		clamp(float64(req.Customer.TxCount24h) / norm.MaxTxCount24h),
		boolF32(req.Terminal.IsOnline),
		boolF32(req.Terminal.CardPresent),
		unknownMerchant,
		float32(mccRiskVal),
		clamp(req.Merchant.AvgAmount / norm.MaxMerchantAvgAmount),
		0, // 14: padding
		0, // 15: padding
	}
}
```

**Commit:**

```bash
git add internal/fraud/vector.go
git commit -m "feat: vectorize returns [16]float32 with stride-16 padding"
```

---

### Task 3: Rewrite `knn.go`

**Files:**
- Modify: `internal/fraud/knn.go`

**Rewrite:**

```go
package fraud

const k = 5
const fraudThreshold = 0.6

// topK is a bounded max-heap (by squared distance) tracking the k nearest neighbors.
// Fixed-size arrays keep it on the stack — no heap allocation in the hot loop.
type topK struct {
	dists  [k]float32
	frauds [k]bool
	maxIdx int
	count  int
}

func (t *topK) insert(dist float32, isFraud bool) {
	if t.count < k {
		t.dists[t.count] = dist
		t.frauds[t.count] = isFraud
		t.count++
		if t.count == k {
			t.findMax()
		}
		return
	}
	if dist < t.dists[t.maxIdx] {
		t.dists[t.maxIdx] = dist
		t.frauds[t.maxIdx] = isFraud
		t.findMax()
	}
}

func (t *topK) findMax() {
	t.maxIdx = 0
	for i := 1; i < k; i++ {
		if t.dists[i] > t.dists[t.maxIdx] {
			t.maxIdx = i
		}
	}
}

// Score runs exact brute-force k-NN over the flat reference store.
// Distance is squared Euclidean (no sqrt) over the 14 feature dimensions only.
// Once the heap is full, early-exit pruning aborts the inner loop as soon as
// the partial sum exceeds the current worst accepted distance.
//
// Pipeline: query [16]float32 → brute-force search → (fraudScore, approved)
func Score(query [16]float32, refs *References) (fraudScore float64, approved bool) {
	var heap topK
	flat := refs.Flat
	labels := refs.Labels
	n := refs.N

	for i := 0; i < n; i++ {
		base := i * stride

		var sum float32
		if heap.count < k {
			// heap not full — no threshold available, compute full distance
			for j := 0; j < 14; j++ {
				d := query[j] - flat[base+j]
				sum += d * d
			}
		} else {
			// heap full — prune early if partial sum already exceeds worst neighbor
			threshold := heap.dists[heap.maxIdx]
			skip := false
			for j := 0; j < 14; j++ {
				d := query[j] - flat[base+j]
				sum += d * d
				if sum >= threshold {
					skip = true
					break
				}
			}
			if skip {
				continue
			}
		}

		heap.insert(sum, labels[i])
	}

	fraudCount := 0
	for i := 0; i < heap.count; i++ {
		if heap.frauds[i] {
			fraudCount++
		}
	}

	fraudScore = float64(fraudCount) / k
	approved = fraudScore < fraudThreshold
	return
}
```

**Build and smoke test:**

```bash
go build ./...

go run ./cmd/api &
sleep 1
curl -s -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d "$(python3 -c 'import sys,json; d=json.load(open("resources/example-payloads.json")); print(json.dumps(d[0]))')"
kill %1
```

Expected: valid `{"approved":...,"fraud_score":...}` response, no crash.

**Commit:**

```bash
git add internal/fraud/knn.go
git commit -m "feat: knn flat float32 stride-16, squared distance, early-exit pruning, zero alloc"
```

---

## Final pipeline

```
POST /fraud-score
  │
  ├─ Bind(echo.Context)   → FraudScoreRequest
  ├─ Vectorize()          → [16]float32          (features 0–13, zeros at 14–15)
  ├─ Score()              → (float64, bool)       (brute-force k-NN, flat []float32)
  └─ JSON response        → {"approved":bool, "fraud_score":float64}
```

## Memory layout

```
refs.Flat (single allocation, stride=16):
┌─────────────────────────────┬──────┬──────┐
│  f0  f1  f2  ...  f13       │  0   │  0   │  ← vector 0
├─────────────────────────────┼──────┼──────┤
│  f0  f1  f2  ...  f13       │  0   │  0   │  ← vector 1
└──────────── N vectors ──────┴──────┴──────┘

refs.Labels (parallel bool slice, len=N):
[ false, true, false, ... ]   ← true = fraud
```

## Files changed

| File | Change |
|------|--------|
| `internal/fraud/references.go` | `References.Flat []float32` + `Labels []bool` + `N int`; stride-16 packing |
| `internal/fraud/vector.go` | `Vectorize` returns `[16]float32`; `clamp` returns `float32` |
| `internal/fraud/knn.go` | `topK` in `float32`; `Score([16]float32)`; no sqrt; early-exit pruning |
