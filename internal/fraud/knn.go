package fraud

import "math"

const k = 5
const fraudThreshold = 0.6

// top5 tracks the 5 largest distances seen so far (max-heap behaviour via manual tracking).
type top5 struct {
	dists   [k]float64
	frauds  [k]bool
	maxIdx  int
	count   int
}

func (t *top5) insert(dist float64, isFraud bool) {
	if t.count < k {
		t.dists[t.count] = dist
		t.frauds[t.count] = isFraud
		t.count++
		if t.count == k {
			t.updateMax()
		}
		return
	}
	if dist < t.dists[t.maxIdx] {
		t.dists[t.maxIdx] = dist
		t.frauds[t.maxIdx] = isFraud
		t.updateMax()
	}
}

func (t *top5) updateMax() {
	t.maxIdx = 0
	for i := 1; i < k; i++ {
		if t.dists[i] > t.dists[t.maxIdx] {
			t.maxIdx = i
		}
	}
}

func Score(query [16]float32, refs *References) (fraudScore float64, approved bool) {
	var heap top5

	for i := 0; i < refs.N; i++ {
		base := i * stride
		d := euclideanSq(query, refs.Flat[base:base+stride])
		heap.insert(d, refs.Labels[i])
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

func euclideanSq(a [16]float32, b []float32) float64 {
	var sum float64
	for i := 0; i < 14; i++ { // only compare first 14 elements (padding ignored)
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}
