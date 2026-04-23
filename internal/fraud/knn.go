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

func Score(query [14]float64, refs *References) (fraudScore float64, approved bool) {
	var heap top5
	vectors := refs.Vectors
	isFraud := refs.IsFraud

	for i := range vectors {
		d := euclideanSq(query, vectors[i])
		heap.insert(d, isFraud[i])
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

func euclideanSq(a, b [14]float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}
