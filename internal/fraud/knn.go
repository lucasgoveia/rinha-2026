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
