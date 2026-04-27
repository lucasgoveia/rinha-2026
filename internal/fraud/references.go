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
