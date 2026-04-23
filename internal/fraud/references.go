package fraud

import (
	"compress/gzip"
	"encoding/json"
	"os"
)

type referenceEntry struct {
	Vector [14]float64 `json:"vector"`
	Label  string      `json:"label"`
}

type References struct {
	Vectors [][14]float64
	IsFraud []bool
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

	refs := &References{
		Vectors: make([][14]float64, len(entries)),
		IsFraud: make([]bool, len(entries)),
	}
	for i, e := range entries {
		refs.Vectors[i] = e.Vector
		refs.IsFraud[i] = e.Label == "fraud"
	}
	return refs, nil
}
