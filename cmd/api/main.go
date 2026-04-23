package main

import (
	"encoding/json"
	"log"
	"os"

	"github.com/labstack/echo/v4"
	"github.com/lucasgoveia/rinha2026/internal/fraud"
	"github.com/lucasgoveia/rinha2026/internal/handler"
)

func main() {
	resourcesPath := os.Getenv("RESOURCES_PATH")
	if resourcesPath == "" {
		resourcesPath = "resources"
	}

	norm, err := loadJSON[fraud.NormConstants](resourcesPath + "/normalization.json")
	if err != nil {
		log.Fatalf("load normalization: %v", err)
	}

	mccRisk, err := loadJSON[map[string]float64](resourcesPath + "/mcc_risk.json")
	if err != nil {
		log.Fatalf("load mcc_risk: %v", err)
	}

	log.Println("loading references...")
	refs, err := fraud.LoadReferences(resourcesPath + "/references.json.gz")
	if err != nil {
		log.Fatalf("load references: %v", err)
	}
	log.Printf("loaded %d reference vectors", len(refs.Vectors))

	h := &handler.Handler{
		Refs:    refs,
		MccRisk: mccRisk,
		Norm:    norm,
	}

	e := echo.New()
	e.HideBanner = true

	e.GET("/ready", h.Ready)
	e.POST("/fraud-score", h.FraudScore)

	log.Fatal(e.Start(":9999"))
}

func loadJSON[T any](path string) (T, error) {
	var v T
	f, err := os.Open(path)
	if err != nil {
		return v, err
	}
	defer f.Close()
	return v, json.NewDecoder(f).Decode(&v)
}
