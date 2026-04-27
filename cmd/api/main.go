package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/labstack/echo/v4"
	"go.opentelemetry.io/contrib/instrumentation/github.com/labstack/echo/otelecho"

	"github.com/lucasgoveia/rinha2026/internal/fraud"
	"github.com/lucasgoveia/rinha2026/internal/handler"
	"github.com/lucasgoveia/rinha2026/internal/telemetry"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	otelShutdown := telemetry.Setup(ctx)
	defer func() {
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		otelShutdown(shutCtx)
	}()

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
	log.Printf("loaded %d reference vectors", refs.N)

	h := &handler.Handler{
		Refs:    refs,
		MccRisk: mccRisk,
		Norm:    norm,
	}

	e := echo.New()
	e.HideBanner = true
	e.Use(otelecho.Middleware(telemetry.ServiceName))

	e.GET("/ready", h.Ready)
	e.POST("/fraud-score", h.FraudScore)

	go func() {
		<-ctx.Done()
		_ = e.Shutdown(context.Background())
	}()

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
