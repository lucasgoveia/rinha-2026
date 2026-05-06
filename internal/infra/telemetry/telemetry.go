package telemetry

import (
	"context"
	"log"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
)

const ServiceName = "rinha-fraud-api"

// Setup initializes the OTel tracer. Returns a shutdown func.
// Falls back to no-op if OTEL_EXPORTER_OTLP_ENDPOINT is unset.
func Setup(ctx context.Context) func(context.Context) {
	if os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT") == "" {
		log.Println("otel: no endpoint configured, tracing disabled")
		return func(context.Context) {}
	}

	res := resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceName(ServiceName),
	)

	exp, err := otlptracegrpc.New(ctx)
	if err != nil {
		log.Printf("otel: trace exporter failed: %v, tracing disabled", err)
		return func(context.Context) {}
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)
	otel.SetTracerProvider(tp)

	log.Printf("otel: tracing -> %s", os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))

	return func(ctx context.Context) {
		_ = tp.Shutdown(ctx)
	}
}
