FROM rust:1.87-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY resources/references.json.gz  resources/
COPY resources/normalization.json  resources/
COPY resources/mcc_risk.json       resources/
ENV RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2,+fma,+f16c,+bmi2,+popcnt -C link-arg=-s"
RUN cargo build --release -p refvec-builder -p api
RUN ./target/release/refvec-builder

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/target/release/api .
COPY --from=builder /app/resources/references.ivfvec  resources/
COPY resources/normalization.json  resources/
COPY resources/mcc_risk.json       resources/
EXPOSE 9999
CMD ["./api"]
