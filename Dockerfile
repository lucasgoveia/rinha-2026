FROM rust:1.87-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN cargo build -p api --release

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/target/release/api .
EXPOSE 9999
CMD ["./api"]
