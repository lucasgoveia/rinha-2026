# ===== Stage 1: instrumented build (PGO profile-generate) =====
FROM rust:1.87-alpine AS pgo-instr
RUN apk add --no-cache musl-dev
RUN rustup component add llvm-tools-preview
WORKDIR /app
COPY rinha-2026/ .
ENV RUSTFLAGS="-Cprofile-generate=/tmp/pgo"
RUN mkdir -p /tmp/pgo \
 && cargo build --release --target x86_64-unknown-linux-musl \
        --bin api --bin builder --bin pgo_train

# ===== Stage 2: collect profile via pgo_train against a throwaway index =====
FROM pgo-instr AS pgo-data
COPY challenge/resources/ /resources/
ENV LLVM_PROFILE_FILE=/tmp/pgo/%m_%p.profraw
RUN /app/target/x86_64-unknown-linux-musl/release/builder \
      /resources/references.json.gz /tmp/index_tmp.bin
RUN /app/target/x86_64-unknown-linux-musl/release/pgo_train \
      /resources/references.json.gz /tmp/index_tmp.bin
RUN SYSROOT=$(rustc --print sysroot) && \
    "$SYSROOT/lib/rustlib/x86_64-unknown-linux-musl/bin/llvm-profdata" merge \
        -o /tmp/pgo.profdata /tmp/pgo

# ===== Stage 3: PGO-optimized rebuild (api + builder) =====
FROM rust:1.87-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY rinha-2026/ .
COPY --from=pgo-data /tmp/pgo.profdata /tmp/pgo.profdata
ENV RUSTFLAGS="-Cprofile-use=/tmp/pgo.profdata"
RUN cargo build --release --target x86_64-unknown-linux-musl \
        --bin api --bin builder

# ===== Stage 4: build the final pre-warmed index with the optimized builder =====
FROM build AS indexed
COPY challenge/resources/ /resources/
RUN /app/target/x86_64-unknown-linux-musl/release/builder \
      /resources/references.json.gz /tmp/index.bin

# ===== Stage 5: minimal scratch runtime image =====
FROM scratch
COPY --from=build   /app/target/x86_64-unknown-linux-musl/release/api /api
COPY --from=indexed /tmp/index.bin                /data/index.bin
COPY --from=indexed /resources/mcc_risk.json      /data/mcc_risk.json
COPY --from=indexed /resources/normalization.json /data/normalization.json
ENTRYPOINT ["/api"]
