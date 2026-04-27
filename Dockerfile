FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o api ./cmd/api

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/api .
COPY resources/ ./resources/
EXPOSE 9999
CMD ["./api"]
