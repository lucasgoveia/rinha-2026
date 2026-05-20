FROM scratch
COPY zig-out/bin/lb /lb
ENTRYPOINT ["/lb", "--listen", "0.0.0.0:9999", \
  "--backend", "/run/sock/api1.sock", \
  "--backend", "/run/sock/api2.sock"]
