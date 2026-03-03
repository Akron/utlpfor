.PHONY: test test-simd bench bench-simd

# Scalar-only tests (no SIMD experiment)
test:
	go test ./... -count=1

# Tests with SIMD experiment enabled
test-simd:
	GOEXPERIMENT=simd go test ./... -count=1

# Benchmarks (scalar)
bench:
	go test -bench=. -benchmem -count=5 ./...

# Benchmarks (SIMD)
bench-simd:
	GOEXPERIMENT=simd go test -bench=. -benchmem -count=5 ./...