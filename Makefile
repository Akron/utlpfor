.PHONY: test test-simd bench bench-simd bench-save-scalar bench-save-simd bench-compare \
       fuzz fuzz-simd fuzz-regression fuzz-regression-simd

FUZZTIME ?= 30s
BENCHCOUNT ?= 10

FUZZ_TARGETS = FuzzPackUnpackUint32RoundTrip \
               FuzzPackDeltaUint32RoundTrip \
               FuzzGetUint32MatchesUnpack \
               FuzzBlockLengthNeverPanics \
               FuzzCorruptDeltaOverflow \
               FuzzDeltaWithExceptions \
               FuzzCompressionRatio

FUZZ_SIMD_TARGETS = FuzzSIMDScalarConsistency

# --- Tests ---

test:
	go test ./... -count=1

test-simd:
	GOEXPERIMENT=simd go test ./... -count=1

# --- Benchmarks ---

bench:
	go test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./...

bench-simd:
	GOEXPERIMENT=simd go test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./...

bench-save-scalar:
	@mkdir -p benchmarks
	go test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./... > benchmarks/scalar-baseline.txt
	@echo "Saved to benchmarks/scalar-baseline.txt"

bench-save-simd:
	@mkdir -p benchmarks
	GOEXPERIMENT=simd go test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./... > benchmarks/simd-baseline.txt
	@echo "Saved to benchmarks/simd-baseline.txt"

bench-compare:
	@command -v benchstat >/dev/null 2>&1 || { echo "Install benchstat: go install golang.org/x/perf/cmd/benchstat@latest"; exit 1; }
	benchstat benchmarks/scalar-baseline.txt benchmarks/simd-baseline.txt

# --- Fuzzing ---

fuzz:
	@for target in $(FUZZ_TARGETS); do \
		echo "=== Fuzzing $$target ($(FUZZTIME)) ==="; \
		go test -fuzz=$$target -fuzztime=$(FUZZTIME) ./... || exit 1; \
	done

fuzz-simd:
	@for target in $(FUZZ_TARGETS) $(FUZZ_SIMD_TARGETS); do \
		echo "=== Fuzzing $$target ($(FUZZTIME), SIMD) ==="; \
		GOEXPERIMENT=simd go test -fuzz=$$target -fuzztime=$(FUZZTIME) ./... || exit 1; \
	done

fuzz-regression:
	go test -run='Fuzz' -count=1 ./...

fuzz-regression-simd:
	GOEXPERIMENT=simd go test -run='Fuzz' -count=1 ./...
