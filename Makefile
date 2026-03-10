.PHONY: test test-simd test-force-scalar test-force-sse2 test-force-avx2 test-force-avx512 \
       bench bench-simd bench-save-scalar bench-save-simd bench-compare \
       compare-with-fastpfor fuzz fuzz-simd fuzz-regression fuzz-regression-simd \
       generate-native generate

FUZZTIME ?= 30s
BENCHCOUNT ?= 10

# Pin benchmarks to P-cores on hybrid CPUs (Intel 12th gen+) for stable results.
# Override with: make bench TASKSET=""  (to disable)
# or:            make bench TASKSET="taskset -c 0-3"  (custom core set)
TASKSET ?= taskset -c 0-7

FUZZ_TARGETS = FuzzPackUnpackUint32RoundTrip \
               FuzzPackDeltaUint32RoundTrip \
               FuzzGetUint32MatchesUnpack \
               FuzzBlockLengthNeverPanics \
               FuzzCorruptDeltaOverflow \
               FuzzDeltaWithExceptions \
               FuzzCompressionRatio

FUZZ_SIMD_TARGETS = FuzzSIMDScalarConsistency

FASTPFOR_DIR ?= ../fastpfor-go

# --- Tests ---

test:
	go test ./... -count=1

test-simd:
	GOEXPERIMENT=simd go test ./... -count=1

test-force-scalar:
	GOEXPERIMENT=simd UTL_SIMD_LEVEL=scalar go test ./... -count=1

test-force-sse2:
	GOEXPERIMENT=simd UTL_SIMD_LEVEL=sse2 go test ./... -count=1

test-force-avx2:
	GOEXPERIMENT=simd UTL_SIMD_LEVEL=avx2 go test ./... -count=1

test-force-avx512:
	GOEXPERIMENT=simd UTL_SIMD_LEVEL=avx512 go test ./... -count=1

# --- Benchmarks ---

bench:
	$(TASKSET) go test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./...

bench-simd:
	GOEXPERIMENT=simd $(TASKSET) go test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./...

bench-save-scalar:
	@mkdir -p benchmarks
	$(TASKSET) go test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./... > benchmarks/scalar-baseline.txt
	@echo "Saved to benchmarks/scalar-baseline.txt"

bench-save-simd:
	@mkdir -p benchmarks
	GOEXPERIMENT=simd $(TASKSET) go test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./... > benchmarks/simd-baseline.txt
	@echo "Saved to benchmarks/simd-baseline.txt"

bench-compare:
	@command -v benchstat >/dev/null 2>&1 || { echo "Install benchstat: go install golang.org/x/perf/cmd/benchstat@latest"; exit 1; }
	benchstat benchmarks/scalar-baseline.txt benchmarks/simd-baseline.txt

FASTPFOR_DIR ?= ../fastpfor-go

# --- Cross-repo comparison with fastpfor-go ---

compare-with-fastpfor:
	@command -v benchstat >/dev/null 2>&1 || { echo "Install benchstat: go install golang.org/x/perf/cmd/benchstat@latest"; exit 1; }
	@mkdir -p $(CURDIR)/benchmarks
	@echo "Running fastpfor-go benchmarks..."
	@cd $(FASTPFOR_DIR) && $(TASKSET) go test -bench='BenchmarkPackUint32$$|BenchmarkUnpackUint32$$|BenchmarkPackDeltaUint32$$|BenchmarkUnpackDeltaUint32$$|BenchmarkPackDeltaMixed$$|BenchmarkUnpackDeltaMixed$$|BenchmarkPackWithExceptions$$|BenchmarkUnpackWithExceptions$$|BenchmarkPackWithLargeExceptions$$|BenchmarkUnpackWithLargeExceptions$$|BenchmarkGetUint32WithExceptions$$|BenchmarkGetUint32Delta$$|BenchmarkBlockLength$$' \
		-benchmem -count=$(BENCHCOUNT) -run='^$$' ./... \
		| sed -E \
			-e 's/BenchmarkPackUint32-/BenchmarkPackSequential-/' \
			-e 's/BenchmarkUnpackUint32-/BenchmarkUnpackSequential-/' \
			-e 's/BenchmarkPackDeltaUint32-/BenchmarkPackDeltaMonotonic-/' \
			-e 's/BenchmarkUnpackDeltaUint32-/BenchmarkUnpackDeltaMonotonic-/' \
			-e 's/BenchmarkPackWithExceptions-/BenchmarkPackWithSmallExceptions-/' \
			-e 's/BenchmarkUnpackWithExceptions-/BenchmarkUnpackWithSmallExceptions-/' \
		> $(CURDIR)/benchmarks/fastpfor-comparable.txt
	@echo "Running utlpfor benchmarks..."
	@GOEXPERIMENT=simd $(TASKSET) go test -bench='BenchmarkPackSequential$$|BenchmarkUnpackSequential$$|BenchmarkPackDeltaMonotonic$$|BenchmarkUnpackDeltaMonotonic$$|BenchmarkPackDeltaMixed$$|BenchmarkUnpackDeltaMixed$$|BenchmarkPackWithSmallExceptions$$|BenchmarkUnpackWithSmallExceptions$$|BenchmarkPackWithLargeExceptions$$|BenchmarkUnpackWithLargeExceptions$$|BenchmarkGetUint32WithExceptions$$|BenchmarkGetUint32Delta$$|BenchmarkBlockLength$$' \
		-benchmem -count=$(BENCHCOUNT) -run='^$$' ./... \
		> $(CURDIR)/benchmarks/utlpfor-comparable.txt
	@echo "--- Comparison (fastpfor-go vs utlpfor) ---"
	@benchstat $(CURDIR)/benchmarks/fastpfor-comparable.txt $(CURDIR)/benchmarks/utlpfor-comparable.txt

# --- Code Generation ---

generate-native:
	go run ./internal/gen | gofmt > simd_spec_amd64.go

generate: generate-native

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
