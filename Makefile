.PHONY: test test-simd test-force-scalar test-force-sse2 test-force-avx2 test-force-avx512 \
       bench bench-simd bench-simd-v2 bench-save-scalar bench-save-simd bench-compare \
       compare-with-fastpfor bench-matrix bench-matrix-table bench-matrix-compare \
       bench-quick bench-quick-save bench-ab \
       bench-threshold threshold-analyze bench-threshold-full \
       fuzz fuzz-simd fuzz-regression fuzz-regression-simd \
       generate-native generate \
       sim bench-binary

# Go toolchain binary to use (defaults to gotip).
# Override to easily revert:
#   make GO=go test
GO ?= gotip

# GOAMD64=v2 is recommended for consumers on modern x86-64 if they want the fastest popcount-heavy path

FUZZTIME ?= 30s
BENCHCOUNT ?= 10

# Pin benchmarks to P-cores on hybrid CPUs (Intel 12th gen+) for stable results.
# TASKSET_MICRO: single P-core for micro-benchmarks (lowest variance, ~1-5%).
# TASKSET:       P-core range for matrix/threshold benchmarks (accept 10-20% variance).
# For cross-run comparison, use benchstat with >= 10 samples and focus on
# relative patterns (ratios between test cases) rather than absolute values.
# Override with: make bench TASKSET_MICRO=""  (to disable)
# or:            make bench TASKSET_MICRO="taskset -c 2"  (custom core)
TASKSET_MICRO ?= taskset -c 0
TASKSET ?= taskset -c 0-7

FUZZ_TARGETS = FuzzPackUnpackUint32RoundTrip \
               FuzzPackDeltaUint32RoundTrip \
               FuzzGetUint32MatchesUnpack \
               FuzzBlockLengthNeverPanics \
               FuzzCorruptDeltaOverflow \
               FuzzDeltaWithExceptions \
               FuzzCompressionRatio \
               FuzzGetUint32CorruptBlock \
               FuzzCorruptBlockAPIs

FUZZ_SIMD_TARGETS = FuzzSIMDScalarConsistency

FASTPFOR_DIR ?= ../fastpfor-go

# --- Tests ---

test:
	$(GO) test ./... -count=1

test-simd:
	GOEXPERIMENT=simd $(GO) test ./... -count=1

test-force-scalar:
	GOEXPERIMENT=simd UTL_SIMD_LEVEL=scalar $(GO) test ./... -count=1

test-force-sse2:
	GOEXPERIMENT=simd UTL_SIMD_LEVEL=sse2 $(GO) test ./... -count=1

test-force-avx2:
	GOEXPERIMENT=simd UTL_SIMD_LEVEL=avx2 $(GO) test ./... -count=1

test-force-avx512:
	GOEXPERIMENT=simd UTL_SIMD_LEVEL=avx512 $(GO) test ./... -count=1

# --- Benchmarks ---

bench:
	$(TASKSET_MICRO) $(GO) test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./...

bench-simd:
	GOEXPERIMENT=simd $(TASKSET_MICRO) $(GO) test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./...

bench-simd-v2:
	GOAMD64=v2 GOEXPERIMENT=simd $(TASKSET_MICRO) $(GO) test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./...

bench-save-scalar:
	@mkdir -p benchmarks
	$(TASKSET_MICRO) $(GO) test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./... > benchmarks/scalar-baseline.txt
	@echo "Saved to benchmarks/scalar-baseline.txt"

bench-save-simd:
	@mkdir -p benchmarks
	GOEXPERIMENT=simd $(TASKSET_MICRO) $(GO) test -bench=. -benchmem -count=$(BENCHCOUNT) -run='^$$' ./... > benchmarks/simd-baseline.txt
	@echo "Saved to benchmarks/simd-baseline.txt"

bench-compare:
	@command -v benchstat >/dev/null 2>&1 || { echo "Install benchstat: $(GO) install golang.org/x/perf/cmd/benchstat@latest"; exit 1; }
	benchstat benchmarks/scalar-baseline.txt benchmarks/simd-baseline.txt

FASTPFOR_DIR ?= ../fastpfor-go

# --- Cross-repo comparison with fastpfor-go ---

compare-with-fastpfor:
	@command -v benchstat >/dev/null 2>&1 || { echo "Install benchstat: $(GO) install golang.org/x/perf/cmd/benchstat@latest"; exit 1; }
	@mkdir -p $(CURDIR)/benchmarks
	@echo "Running fastpfor-go benchmarks..."
	@cd $(FASTPFOR_DIR) && $(TASKSET) $(GO) test -bench='BenchmarkPackUint32$$|BenchmarkUnpackUint32$$|BenchmarkPackDeltaUint32$$|BenchmarkUnpackDeltaUint32$$|BenchmarkPackDeltaMixed$$|BenchmarkUnpackDeltaMixed$$|BenchmarkPackWithExceptions$$|BenchmarkUnpackWithExceptions$$|BenchmarkPackWithLargeExceptions$$|BenchmarkUnpackWithLargeExceptions$$|BenchmarkGetUint32WithExceptions$$|BenchmarkGetUint32Delta$$|BenchmarkBlockLength$$' \
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
	@GOEXPERIMENT=simd $(TASKSET) $(GO) test -bench='BenchmarkPackSequential$$|BenchmarkUnpackSequential$$|BenchmarkPackDeltaMonotonic$$|BenchmarkUnpackDeltaMonotonic$$|BenchmarkPackDeltaMixed$$|BenchmarkUnpackDeltaMixed$$|BenchmarkPackWithSmallExceptions$$|BenchmarkUnpackWithSmallExceptions$$|BenchmarkPackWithLargeExceptions$$|BenchmarkUnpackWithLargeExceptions$$|BenchmarkGetUint32WithExceptions$$|BenchmarkGetUint32Delta$$|BenchmarkBlockLength$$' \
		-benchmem -count=$(BENCHCOUNT) -run='^$$' ./... \
		> $(CURDIR)/benchmarks/utlpfor-comparable.txt
	@echo "--- Comparison (fastpfor-go vs utlpfor) ---"
	@benchstat $(CURDIR)/benchmarks/fastpfor-comparable.txt $(CURDIR)/benchmarks/utlpfor-comparable.txt

# --- Quick Comparison (~20 benchmarks x 4 SIMD levels, ~5 min) ---

QUICK_COUNT ?= 10

bench-quick:
	@command -v $(GO) >/dev/null 2>&1 || { echo "$(GO) not found" >&2; exit 1; }
	@for level in scalar sse2 avx2 avx512; do \
		printf '\n=== %s ===\n' "$$level"; \
		out="$$(GOEXPERIMENT=simd UTL_SIMD_LEVEL=$$level $(TASKSET) $(GO) test \
			-bench='BenchmarkQuickCompare|BenchmarkQuickCompareUint64' -benchmem -count=$(QUICK_COUNT) \
			-run='^$$' -timeout=300s ./... 2>&1 || true)"; \
		filtered="$$(printf '%s\n' "$$out" | grep -E 'BenchmarkQuickCompare(Uint64)?|PASS|FAIL|^ok[[:space:]]' || true)"; \
		if [ -n "$$filtered" ]; then \
			printf '%s\n' "$$filtered"; \
		else \
			printf '[skip] %s benchmarks unsupported on this CPU\n' "$$level"; \
		fi; \
	done

bench-quick-save:
	@command -v $(GO) >/dev/null 2>&1 || { echo "$(GO) not found" >&2; exit 1; }
	@mkdir -p benchmarks
	@STAMP=$$(date +%Y%m%d-%H%M); \
	for level in scalar sse2 avx2 avx512; do \
		printf '=== Running %s benchmarks ===\n' "$$level" >&2; \
		GOEXPERIMENT=simd UTL_SIMD_LEVEL=$$level $(TASKSET) $(GO) test \
			-bench='BenchmarkQuickCompare|BenchmarkQuickCompareUint64' -benchmem -count=$(QUICK_COUNT) \
			-run='^$$' -timeout=300s ./... \
			> benchmarks/quick-$$level-$$STAMP.txt 2>&1 || true; \
		if grep -Eq 'BenchmarkQuickCompare(Uint64)?/' benchmarks/quick-$$level-$$STAMP.txt; then \
			printf '  saved benchmarks/quick-%s-%s.txt\n' "$$level" "$$STAMP" >&2; \
		else \
			printf '  [skip] %s unsupported on this CPU\n' "$$level" >&2; \
		fi; \
	done

# --- Benchmark Matrix ---

MATRIX_BENCH ?= BenchmarkMatrix/|BenchmarkMatrixUint64/
MATRIX_COUNT ?= 5

bench-matrix:
	@command -v $(GO) >/dev/null 2>&1 || { echo "$(GO) not found"; exit 1; }
	@mkdir -p benchmarks
	@COMMIT=$$(git rev-parse --short HEAD); \
	echo "# Benchmarks at commit $$COMMIT"; \
	for level in scalar sse2 avx2 avx512; do \
		echo "=== Running $$level benchmarks ==="; \
		out="$$( { echo "# git commit: $$COMMIT"; \
		  GOEXPERIMENT=simd UTL_SIMD_LEVEL=$$level $(TASKSET) $(GO) test \
		    -bench='$(MATRIX_BENCH)' -benchmem -count=$(MATRIX_COUNT) \
		    -run='^$$' -timeout=300s ./... 2>&1; } || true)"; \
		echo "$$out" > benchmarks/matrix-$$level.txt; \
		if ! echo "$$out" | grep -q 'BenchmarkMatrix/'; then \
			echo "[skip] $$level matrix benchmarks unsupported on this CPU"; \
		fi; \
	done
	@echo "=== Formatting table ==="
	@$(GO) run ./internal/benchfmt \
		benchmarks/matrix-scalar.txt \
		benchmarks/matrix-sse2.txt \
		benchmarks/matrix-avx2.txt \
		benchmarks/matrix-avx512.txt

bench-matrix-table:
	@$(GO) run ./internal/benchfmt \
		benchmarks/matrix-scalar.txt \
		benchmarks/matrix-sse2.txt \
		benchmarks/matrix-avx2.txt \
		benchmarks/matrix-avx512.txt

bench-matrix-compare:
	@command -v benchstat >/dev/null 2>&1 || \
		{ echo "Install benchstat: $(GO) install golang.org/x/perf/cmd/benchstat@latest"; exit 1; }
	@if [ -z "$(OLD)" ] || [ -z "$(NEW)" ]; then \
		echo "Usage: make bench-matrix-compare OLD=benchmarks/matrix-avx2-old.txt NEW=benchmarks/matrix-avx2.txt"; \
		exit 1; \
	fi
	benchstat $(OLD) $(NEW)

# Interleaved A/B benchmarks - Usage:
#   make bench-ab                                        # HEAD vs worktree
#   make bench-ab ROUNDS=10 BENCH='BenchmarkUnpackDelta' # subset, more rounds
#   benchstat benchmarks/ab-old.txt benchmarks/ab-new.txt
AB_ROUNDS ?= 6
AB_BENCH ?= 'BenchmarkUnpackDelta|BenchmarkUnpackUint32$$|BenchmarkUnpackSequential|BenchmarkPackDeltaMonotonic|BenchmarkPackSequential'

bench-ab:
	@command -v benchstat >/dev/null 2>&1 || { echo "Install benchstat: $(GO) install golang.org/x/perf/cmd/benchstat@latest"; exit 1; }
	@mkdir -p benchmarks
	@git worktree add --detach benchmarks/.ab-head HEAD >/dev/null 2>&1 || true
	@(cd benchmarks/.ab-head && GOEXPERIMENT=simd $(GO) test -c -o $(CURDIR)/utlpfor-ab-old.test .)
	@GOEXPERIMENT=simd $(GO) test -c -o utlpfor-ab-new.test .
	@rm -f benchmarks/ab-old.txt benchmarks/ab-new.txt
	@i=1; while [ $$i -le $(AB_ROUNDS) ]; do \
		if [ $$((i % 2)) -eq 1 ]; then ORDER="utlpfor-ab-old.test utlpfor-ab-new.test"; \
		else ORDER="utlpfor-ab-new.test utlpfor-ab-old.test"; fi; \
		for bin in $$ORDER; do \
			case "$$bin" in \
				*old*) OUT=benchmarks/ab-old.txt ;; \
				*new*) OUT=benchmarks/ab-new.txt ;; \
			esac; \
			echo "=== ab round $$i/$(AB_ROUNDS) $$bin ===" >&2; \
			GOEXPERIMENT=simd taskset -c 2 ./$$bin \
				-test.bench=$(AB_BENCH) -test.benchmem -test.count=1 -test.run='^$$' \
				-test.timeout=900s >> $$OUT 2>&1; \
		done; \
		i=$$((i + 1)); \
	done
	@rm -f utlpfor-ab-old.test utlpfor-ab-new.test
	@git worktree remove --force benchmarks/.ab-head >/dev/null 2>&1 || true
	@echo "Saved benchmarks/ab-old.txt / benchmarks/ab-new.txt"
	@echo "Compare with: benchstat benchmarks/ab-old.txt benchmarks/ab-new.txt"

# --- Code Generation ---

generate-native:
	$(GO) run ./internal/gen | gofmt > simd_spec_amd64.go

generate: generate-native

# --- Fuzzing ---

fuzz:
	@for target in $(FUZZ_TARGETS); do \
		echo "=== Fuzzing $$target ($(FUZZTIME)) ==="; \
		$(GO) test -fuzz=$$target -fuzztime=$(FUZZTIME) ./... || exit 1; \
	done

fuzz-simd:
	@for target in $(FUZZ_TARGETS) $(FUZZ_SIMD_TARGETS); do \
		echo "=== Fuzzing $$target ($(FUZZTIME), SIMD) ==="; \
		GOEXPERIMENT=simd $(GO) test -fuzz=$$target -fuzztime=$(FUZZTIME) ./... || exit 1; \
	done

fuzz-regression:
	$(GO) test -run='Fuzz' -count=1 ./...

fuzz-regression-simd:
	GOEXPERIMENT=simd $(GO) test -run='Fuzz' -count=1 ./...

SIM_RUNS ?= 10
SIM_WARMUP ?= 5

sim:
	GOEXPERIMENT=simd $(GO) run ./internal/sim -runs $(SIM_RUNS) -warmup $(SIM_WARMUP)

# --- GetUint32 Threshold Tuning ---

THRESHOLD_COUNT ?= 5

bench-threshold:
	@mkdir -p benchmarks
	@for level in scalar sse2 avx2 avx512; do \
		printf '=== Running %s GetUint32_Approaches ===\n' "$$level"; \
		GOEXPERIMENT=simd UTL_SIMD_LEVEL=$$level $(TASKSET) $(GO) test \
			-bench=BenchmarkGetUint32_Approaches -benchmem -count=$(THRESHOLD_COUNT) \
			-run='^$$' -timeout=300s ./... \
			> benchmarks/get-$$level.txt 2>&1 || true; \
		if grep -q 'BenchmarkGetUint32_Approaches/' benchmarks/get-$$level.txt; then \
			printf '  saved benchmarks/get-%s.txt\n' "$$level"; \
		else \
			printf '  [skip] %s unsupported on this CPU\n' "$$level"; \
		fi; \
	done

threshold-analyze:
	@$(GO) run ./internal/threshold \
		benchmarks/get-scalar.txt \
		benchmarks/get-sse2.txt \
		benchmarks/get-avx2.txt \
		benchmarks/get-avx512.txt

bench-threshold-full: bench-threshold threshold-analyze

# --- Cross-Machine Benchmark Binary ---
#
# Build a standalone test binary that can be copied to a foreign machine
# and executed without a Go toolchain. The binary contains all benchmarks
# and can be run with standard `go test` flags.
#
# Usage:
#   make bench-binary                       # build the test binary
#   scp utlpfor-bench.test user@remote:     # copy to remote machine
#   ssh user@remote                         # log in
#   # On the remote machine:
#   UTL_SIMD_LEVEL=scalar ./utlpfor-bench.test \
#       -test.bench='BenchmarkMatrix/|BenchmarkMatrixUint64/' \
#       -test.benchmem -test.count=5 -test.run='^$' -test.timeout=600s \
#       > utlpfor-matrix-scalar.txt
#   UTL_SIMD_LEVEL=sse2 ./utlpfor-bench.test \
#       -test.bench='BenchmarkMatrix/|BenchmarkMatrixUint64/' \
#       -test.benchmem -test.count=5 -test.run='^$' -test.timeout=600s \
#       > utlpfor-matrix-sse2.txt
#   UTL_SIMD_LEVEL=avx2 ./utlpfor-bench.test \
#       -test.bench='BenchmarkMatrix/|BenchmarkMatrixUint64/' \
#       -test.benchmem -test.count=5 -test.run='^$' -test.timeout=600s \
#       > utlpfor-matrix-avx2.txt
#   UTL_SIMD_LEVEL=avx512 ./utlpfor-bench.test \
#       -test.bench='BenchmarkMatrix/|BenchmarkMatrixUint64/' \
#       -test.benchmem -test.count=5 -test.run='^$' -test.timeout=600s \
#       > utlpfor-matrix-avx512.txt
#   # Copy results back and format:
#   scp user@remote:matrix-*.txt benchmarks/
#   make bench-matrix-table ...
#
# Cross-compilation for different architectures:
#   GOOS=linux GOARCH=amd64 make bench-binary

BENCH_BINARY ?= utlpfor-bench.test

bench-binary:
	GOEXPERIMENT=simd $(GO) test -c -o $(BENCH_BINARY) .
	@printf 'Built %s (%s)\n' "$(BENCH_BINARY)" "$$(du -h $(BENCH_BINARY) | cut -f1)"
	@echo "Copy to remote and run with -test.bench flags (see Makefile for examples)"
