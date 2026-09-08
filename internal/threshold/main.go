// Command threshold reads BenchmarkGetUint32_Approaches output files
// (one per SIMD level) and suggests deltaFullUnpackThreshold values.
//
// It compares single-lane walk cost at each position range against full-unpack
// cost (pos_00_15 with scratch is the cheapest; full unpack for the same block
// config is the reference). The crossover point where single-lane walk exceeds
// full unpack suggests the optimal threshold.
//
// Usage:
//
//	go run ./internal/threshold benchmarks/get-scalar.txt benchmarks/get-sse2.txt \
//	    benchmarks/get-avx2.txt benchmarks/get-avx512.txt
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

var benchRe = regexp.MustCompile(
	`^BenchmarkGetUint32_Approaches/(.+?)/pos_(\d+)_(\d+)/single(?:-\d+)?\s+\d+\s+(\d+(?:\.\d+)?)\s+ns/op`)

type result struct {
	config string
	posLo  int
	posHi  int
	nsOp   float64
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr,
			"Usage: %s <scalar.txt> [sse2.txt] [avx2.txt] [avx512.txt]\n\n"+
				"Reads BenchmarkGetUint32_Approaches output and suggests\n"+
				"deltaFullUnpackThreshold values per SIMD level.\n\n"+
				"Generate input with:\n"+
				"  for level in scalar sse2 avx2 avx512; do\n"+
				"    GOEXPERIMENT=simd UTL_SIMD_LEVEL=$level gotip test \\\n"+
				"      -bench=BenchmarkGetUint32_Approaches -benchmem -count=5 \\\n"+
				"      -run='^$' -timeout=300s ./... \\\n"+
				"      > benchmarks/get-$level.txt\n"+
				"  done\n",
			os.Args[0])
		os.Exit(1)
	}

	labels := []string{"scalar", "sse2", "avx2", "avx512"}

	for i, path := range os.Args[1:] {
		if i >= len(labels) {
			break
		}
		label := labels[i]
		results := parseFile(path)
		if len(results) == 0 {
			fmt.Fprintf(os.Stderr, "warning: no results in %s\n", path)
			continue
		}
		analyzeLevel(label, results)
	}
}

func parseFile(path string) []result {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: %s: %v\n", path, err)
		return nil
	}
	defer f.Close()

	var rawResults []result
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		m := benchRe.FindStringSubmatch(scanner.Text())
		if m == nil {
			continue
		}
		config := m[1]
		posLo, _ := strconv.Atoi(m[2])
		posHi, _ := strconv.Atoi(m[3])
		nsOp, _ := strconv.ParseFloat(m[4], 64)
		rawResults = append(rawResults, result{config, posLo, posHi, nsOp})
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: %s: %v\n", path, err)
		return nil
	}

	type key struct {
		config string
		posLo  int
		posHi  int
	}
	grouped := map[key][]float64{}
	for _, r := range rawResults {
		k := key{r.config, r.posLo, r.posHi}
		grouped[k] = append(grouped[k], r.nsOp)
	}

	var results []result
	for k, vals := range grouped {
		results = append(results, result{k.config, k.posLo, k.posHi, median(vals)})
	}
	return results
}

func analyzeLevel(label string, results []result) {
	fmt.Printf("\n=== %s ===\n\n", strings.ToUpper(label))

	configs := map[string]bool{}
	for _, r := range results {
		configs[r.config] = true
	}

	var sortedConfigs []string
	for c := range configs {
		sortedConfigs = append(sortedConfigs, c)
	}
	sort.Strings(sortedConfigs)

	posRanges := [][2]int{{0, 15}, {16, 31}, {32, 63}, {64, 95}, {96, 127}}
	posInLaneMap := map[int]string{
		0:  "posInLane=0",
		16: "posInLane=1",
		32: "posInLane=2-3",
		64: "posInLane=4-5",
		96: "posInLane=6-7",
	}

	fmt.Printf("%-20s", "Config")
	for _, pr := range posRanges {
		fmt.Printf(" | %14s", posInLaneMap[pr[0]])
	}
	fmt.Println(" |")
	fmt.Print(strings.Repeat("-", 20))
	for range posRanges {
		fmt.Print("-|-" + strings.Repeat("-", 14))
	}
	fmt.Println("-|")

	for _, config := range sortedConfigs {
		fmt.Printf("%-20s", config)
		for _, pr := range posRanges {
			found := false
			for _, r := range results {
				if r.config == config && r.posLo == pr[0] && r.posHi == pr[1] {
					fmt.Printf(" | %10.1f ns", r.nsOp)
					found = true
					break
				}
			}
			if !found {
				fmt.Printf(" | %14s", "N/A")
			}
		}
		fmt.Println(" |")
	}

	fmt.Println("\nThreshold suggestions (posInLane above which full-unpack is faster):")
	fmt.Println("  Compare delta single-lane cost at each range against full-unpack baseline.")
	fmt.Println("  When single-lane ns exceeds ~2x of pos_00_15 cost, full-unpack may win.")
	fmt.Println()

	for _, config := range sortedConfigs {
		if !strings.HasPrefix(config, "delta") {
			continue
		}
		var baseline float64
		for _, r := range results {
			if r.config == config && r.posLo == 0 {
				baseline = r.nsOp
				break
			}
		}
		if baseline == 0 {
			continue
		}

		threshold := -1
		for _, pr := range posRanges {
			for _, r := range results {
				if r.config == config && r.posLo == pr[0] {
					ratio := r.nsOp / baseline
					if ratio > 2.5 && threshold < 0 {
						threshold = pr[0] / 16
					}
				}
			}
		}

		if threshold >= 0 {
			fmt.Printf("  %-20s: threshold = %d (posInLane)\n", config, threshold)
		} else {
			fmt.Printf("  %-20s: lane walk always preferred (no crossover)\n", config)
		}
	}
}

func median(vals []float64) float64 {
	if len(vals) == 0 {
		return math.NaN()
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}
