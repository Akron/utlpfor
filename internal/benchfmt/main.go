// Command benchfmt reads raw Go benchmark output files (one per SIMD level)
// and prints a formatted comparison table with median ns/op values.
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
	`^BenchmarkMatrix/(.+?)(?:-\d+)?\s+\d+\s+(\d+(?:\.\d+)?)\s+ns/op`)

func main() {
	if len(os.Args) < 5 {
		fmt.Fprintf(os.Stderr,
			"Usage: %s scalar.txt sse2.txt avx2.txt avx512.txt\n",
			os.Args[0])
		os.Exit(1)
	}

	labels := [4]string{"Scalar", "SSE2", "AVX2", "AVX512"}
	allResults := map[string]*[4][]float64{}

	for level, path := range os.Args[1:5] {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: %s: %v (skipping)\n",
				path, err)
			continue
		}
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			m := benchRe.FindStringSubmatch(scanner.Text())
			if m == nil {
				continue
			}
			name := m[1]
			ns, _ := strconv.ParseFloat(m[2], 64)
			if _, ok := allResults[name]; !ok {
				allResults[name] = &[4][]float64{}
			}
			allResults[name][level] = append(
				allResults[name][level], ns)
		}
		f.Close()
	}

	if len(allResults) == 0 {
		fmt.Fprintln(os.Stderr, "no benchmark results found")
		os.Exit(1)
	}

	var names []string
	for n := range allResults {
		names = append(names, n)
	}
	sort.Strings(names)

	nameWidth := 42
	colWidth := 10
	fmt.Printf("%-*s", nameWidth, "Benchmark")
	for _, l := range labels {
		fmt.Printf(" | %*s", colWidth, l)
	}
	fmt.Println(" |")
	fmt.Print(strings.Repeat("-", nameWidth))
	for range labels {
		fmt.Print("-|-" + strings.Repeat("-", colWidth))
	}
	fmt.Println("-|")

	for _, name := range names {
		r := allResults[name]
		fmt.Printf("%-*s", nameWidth, name)
		for level := range 4 {
			if len(r[level]) == 0 {
				fmt.Printf(" | %*s", colWidth, "N/A")
				continue
			}
			median := medianFloat64(r[level])
			fmt.Printf(" | %*.0f ns", colWidth-3, median)
		}
		fmt.Println(" |")
	}

	fmt.Printf("\n(%d benchmarks)\n", len(names))
}

func medianFloat64(vals []float64) float64 {
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
