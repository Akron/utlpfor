// Command benchfmt reads raw Go benchmark output files (one per SIMD level)
// and prints a Markdown comparison table with median ns/op values.
//
// The output table has option columns (bit-width, excCount, for, delta,
// zigzag, nofor+nopatch) and result columns (Scalar, SSE2, AVX2, AVX512).
// Scalar-only operations (Header, BlockLength) show "=" in SIMD columns.
// Different FOR widths are combined into a single "for: +/-" column.
//
// Uint64 benchmarks (BenchmarkMatrixUint64) are printed in a separate
// table with columns: Operation, scenario, delta, Scalar, SSE2, AVX2, AVX512.
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

// benchRe matches Go benchmark output lines from BenchmarkMatrix or BenchmarkQuickCompare.
var benchRe = regexp.MustCompile(
	`^Benchmark(?:Matrix|QuickCompare)/(.+?)(?:-\d+)?\s+\d+\s+(\d+(?:\.\d+)?)\s+ns/op`)

// benchUint64Re matches Go benchmark output lines from BenchmarkMatrixUint64 or BenchmarkQuickCompareUint64.
var benchUint64Re = regexp.MustCompile(
	`^Benchmark(?:MatrixUint64|QuickCompareUint64)/(.+?)(?:-\d+)?\s+\d+\s+(\d+(?:\.\d+)?)\s+ns/op`)

// nameRe parses the structured uint32 benchmark name format:
// method_XXbit_XXexc_Xfor_+/-delta_+/-zz_+/-nfnp
var nameRe = regexp.MustCompile(
	`^(\w+)_(\d+)bit_(\d+)exc_(\d+)for_([+-])delta_([+-])zz_([+-])nfnp$`)

// nameUint64Re parses the structured uint64 benchmark name format:
// method_scenario or method_scenario+delta
var nameUint64Re = regexp.MustCompile(
	`^(pac64|unp64|get64|mix64)_(\w+?)(\+delta)?$`)

// rowKey identifies a unique row in the uint32 output table.
// FOR width is collapsed to a boolean (any non-zero width -> true).
type rowKey struct {
	method   string
	bitWidth int
	excCount int
	hasFOR   bool
	hasDelta bool
	hasZZ    bool
	hasNFNP  bool
}

func (k rowKey) sortTuple() string {
	return fmt.Sprintf("%02d_%03d_%03d_%t_%t_%t_%t",
		methodSortOrder(k.method), k.bitWidth, k.excCount,
		k.hasFOR, k.hasDelta, k.hasZZ, k.hasNFNP)
}

// rowKey64 identifies a unique row in the uint64 output table.
type rowKey64 struct {
	method   string
	scenario string
	hasDelta bool
}

func (k rowKey64) sortTuple() string {
	return fmt.Sprintf("%02d_%s_%t",
		methodSortOrder64(k.method), k.scenario, k.hasDelta)
}

func methodSortOrder(m string) int {
	order := map[string]int{
		"pac": 0, "unp": 1, "get": 2, "hdr": 3, "len": 4, "mix": 5,
	}
	if v, ok := order[m]; ok {
		return v
	}
	return 99
}

func methodSortOrder64(m string) int {
	order := map[string]int{
		"pac64": 0, "unp64": 1, "get64": 2, "mix64": 3,
	}
	if v, ok := order[m]; ok {
		return v
	}
	return 99
}

var methodDisplayNames = map[string]string{
	"pac":   "PackUint32",
	"unp":   "UnpackUint32",
	"get":   "GetUint32",
	"hdr":   "Header",
	"len":   "BlockLength",
	"mix":   "Mixed",
	"pac64": "PackUint64",
	"unp64": "UnpackUint64",
	"get64": "GetUint64",
	"mix64": "MixedUint64",
}

func isScalarOnly(method string) bool {
	return method == "hdr" || method == "len"
}

func boolFlag(b bool) string {
	if b {
		return "+"
	}
	return "-"
}

type rowData struct {
	key  rowKey
	vals [4][]float64
}

type rowData64 struct {
	key  rowKey64
	vals [4][]float64
}

func main() {
	if len(os.Args) < 5 {
		fmt.Fprintf(os.Stderr,
			"Usage: %s scalar.txt sse2.txt avx2.txt avx512.txt\n",
			os.Args[0])
		os.Exit(1)
	}

	allRows := map[string]*rowData{}
	var sortKeys []string

	allRows64 := map[string]*rowData64{}
	var sortKeys64 []string

	for level, path := range os.Args[1:5] {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: %s: %v (skipping)\n", path, err)
			continue
		}
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()

			// Try uint64 benchmarks first (more specific match).
			if m := benchUint64Re.FindStringSubmatch(line); m != nil {
				benchName := m[1]
				ns, _ := strconv.ParseFloat(m[2], 64)

				nm := nameUint64Re.FindStringSubmatch(benchName)
				if nm == nil {
					continue
				}

				rk := rowKey64{
					method:   nm[1],
					scenario: nm[2],
					hasDelta: nm[3] == "+delta",
				}
				sk := rk.sortTuple()
				if _, ok := allRows64[sk]; !ok {
					allRows64[sk] = &rowData64{key: rk}
					sortKeys64 = append(sortKeys64, sk)
				}
				allRows64[sk].vals[level] = append(allRows64[sk].vals[level], ns)
				continue
			}

			// Try uint32 benchmarks.
			m := benchRe.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			benchName := m[1]
			ns, _ := strconv.ParseFloat(m[2], 64)

			nm := nameRe.FindStringSubmatch(benchName)
			if nm == nil {
				continue
			}

			method := nm[1]
			bw, _ := strconv.Atoi(nm[2])
			exc, _ := strconv.Atoi(nm[3])
			fw, _ := strconv.Atoi(nm[4])

			rk := rowKey{
				method:   method,
				bitWidth: bw,
				excCount: exc,
				hasFOR:   fw > 0,
				hasDelta: nm[5] == "+",
				hasZZ:    nm[6] == "+",
				hasNFNP:  nm[7] == "+",
			}

			sk := rk.sortTuple()
			if _, ok := allRows[sk]; !ok {
				allRows[sk] = &rowData{key: rk}
				sortKeys = append(sortKeys, sk)
			}
			allRows[sk].vals[level] = append(allRows[sk].vals[level], ns)
		}
		f.Close()
	}

	if len(allRows) == 0 && len(allRows64) == 0 {
		fmt.Fprintln(os.Stderr, "no benchmark results found")
		os.Exit(1)
	}

	totalRows := 0

	// Print uint32 table.
	if len(allRows) > 0 {
		sort.Strings(sortKeys)
		totalRows += printUint32Table(allRows, sortKeys)
	}

	// Print uint64 table.
	if len(allRows64) > 0 {
		if len(allRows) > 0 {
			fmt.Println()
		}
		sort.Strings(sortKeys64)
		totalRows += printUint64Table(allRows64, sortKeys64)
	}

	fmt.Fprintf(os.Stderr, "(%d rows)\n", totalRows)
}

func printUint32Table(allRows map[string]*rowData, sortKeys []string) int {
	const numCols = 12
	headers := [numCols]string{
		"Operation", "bit-width", "excCount", "for",
		"delta", "zigzag", "nofor+nopatch",
		"", // separator column
		"Scalar", "SSE2", "AVX2", "AVX512",
	}

	colWidths := [numCols]int{}
	for i, h := range headers {
		colWidths[i] = len(h)
	}

	type fmtRow [numCols]string
	var rows []fmtRow

	for _, sk := range sortKeys {
		rd := allRows[sk]
		k := rd.key
		var fr fmtRow

		name := methodDisplayNames[k.method]
		if name == "" {
			name = k.method
		}
		fr[0] = name
		fr[1] = strconv.Itoa(k.bitWidth)
		fr[2] = strconv.Itoa(k.excCount)
		fr[3] = boolFlag(k.hasFOR)
		fr[4] = boolFlag(k.hasDelta)
		fr[5] = boolFlag(k.hasZZ)
		fr[6] = boolFlag(k.hasNFNP)
		fr[7] = ""

		scalarOnly := isScalarOnly(k.method)
		for level := range 4 {
			col := 8 + level
			if scalarOnly && level > 0 {
				fr[col] = "="
			} else if len(rd.vals[level]) == 0 {
				fr[col] = "N/A"
			} else {
				fr[col] = fmtNs(medianFloat64(rd.vals[level]))
			}
		}

		for i, c := range fr {
			if len(c) > colWidths[i] {
				colWidths[i] = len(c)
			}
		}
		rows = append(rows, fr)
	}

	if colWidths[7] < 1 {
		colWidths[7] = 1
	}

	printRow(headers[:], colWidths[:])
	printSeparator(colWidths[:], 8)
	for _, fr := range rows {
		printRowAligned(fr[:], colWidths[:], 8)
	}

	return len(rows)
}

func printUint64Table(allRows64 map[string]*rowData64, sortKeys64 []string) int {
	const numCols64 = 8
	headers := [numCols64]string{
		"Operation", "scenario", "delta",
		"", // separator column
		"Scalar", "SSE2", "AVX2", "AVX512",
	}

	colWidths := [numCols64]int{}
	for i, h := range headers {
		colWidths[i] = len(h)
	}

	type fmtRow [numCols64]string
	var rows []fmtRow

	for _, sk := range sortKeys64 {
		rd := allRows64[sk]
		k := rd.key
		var fr fmtRow

		name := methodDisplayNames[k.method]
		if name == "" {
			name = k.method
		}
		fr[0] = name
		fr[1] = k.scenario
		fr[2] = boolFlag(k.hasDelta)
		fr[3] = ""

		for level := range 4 {
			col := 4 + level
			if len(rd.vals[level]) == 0 {
				fr[col] = "N/A"
			} else {
				fr[col] = fmtNs(medianFloat64(rd.vals[level]))
			}
		}

		for i, c := range fr {
			if len(c) > colWidths[i] {
				colWidths[i] = len(c)
			}
		}
		rows = append(rows, fr)
	}

	if colWidths[3] < 1 {
		colWidths[3] = 1
	}

	printRow(headers[:], colWidths[:])
	printSeparator(colWidths[:], 4)
	for _, fr := range rows {
		printRowAligned(fr[:], colWidths[:], 4)
	}

	return len(rows)
}

func fmtNs(median float64) string {
	if median < 100 {
		return fmt.Sprintf("%.1f ns", median)
	}
	return fmt.Sprintf("%.0f ns", median)
}

// printRow prints a Markdown table row with left-aligned cells.
func printRow(cells []string, widths []int) {
	fmt.Print("|")
	for i, c := range cells {
		fmt.Printf(" %-*s |", widths[i], c)
	}
	fmt.Println()
}

// printSeparator prints the Markdown separator row. Columns at index
// >= resultStart use right-alignment markers.
func printSeparator(widths []int, resultStart int) {
	fmt.Print("|")
	for i, w := range widths {
		if i >= resultStart {
			fmt.Printf(" %s:|", strings.Repeat("-", w))
		} else {
			fmt.Printf(" %s |", strings.Repeat("-", w))
		}
	}
	fmt.Println()
}

// printRowAligned prints a Markdown table row with right-aligned result
// columns (those at index >= resultStart).
func printRowAligned(cells []string, widths []int, resultStart int) {
	fmt.Print("|")
	for i, c := range cells {
		if i >= resultStart {
			fmt.Printf(" %*s |", widths[i], c)
		} else {
			fmt.Printf(" %-*s |", widths[i], c)
		}
	}
	fmt.Println()
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
