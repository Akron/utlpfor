// Command benchfmt reads raw Go benchmark output files (one per SIMD level)
// and prints a Markdown comparison table with median ns/op values.
//
// The output table has option columns (bit-width, excCount, for, delta,
// zigzag, nofor+nopatch) and result columns (Scalar, SSE2, AVX2, AVX512).
// Scalar-only operations (Header, BlockLength) show "=" in SIMD columns.
// Different FOR widths are combined into a single "for: +/-" column.
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

// nameRe parses the structured benchmark name format:
// method_XXbit_XXexc_Xfor_+/-delta_+/-zz_+/-nfnp
var nameRe = regexp.MustCompile(
	`^(\w+)_(\d+)bit_(\d+)exc_(\d+)for_([+-])delta_([+-])zz_([+-])nfnp$`)

// rowKey identifies a unique row in the output table.
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

func methodSortOrder(m string) int {
	order := map[string]int{
		"pac": 0, "unp": 1, "get": 2, "hdr": 3, "len": 4, "mix": 5,
	}
	if v, ok := order[m]; ok {
		return v
	}
	return 99
}

var methodDisplayNames = map[string]string{
	"pac": "PackUint32",
	"unp": "UnpackUint32",
	"get": "GetUint32",
	"hdr": "Header",
	"len": "BlockLength",
	"mix": "Mixed",
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

func main() {
	if len(os.Args) < 5 {
		fmt.Fprintf(os.Stderr,
			"Usage: %s scalar.txt sse2.txt avx2.txt avx512.txt\n",
			os.Args[0])
		os.Exit(1)
	}

	allRows := map[string]*rowData{}
	var sortKeys []string

	for level, path := range os.Args[1:5] {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: %s: %v (skipping)\n", path, err)
			continue
		}
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			m := benchRe.FindStringSubmatch(scanner.Text())
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

	if len(allRows) == 0 {
		fmt.Fprintln(os.Stderr, "no benchmark results found")
		os.Exit(1)
	}

	sort.Strings(sortKeys)

	const numCols = 12
	headers := [numCols]string{
		"Operation", "bit-width", "excCount", "for",
		"delta", "zigzag", "nofor+nopatch",
		"", // separator column
		"Scalar", "SSE2", "AVX2", "AVX512",
	}

	// Build formatted cell values and track max widths.
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
				median := medianFloat64(rd.vals[level])
				if median < 100 {
					fr[col] = fmt.Sprintf("%.1f ns", median)
				} else {
					fr[col] = fmt.Sprintf("%.0f ns", median)
				}
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

	// Print header row.
	printRow(headers[:], colWidths[:])

	// Print separator row with alignment markers.
	fmt.Print("|")
	for i := range numCols {
		w := colWidths[i]
		if i >= 8 {
			fmt.Printf(" %s:|", strings.Repeat("-", w))
		} else {
			fmt.Printf(" %s |", strings.Repeat("-", w))
		}
	}
	fmt.Println()

	// Print data rows.
	for _, fr := range rows {
		printRowAligned(fr[:], colWidths[:])
	}

	fmt.Fprintf(os.Stderr, "(%d rows)\n", len(rows))
}

// printRow prints a Markdown table row with left-aligned cells.
func printRow(cells []string, widths []int) {
	fmt.Print("|")
	for i, c := range cells {
		fmt.Printf(" %-*s |", widths[i], c)
	}
	fmt.Println()
}

// printRowAligned prints a Markdown table row with right-aligned result columns.
func printRowAligned(cells []string, widths []int) {
	fmt.Print("|")
	for i, c := range cells {
		if i >= 8 {
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
