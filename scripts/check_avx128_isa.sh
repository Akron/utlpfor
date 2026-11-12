#!/bin/bash
# ISA check for the 128-bit ("AVX") kernels of utlpfor.
#
# archsimd emits only VEX-encoded instructions, so the 128-bit kernels need AVX
# but they must NOT need AVX2 (or AVX-512): the dispatch selects them on every
# CPU that has AVX and lacks AVX2 (Sandy Bridge, Ivy Bridge, Bulldozer ...).
# CPUs without AVX never reach them (scalar code runs there).
#
# The script builds the test binary with GOEXPERIMENT=simd, disassembles it and
# fails if a 128-bit kernel contains an instruction that needs more than AVX.
#
# Usage: scripts/check_avx128_isa.sh [PACKAGE_DIR]   (default: .)
# Needs: Go 1.27 (override with GO=...), awk.
set -eu

GO=${GO:-go}
dir=${1:-.}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

(cd "$dir" && GOEXPERIMENT=simd "$GO" test -c -o "$tmp/pkg.test" .)
"$GO" tool objdump "$tmp/pkg.test" > "$tmp/pkg.dis"

# A function is a 128-bit kernel when its name contains SSE2 (historical name of
# the level, see Q-3 of utlpfor-sse-issue-currentplan.md) or AVX128, and it is
# not test code.
awk '
function report(why) {
	bad++
	printf("%s: %s: %s\n", fn, why, $0)
}
/^TEXT / {
	fn = $2
	src = $3
	kernel = (fn ~ /utlpfor\.[A-Za-z0-9_]*(SSE2|AVX128)/ && src !~ /_test\.go/)
	if (kernel) {
		nfn++
	}
	next
}
kernel {
	# fields: file:line, address, bytes, mnemonic, operands...
	insn = $4
	ninsn++
	if (insn ~ /^V(PBROADCAST|BROADCAST|PERM|PSLLV|PSRLV|PSRAV|INSERTI128|EXTRACTI128|PGATHER|GATHER|PMASKMOV|PBLENDD)/) {
		report("AVX2-only instruction")
	}
	# Y/Z registers, EVEX-only registers X16-X31, mask registers.
	if ($0 ~ /[ ,(][YZ]([0-9]|[12][0-9]|3[01])([ ,)]|$)/ ||
	    $0 ~ /[ ,(]X(1[6-9]|2[0-9]|3[01])([ ,)]|$)/ ||
	    $0 ~ /[ ,(]K[0-7]([ ,)]|$)/) {
		report("wide or EVEX register")
	}
}
END {
	printf("checked %d kernels, %d instructions\n", nfn, ninsn)
	if (nfn == 0) {
		print "no 128-bit kernel found: the name filter is stale" > "/dev/stderr"
		exit 2
	}
	if (bad > 0) {
		printf("FAIL: %d offending instructions\n", bad)
		exit 1
	}
	print "OK: the 128-bit kernels need AVX only"
}
' "$tmp/pkg.dis"
