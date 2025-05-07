#!/usr/bin/env bash
#
# transpose.sh: read a 2-D C array from stdin,
# transpose it, and dump it back out with dimensions swapped.

awk '
BEGIN {
    in_matrix = 0
    row = 0
    cols = 0
}

# Capture the declaration header (strip trailing "{")
NR == 1 {
    header = $0
    sub(/{[[:space:]]*$/, "", header)
}

# Detect the start of the “= {” line
/=\s*{/ {
    in_matrix = 1
    next
}

# Once inside the matrix…
in_matrix {
    # If we see the closing “};”, finish up
    if (/^\s*};/) {
        nrows = row
        ncols = cols

        # Swap the two macros in the header, e.g. [A][B] → [B][A]
        header2 = header
        if (match(header, /\[([^]]+)\]\[([^]]+)\]/, m)) {
            header2 = gensub(/\[[^]]+\]\[[^]]+\]/,
                             "[" m[2] "][" m[1] "]", 1, header)
        }

        print header2 "{"
        for (i = 1; i <= ncols; i++) {
            printf "    { "
            for (j = 1; j <= nrows; j++) {
                printf data[j][i] (j < nrows ? ", " : " ")
            }
            print "},"
        }
        print "};"
        exit
    }

    # Strip braces and skip empty lines
    line = $0
    gsub(/[{}]/, "", line)
    if (line ~ /^[[:space:]]*$/) next

    # New row
    row++
    n = split(line, arr, ",")

    # Parse each comma-separated value
    c = 0
    for (k = 1; k <= n; k++) {
        val = arr[k]
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
        if (val != "") {
            c++
            data[row][c] = val
        }
    }
    if (c > cols) cols = c
}
' 
