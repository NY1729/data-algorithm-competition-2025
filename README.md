# Explain

2025年度データ構造とアルゴリズムBでの優勝コードです。

## Problem

Given:

- A database of 1,000,000 strings (length 15, alphabet A–J)
- 1,000,000 query strings (same format)

For each query, output `1` if any string in the database has edit distance ≤ 3 from the query, otherwise `0`.
Index size was limited until 200 MB.

---

## Optimizations

### 1. Inverted Index + Hash Table (Candidate Filtering)

Rather than comparing every query against all 1M records, a multi-key inverted index is built at preprocessing time. For each query, only a small set of candidate records is retrieved using 10 character-pair keys with ±1 positional tolerance. This dramatically reduces the number of edit distance computations required.

### 2. AVX2 SIMD Bit-Parallel Edit Distance (16-way parallelism)

Edit distance computation uses the **bit-parallel Myers algorithm**, extended to process **16 strings simultaneously** using AVX2 256-bit SIMD intrinsics (`_mm256_*`). Each SIMD lane holds one candidate string, and all 16 are evaluated in a single pass over the query string.

### 3. Huge Pages (TLB Miss Reduction)

Database and index structures are allocated using 2MB huge pages via `mmap` with `MAP_HUGETLB`. This reduces TLB pressure significantly when traversing large arrays.

### 4. VByte Compression

The inverted index is stored in VByte-compressed format, reducing memory footprint and improving cache utilization during decoding.

---

## Performance

> Measured on a personal machine (WSL2, Ubuntu, AMD Ryzen 7 5700X / Zen 3, AVX2).

```
$ perf stat ./search query_file index_db > output

   898.94 msec task-clock
 3,586,451,456 cycles        (3.990 GHz)
 4,624,056,994 instructions  (1.29 IPC)
   388,477,736 branches
    11,396,620 branch-misses (2.93%)

0.899 seconds elapsed
```

Accuracy verified against reference output:

```
$ ./calc_hamming output reference
Total Bits(Chars):   1,000,000
Hamming Distance:    0
Accuracy:            100.000000 %
```

**1,000,000 queries × 1,000,000 records — solved in ~0.9s with 100% accuracy.**

---

## Files

```
.
├── search.c  # Query-time search (main binary)
├── prep.c    # Preprocessing: builds the inverted index from the database
└── README.md
```

---

## Build & Usage

```bash
# Build
gcc -o search -O2 search.c
gcc -o prep -O2 prep.c

# Step 1: Build index from database file
./prep database.txt > index_db

# Step 2: Search
./search query.txt index_db > output.txt
```
