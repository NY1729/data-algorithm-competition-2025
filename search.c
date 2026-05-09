#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define MADV_RANDOM 1
#define MADV_SEQUENTIAL 2
#define MADV_HUGEPAGE 14

#define BATCH_SIZE 32
#define N 1000000
#define LEN 15
#define K 3
#define PAIRS 10
#define MAX_KEY_LEN6 1000000

uint64_t *DataBase;
uint64_t Query[N];

uint32_t *HashTables[PAIRS];
uint64_t *db_raw[PAIRS];

typedef struct
{
    int ub;
    int lb;
} range;

typedef struct
{
    __m256i lo;
    __m256i hi;
} PatternTable;

const int pair_offsets[PAIRS][2] = {
    {0, 12}, {0, 9}, {3, 12}, {0, 6}, {3, 9}, {6, 12}, {3, 6}, {6, 9}, {9, 12}, {0, 3}};

uint16_t key(uint64_t s, int off)
{
    uint16_t c1 = (s >> (off * 4)) & 0xF;
    uint16_t c2 = (s >> ((off + 1) * 4)) & 0xF;
    uint16_t c3 = (s >> ((off + 2) * 4)) & 0xF;
    return (uint16_t)(c1 * 100 + c2 * 10 + c3);
}

void import_file(const char *File_Name, uint64_t *Storage, int limit)
{
    FILE *fp = fopen(File_Name, "r");
    if (!fp)
    {
        perror(File_Name);
        exit(1);
    }
    char buffer[256];
    int i = 0;
    while (i < limit && fgets(buffer, sizeof(buffer), fp))
    {
        uint64_t packed = 0;
        for (int j = 0; j < 15; j++)
        {
            uint64_t val = buffer[j] - 'A';
            packed |= (val << (j * 4));
        }
        Storage[i] = packed;
        i++;
    }
    fclose(fp);
}

uint32_t read_vbyte(const uint8_t **ptr)
{
    uint32_t x = 0;
    int shift = 0;
    while (true)
    {
        uint8_t b = **ptr;
        (*ptr)++;
        x |= ((uint32_t)(b & 127)) << shift;
        if (!(b & 128))
            break;
        shift += 7;
    }
    return x;
}

__attribute__((target("avx2"))) PatternTable build_PatternTable(const uint16_t *pattern_mask)
{

    __m256i TBL_LO = _mm256_setr_epi8(
        pattern_mask[0] & 0xFF, pattern_mask[1] & 0xFF, pattern_mask[2] & 0xFF, pattern_mask[3] & 0xFF,
        pattern_mask[4] & 0xFF, pattern_mask[5] & 0xFF, pattern_mask[6] & 0xFF, pattern_mask[7] & 0xFF,
        pattern_mask[8] & 0xFF, pattern_mask[9] & 0xFF, pattern_mask[10] & 0xFF, pattern_mask[11] & 0xFF,
        pattern_mask[12] & 0xFF, pattern_mask[13] & 0xFF, pattern_mask[14] & 0xFF, pattern_mask[15] & 0xFF,
        pattern_mask[0] & 0xFF, pattern_mask[1] & 0xFF, pattern_mask[2] & 0xFF, pattern_mask[3] & 0xFF,
        pattern_mask[4] & 0xFF, pattern_mask[5] & 0xFF, pattern_mask[6] & 0xFF, pattern_mask[7] & 0xFF,
        pattern_mask[8] & 0xFF, pattern_mask[9] & 0xFF, pattern_mask[10] & 0xFF, pattern_mask[11] & 0xFF,
        pattern_mask[12] & 0xFF, pattern_mask[13] & 0xFF, pattern_mask[14] & 0xFF, pattern_mask[15] & 0xFF);

    __m256i TBL_HI = _mm256_setr_epi8(
        pattern_mask[0] >> 8, pattern_mask[1] >> 8, pattern_mask[2] >> 8, pattern_mask[3] >> 8,
        pattern_mask[4] >> 8, pattern_mask[5] >> 8, pattern_mask[6] >> 8, pattern_mask[7] >> 8,
        pattern_mask[8] >> 8, pattern_mask[9] >> 8, pattern_mask[10] >> 8, pattern_mask[11] >> 8,
        pattern_mask[12] >> 8, pattern_mask[13] >> 8, pattern_mask[14] >> 8, pattern_mask[15] >> 8,
        pattern_mask[0] >> 8, pattern_mask[1] >> 8, pattern_mask[2] >> 8, pattern_mask[3] >> 8,
        pattern_mask[4] >> 8, pattern_mask[5] >> 8, pattern_mask[6] >> 8, pattern_mask[7] >> 8,
        pattern_mask[8] >> 8, pattern_mask[9] >> 8, pattern_mask[10] >> 8, pattern_mask[11] >> 8,
        pattern_mask[12] >> 8, pattern_mask[13] >> 8, pattern_mask[14] >> 8, pattern_mask[15] >> 8);
    PatternTable pt = {TBL_LO, TBL_HI};
    return pt;
}

__attribute__((target("avx2"))) bool
bit_parallel_x16(const PatternTable *TBL, uint64_t *restrict B)
{
    const __m256i ONE = _mm256_set1_epi16(1);
    const __m256i ALL1 = _mm256_set1_epi16(0xFFFF);
    const __m256i Kp1 = _mm256_set1_epi16(K + 1);
    const __m256i MASK_0F = _mm256_set1_epi8(0x0F);

    const __m256i gather_mask = _mm256_setr_epi8(
        0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    const __m128i mask_lo = _mm_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

    const __m128i mask_hi = _mm_setr_epi8(
        4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11);

    const __m256i reorder_mask = _mm256_set_m128i(mask_hi, mask_lo);

    __m256i VP = ALL1;
    __m256i VN = _mm256_setzero_si256();
    __m256i D = _mm256_set1_epi16(LEN);

    __m256i v0 = _mm256_loadu_si256((const __m256i *)&B[0]);
    __m256i v1 = _mm256_loadu_si256((const __m256i *)&B[4]);
    __m256i v2 = _mm256_loadu_si256((const __m256i *)&B[8]);
    __m256i v3 = _mm256_loadu_si256((const __m256i *)&B[12]);

#pragma unroll LEN
    for (int i = 0; i < LEN; i++)
    {

        __m256i t0 = _mm256_shuffle_epi8(v0, gather_mask);
        __m256i t1 = _mm256_shuffle_epi8(v1, gather_mask);
        __m256i t2 = _mm256_shuffle_epi8(v2, gather_mask);
        __m256i t3 = _mm256_shuffle_epi8(v3, gather_mask);

        __m256i m01 = _mm256_or_si256(t0, _mm256_slli_si256(t1, 2));
        __m256i m23 = _mm256_or_si256(t2, _mm256_slli_si256(t3, 2));

        __m256i m = _mm256_or_si256(m01, _mm256_slli_si256(m23, 4));

        __m256i m_perm = _mm256_permute4x64_epi64(m, 0x88);

        __m256i CHARS = _mm256_shuffle_epi8(m_perm, reorder_mask);

        CHARS = _mm256_and_si256(CHARS, MASK_0F);

        __m256i lo = _mm256_shuffle_epi8(TBL->lo, CHARS);
        __m256i hi = _mm256_shuffle_epi8(TBL->hi, CHARS);
        __m256i PM = _mm256_unpacklo_epi8(lo, hi);

        __m256i D0 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_xor_si256(_mm256_add_epi16(_mm256_and_si256(PM, VP), VP), VP),
                PM),
            VN);

        __m256i HP = _mm256_or_si256(VN, _mm256_andnot_si256(_mm256_or_si256(D0, VP), ALL1));
        __m256i HN = _mm256_and_si256(D0, VP);

        D = _mm256_add_epi16(D, _mm256_and_si256(_mm256_srli_epi16(HP, LEN - 1), ONE));
        D = _mm256_sub_epi16(D, _mm256_and_si256(_mm256_srli_epi16(HN, LEN - 1), ONE));

        VP = _mm256_or_si256(
            _mm256_slli_epi16(HN, 1),
            _mm256_andnot_si256(
                _mm256_or_si256(D0, _mm256_or_si256(_mm256_slli_epi16(HP, 1), ONE)),
                ALL1));
        VN = _mm256_and_si256(D0, _mm256_or_si256(_mm256_slli_epi16(HP, 1), ONE));

        v0 = _mm256_srli_epi64(v0, 4);
        v1 = _mm256_srli_epi64(v1, 4);
        v2 = _mm256_srli_epi64(v2, 4);
        v3 = _mm256_srli_epi64(v3, 4);
    }

    __m256i mask = _mm256_cmpgt_epi16(Kp1, D);
    return !_mm256_testz_si256(mask, mask);
}

void *allocate_huge(size_t size)
{

    void *ptr;

    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);

    if (ptr != MAP_FAILED)
    {
        return ptr;
    }

    size_t alignment = 2 * 1024 * 1024;

    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    if (posix_memalign(&ptr, alignment, aligned_size) != 0)
    {
        perror("posix_memalign failed");
        exit(1);
    }

    madvise(ptr, aligned_size, MADV_HUGEPAGE);
    memset(ptr, 0, aligned_size);

    return ptr;
}

uint8_t *map_index_file(const char *filename, size_t *file_size_out)
{
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        perror(filename);
        exit(1);
    }

    struct stat sb;
    fstat(fd, &sb);
    *file_size_out = sb.st_size;

    uint8_t *map_base = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (map_base == MAP_FAILED)
    {
        perror("mmap");
        exit(1);
    }

    close(fd);
    return map_base;
}

void allocate_memory(const long long *sizes, const uint8_t **ptr)
{
    size_t total_size = 0;

    total_size += PAIRS * (MAX_KEY_LEN6 + 1) * sizeof(uint32_t);
    for (int p = 0; p < PAIRS; p++)
    {
        total_size += (size_t)(sizes[p] + BATCH_SIZE) * sizeof(uint64_t);
    }

    void *mem_block = allocate_huge(total_size);
    madvise(mem_block, total_size, MADV_SEQUENTIAL);
    const uint8_t *cursor = (const uint8_t *)mem_block;
    *ptr = cursor;
}

void init_database_ptr(uint8_t *map_base, long long **sizes_out, const uint8_t **ptr_out)
{
    *sizes_out = (long long *)map_base;

    DataBase = (uint64_t *)(map_base + sizeof(long long) * PAIRS);
    madvise(DataBase, sizeof(uint64_t) * N, MADV_SEQUENTIAL);

    size_t header_offset = sizeof(long long) * PAIRS + (sizeof(uint64_t) * N);
    *ptr_out = map_base + header_offset;
}

void decode_hash_tables(const uint8_t **ptr, const uint8_t **memory_ptr)
{
    const uint8_t *cursor = *ptr;

    uint32_t *block_ptr = (uint32_t *)*memory_ptr;

    for (int p = 0; p < PAIRS; p++)
    {
        HashTables[p] = block_ptr;
        block_ptr += (MAX_KEY_LEN6 + 1);
        HashTables[p][0] = 0;
        int prev_val = 0;
        for (int k = 1; k <= MAX_KEY_LEN6; ++k)
        {
            int diff = read_vbyte(&cursor);
            int val = prev_val + diff;
            HashTables[p][k] = val;
            prev_val = val;
        }
    }
    *ptr = cursor;

    *memory_ptr = (const uint8_t *)block_ptr;
}

void decode_db_ids(const long long *sizes, const uint8_t **ptr, const uint8_t **memory_ptr)
{
    const uint8_t *cursor = *ptr;

    size_t total_count = 0;
    for (int p = 0; p < PAIRS; p++)
    {
        total_count += (size_t)(sizes[p] + BATCH_SIZE);
    }

    uint64_t *block_ptr = (uint64_t *)*memory_ptr;

    const uint64_t mask = 0xFFFFF;

    for (int p = 0; p < PAIRS; p++)
    {
        uint64_t *ids = block_ptr;
        db_raw[p] = ids;

        long long count = sizes[p];
        long long i = 0;

        while (i < count)
        {

            uint64_t packed = *(const uint64_t *)cursor;
            cursor += 8;

            ids[i++] = DataBase[(uint32_t)(packed & mask)];
            if (i >= count)
                break;

            ids[i++] = DataBase[(uint32_t)((packed >> 20) & mask)];
            if (i >= count)
                break;

            ids[i++] = DataBase[(uint32_t)((packed >> 40) & mask)];
        }

        memset(ids + count, 0, BATCH_SIZE * sizeof(uint64_t));
        block_ptr += (count + BATCH_SIZE);
    }

    *ptr = cursor;
}

void preprocess_query(uint64_t query, uint16_t *keys_out, PatternTable *pt_out)
{
    for (int i = 0; i < 13; i++)
        keys_out[i] = key(query, i);

    uint16_t pattern_mask[16] = {0};
    for (int i = 0; i < LEN; i++)
        pattern_mask[(uint8_t)(query >> (i * 4)) & 0xF] |= (1u << i);

    *pt_out = build_PatternTable(pattern_mask);
}

void fetch_candidate_ranges(const uint16_t *keys, range *ranges_out)
{
    for (int i = 0; i < PAIRS; i++)
    {
        int a = pair_offsets[i][0];
        int b = pair_offsets[i][1];
        uint32_t k_val = keys[a] * 1000u + keys[b];
        ranges_out[i].lb = HashTables[i][k_val];
        ranges_out[i].ub = HashTables[i][k_val + 1];
    }
}

bool scan_candidates(const range *ranges, const PatternTable *pt)
{
    uint64_t batches[BATCH_SIZE];
    int batch_count = 0;

#pragma unroll PAIRS
    for (int i = 0; i < PAIRS; i++)
    {
        uint64_t *ids = db_raw[i];
        for (int idx = ranges[i].lb; idx < ranges[i].ub; idx++)
        {

            batches[batch_count++] = ids[idx];

            if (batch_count == BATCH_SIZE)
            {
                for (int offset = 0; offset < BATCH_SIZE; offset += 16)
                {
                    if (bit_parallel_x16(pt, batches + offset))
                        return true;
                }
                batch_count = 0;
            }
        }
    }

    if (batch_count > 0)
    {
        while (batch_count % 16 != 0)
            batches[batch_count++] = ~0ULL;
        for (int offset = 0; offset < batch_count; offset += 16)
        {
            if (bit_parallel_x16(pt, batches + offset))
                return true;
        }
    }

    return false;
}

__attribute__((target("avx2"))) void process_queries()
{
    char *results = (char *)malloc(N);

    for (int q = 0; q < N; q++)
    {
        uint16_t keys[13];
        PatternTable pt;
        range ranges[PAIRS];

        preprocess_query(Query[q], keys, &pt);
        fetch_candidate_ranges(keys, ranges);

        bool found = scan_candidates(ranges, &pt);

        results[q] = found + '0';
    }

    fwrite(results, 1, N, stdout);
}

int main(int argc, char const *argv[])
{
    if (argc < 3)
        return 1;

    import_file(argv[1], Query, N);

    size_t file_size;
    uint8_t *map_base = map_index_file(argv[2], &file_size);

    long long *sizes;
    const uint8_t *ptr;
    const uint8_t *memory_ptr;

    init_database_ptr(map_base, &sizes, &ptr);

    allocate_memory(sizes, &memory_ptr);

    decode_hash_tables(&ptr, &memory_ptr);
    decode_db_ids(sizes, &ptr, &memory_ptr);
    process_queries();

    return 0;
}