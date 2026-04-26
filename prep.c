#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#define N 1000000
#define LEN 15
#define K 3
#define PAIRS 10
#define MAX_KEY_LEN6 1000000

long long CAPS[PAIRS] = {
    N * 4, // {0,12}
    N * 6, // {0,9}
    N * 6, // {3,12}
    N * 6, // {0,6}
    N * 9, // {3,9}
    N * 6, // {6,12}
    N * 9, // {3,6}
    N * 9, // {6,9}
    N * 6, // {9,12}
    N * 6  // {0,3}
};

uint64_t DataBase[N];

typedef struct
{
    uint32_t key;
    int db_id;
} IndexEntry;

IndexEntry *Indices[PAIRS];

static const int pair_offsets[PAIRS][2] = {
    {0, 12}, {0, 9}, {3, 12}, {0, 6}, {3, 9}, {6, 12}, {3, 6}, {6, 9}, {9, 12}, {0, 3}};

static inline uint16_t key(uint64_t s, int off)
{
    uint16_t c1 = (s >> (off * 4)) & 0xF;
    uint16_t c2 = (s >> ((off + 1) * 4)) & 0xF;
    uint16_t c3 = (s >> ((off + 2) * 4)) & 0xF;
    return (uint16_t)(c1 * 100 + c2 * 10 + c3);
}

static inline void write_vbyte(uint32_t x, FILE *fp)
{
    while (x >= 128)
    {
        fputc((x & 127) | 128, fp);
        x >>= 7;
    }
    fputc(x, fp);
}

static inline void import_file(const char *File_Name, uint64_t *Storage, int limit)
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

int compare_entries(const void *a, const void *b)
{
    const IndexEntry *ea = (const IndexEntry *)a;
    const IndexEntry *eb = (const IndexEntry *)b;
    if (ea->key < eb->key)
        return -1;
    if (ea->key > eb->key)
        return 1;
    if (ea->db_id < eb->db_id)
        return -1;
    if (ea->db_id > eb->db_id)
        return 1;
    return 0;
}

void init_indices()
{
    for (int p = 0; p < PAIRS; p++)
    {
        Indices[p] = malloc(sizeof(IndexEntry) * CAPS[p]);
        if (!Indices[p])
        {
            fprintf(stderr, "Memory allocation failed for pair %d\n", p);
            exit(1);
        }
    }
}

void build_indices(long long *ptrs)
{
    for (int i = 0; i < N; ++i)
    {
        uint16_t key3[13];
        for (int b = 0; b < 13; b++)
            key3[b] = key(DataBase[i], b);

        for (int p = 0; p < PAIRS; p++)
        {
            int a = pair_offsets[p][0];
            int b = pair_offsets[p][1];
            int shift[3] = {0, -1, 1};

            for (int s1 = 0; s1 < 3; s1++)
            {
                for (int s2 = 0; s2 < 3; s2++)
                {
                    int a1 = a + shift[s1];
                    int b1 = b + shift[s2];
                    if (a1 < 0 || a1 > 12 || b1 < 0 || b1 > 12)
                        continue;

                    uint32_t k = key3[a1] * 1000u + key3[b1];
                    Indices[p][ptrs[p]++] = (IndexEntry){k, i};
                }
            }
        }
    }
}

void sort_and_unique(long long *ptrs, long long *final_sizes)
{
    for (int p = 0; p < PAIRS; p++)
    {
        qsort(Indices[p], ptrs[p], sizeof(IndexEntry), compare_entries);
        long long unique = 0;
        if (ptrs[p] > 0)
        {
            Indices[p][unique++] = Indices[p][0];
            for (long long i = 1; i < ptrs[p]; ++i)
            {
                if (compare_entries(&Indices[p][i], &Indices[p][i - 1]) != 0)
                {
                    Indices[p][unique++] = Indices[p][i];
                }
            }
        }
        ptrs[p] = unique;
        final_sizes[p] = unique;
    }
}

void write_header(long long *final_sizes)
{
    fwrite(final_sizes, sizeof(long long), PAIRS, stdout);
}

void write_database()
{
    fwrite(DataBase, sizeof(uint64_t), N, stdout);
}

void write_hashtables(long long *ptrs)
{
    uint32_t *temp_HashTable = calloc(MAX_KEY_LEN6 + 1, sizeof(uint32_t));
    if (!temp_HashTable)
        exit(1);

    for (int p = 0; p < PAIRS; p++)
    {
        memset(temp_HashTable, 0, (MAX_KEY_LEN6 + 1) * sizeof(uint32_t));
        for (long long i = 0; i < ptrs[p]; ++i)
        {
            temp_HashTable[Indices[p][i].key + 1]++;
        }

        int prev_val = 0;
        for (int k = 1; k <= MAX_KEY_LEN6; ++k)
        {
            temp_HashTable[k] += temp_HashTable[k - 1];

            int val = temp_HashTable[k];
            write_vbyte(val - prev_val, stdout);
            prev_val = val;
        }
    }
    free(temp_HashTable);
}

void write_packed_ids(long long *ptrs)
{
    for (int p = 0; p < PAIRS; p++)
    {
        if (ptrs[p] == 0)
        {
            free(Indices[p]);
            continue;
        }

        uint64_t buffer = 0;
        int count = 0;

        for (long long i = 0; i < ptrs[p]; ++i)
        {
            uint32_t id = Indices[p][i].db_id;

            buffer |= ((uint64_t)id << (count * 20));
            count++;

            if (count == 3)
            {
                fwrite(&buffer, sizeof(uint64_t), 1, stdout);
                buffer = 0;
                count = 0;
            }
        }

        if (count > 0)
        {
            fwrite(&buffer, sizeof(uint64_t), 1, stdout);
        }

        free(Indices[p]);
    }
}

int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        return 1;
    }

    import_file(argv[1], DataBase, N);

    long long ptrs[PAIRS] = {0};
    long long final_sizes[PAIRS] = {0};

    init_indices();

    build_indices(ptrs);

    sort_and_unique(ptrs, final_sizes);

    write_header(final_sizes);
    write_database();
    write_hashtables(ptrs);
    write_packed_ids(ptrs);

    return 0;
}