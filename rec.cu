#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include "sha3.cuh"
#include "ht.cuh"

#define N_BITS 80
#define M_BITS 24
#define PREFIX "G2406344B"
#define PREFIX_LEN 9
#define HASH_OUT_SIZE 32
#define N_BYTES (N_BITS / 8)
#define MAX_CHAIN_MULTIPLIER 4
#define TABLE_SIZE (1ULL << 28)

__device__ bool d_collision_found = false;
__device__ uint8_t d_seed1_prev[N_BYTES];
__device__ uint8_t d_seed2_prev[N_BYTES];
__device__ uint8_t d_collision_hash[N_BYTES];
__device__ unsigned long long d_chain1_steps = 0;
__device__ unsigned long long d_chain2_steps = 0;

__device__ bool is_distinguished(const uint8_t* hash) {
    int full_bytes = M_BITS / 8;
    int rem_bits = M_BITS % 8;
    for (int i = 0; i < full_bytes; ++i) {
        if (hash[i] != 0) return false;
    }
    if (rem_bits > 0) {
        uint8_t mask = 0xFF << (8 - rem_bits);
        if ((hash[full_bytes] & mask) != 0) return false;
    }
    return true;
}

__device__ bool match_prefix(const uint8_t* a, const uint8_t* b) {
    int full_bytes = N_BITS / 8;
    int rem_bits = N_BITS % 8;
    for (int i = 0; i < full_bytes; ++i) {
        if (a[i] != b[i]) return false;
    }
    if (rem_bits > 0) {
        uint8_t mask = 0xFF << (8 - rem_bits);
        if ((a[full_bytes] & mask) != (b[full_bytes] & mask)) return false;
    }
    return true;
}

__device__ void compute_hash(const uint8_t* input_seed, uint8_t* output) {
    uint8_t local_prefix[PREFIX_LEN];
    for (int i = 0; i < PREFIX_LEN; ++i) {
        local_prefix[i] = PREFIX[i];
    }
    uint8_t input[PREFIX_LEN + N_BYTES];
    memcpy(input, local_prefix, PREFIX_LEN);
    memcpy(input + PREFIX_LEN, input_seed, N_BYTES);
    sha3_256(input, PREFIX_LEN + N_BYTES, output);
}

__global__ void build_chain_kernel(SimpleHashTable<N_BYTES, N_BYTES>::Entry* table, const uint8_t* seed1) {
    SimpleHashTable<N_BYTES, N_BYTES> hashtable(table, TABLE_SIZE);

    uint8_t current[N_BYTES];
    uint8_t hash[N_BYTES];

    memcpy(current, seed1, N_BYTES);

    while (true) {
        compute_hash(current, hash);
        hashtable.insert(hash, current);
        atomicAdd(&d_chain1_steps, 1ULL);

        if (is_distinguished(hash)) {
            printf("Distinguished point found in Chain 1, stopping chain.\n");
            break;
        }

        memcpy(current, hash, N_BYTES);
    }
}

__global__ void search_chain_kernel(SimpleHashTable<N_BYTES, N_BYTES>::Entry* table, const uint8_t* seed2) {
    SimpleHashTable<N_BYTES, N_BYTES> hashtable(table, TABLE_SIZE);

    uint8_t current[N_BYTES];
    uint8_t hash[N_BYTES];
    uint8_t found_seed[N_BYTES];

    memcpy(current, seed2, N_BYTES);

    while (true) {
        if (d_collision_found) return;

        compute_hash(current, hash);

        if (hashtable.find(hash, found_seed)) {
            if (!match_prefix(found_seed, current)) { 
                memcpy(d_seed1_prev, found_seed, N_BYTES);
                memcpy(d_seed2_prev, current, N_BYTES);
                memcpy(d_collision_hash, hash, N_BYTES);
                d_collision_found = true;
                return;
            }
        }

        if (is_distinguished(hash)) {
            printf("Seed2 chain hit a Distinguished Point, no collision found, stopping.\n");
            return;
        }

        memcpy(current, hash, N_BYTES);
        atomicAdd(&d_chain2_steps, 1ULL);

        if (d_chain2_steps >= (1ULL << 30)) {
            printf("Chain 2 reached max steps, force exit.\n");
            return;
        }
    }
}

int main() {
    SimpleHashTable<N_BYTES, N_BYTES>::Entry* d_table;
    uint8_t* d_seed1;
    uint8_t* d_seed2;

    cudaMalloc(&d_table, sizeof(SimpleHashTable<N_BYTES, N_BYTES>::Entry) * TABLE_SIZE);
    cudaMemset(d_table, 0xFF, sizeof(SimpleHashTable<N_BYTES, N_BYTES>::Entry) * TABLE_SIZE);

    cudaMalloc(&d_seed1, N_BYTES);
    cudaMalloc(&d_seed2, N_BYTES);

    cudaMemset(&d_collision_found, 0, sizeof(bool));
    cudaMemset(&d_chain1_steps, 0, sizeof(unsigned long long));
    cudaMemset(&d_chain2_steps, 0, sizeof(unsigned long long));

    uint8_t h_seed1[N_BYTES] = { 0xc4, 0x5a, 0xef, 0x62, 0x5f, 0x46, 0x1b, 0x10, 0x0e, 0xce };
    uint8_t h_seed2[N_BYTES] = { 0x80, 0x32, 0x7d, 0x09, 0x6c, 0xb0, 0x85, 0xe3, 0x08, 0xd0 };

    cudaMemcpy(d_seed1, h_seed1, N_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed2, h_seed2, N_BYTES, cudaMemcpyHostToDevice);

    build_chain_kernel<<<1, 1>>>(d_table, d_seed1);
    cudaDeviceSynchronize();

    search_chain_kernel<<<1, 1>>>(d_table, d_seed2);
    cudaDeviceSynchronize();

    bool h_collision_found = false;
    unsigned long long h_chain1_steps = 0;
    unsigned long long h_chain2_steps = 0;

    cudaMemcpyFromSymbol(&h_collision_found, d_collision_found, sizeof(bool));
    cudaMemcpyFromSymbol(&h_chain1_steps, d_chain1_steps, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&h_chain2_steps, d_chain2_steps, sizeof(unsigned long long));

    if (h_collision_found) {
        uint8_t h_seed1_prev[N_BYTES];
        uint8_t h_seed2_prev[N_BYTES];
        uint8_t h_collision_hash[N_BYTES];

        cudaMemcpyFromSymbol(h_seed1_prev, d_seed1_prev, sizeof(h_seed1_prev));
        cudaMemcpyFromSymbol(h_seed2_prev, d_seed2_prev, sizeof(h_seed2_prev));
        cudaMemcpyFromSymbol(h_collision_hash, d_collision_hash, sizeof(h_collision_hash));

        printf("Collision found!\n");
        printf("Seed1 previous: ");
        for (int i = 0; i < N_BYTES; ++i) printf("%02x", h_seed1_prev[i]);
        printf("\n");
        printf("Seed2 previous: ");
        for (int i = 0; i < N_BYTES; ++i) printf("%02x", h_seed2_prev[i]);
        printf("\n");
        printf("Collision point (n-bit match): ");
        for (int i = 0; i < N_BYTES; ++i) printf("%02x", h_collision_hash[i]);
        printf("\n");
    } else {
        printf("No collision found.\n");
    }

    printf("Chain 1 total steps: %llu\n", h_chain1_steps);
    printf("Chain 2 steps to collision: %llu\n", h_chain2_steps);

    cudaFree(d_table);
    cudaFree(d_seed1);
    cudaFree(d_seed2);

    return 0;
}
