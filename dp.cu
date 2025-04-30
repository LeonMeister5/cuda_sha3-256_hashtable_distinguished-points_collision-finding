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
#define TABLE_SIZE (1ULL << 28)
#define THREADS_PER_BLOCK 256
#define BLOCKS 128

__device__ unsigned long long d_total_hashes = 0;
__device__ unsigned long long d_total_inserts = 0;

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

__device__ uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ void fill_random_seed(uint8_t* seed_buf) {
    uint64_t tid = (blockIdx.x * blockDim.x + threadIdx.x)
                 ^ ((blockIdx.y * blockDim.y + threadIdx.y) * 0x9E3779B97F4A7C15ULL);

    uint64_t state = clock64() ^ tid;

    int bytes_remaining = N_BYTES;
    int offset = 0;
    while (bytes_remaining > 0) {
        uint64_t rnd = splitmix64(state);
        int chunk = bytes_remaining < 8 ? bytes_remaining : 8;
        memcpy(seed_buf + offset, &rnd, chunk);
        offset += chunk;
        bytes_remaining -= chunk;
    }
}

__device__ int device_memcmp_seed(const uint8_t* a, const uint8_t* b) {
    for (int i = 0; i < N_BYTES; ++i) {
        if (a[i] != b[i]) return a[i] - b[i];
    }
    return 0;
}

__global__ void dp_kernel(SimpleHashTable<N_BYTES, N_BYTES>::Entry* table, bool* found,
                          uint8_t* out_seed1, uint8_t* out_seed2, uint8_t* out_collision_dp) {
    SimpleHashTable<N_BYTES, N_BYTES> hashtable(table, TABLE_SIZE);

    uint8_t seed[N_BYTES];
    uint8_t current[N_BYTES];
    uint8_t hash[HASH_OUT_SIZE];
    uint8_t existing_seed[N_BYTES];

    while (!(*found)) {
        fill_random_seed(seed);
        memcpy(current, seed, N_BYTES);

        while (!(*found)) {
            compute_hash(current, hash);
            atomicAdd(&d_total_hashes, 1ULL);
            memcpy(current, hash, N_BYTES);

            if (is_distinguished(current)) {
                if (hashtable.find(current, existing_seed)) {
                    if (device_memcmp_seed(existing_seed, seed) != 0) {
                        memcpy(out_seed1, existing_seed, N_BYTES);
                        memcpy(out_seed2, seed, N_BYTES);
                        memcpy(out_collision_dp, current, N_BYTES);
                        *found = true;
                        return;
                    }
                } else {
                    hashtable.insert(current, seed);
                    atomicAdd(&d_total_inserts, 1ULL);
                }
                break;
            }
        }
    }
}

int main() {
    SimpleHashTable<N_BYTES, N_BYTES>::Entry* d_table;
    bool* d_found;
    uint8_t* d_seed1;
    uint8_t* d_seed2;
    uint8_t* d_collision_dp;

    cudaMalloc(&d_table, sizeof(SimpleHashTable<N_BYTES, N_BYTES>::Entry) * TABLE_SIZE);
    cudaMemset(d_table, 0xFF, sizeof(SimpleHashTable<N_BYTES, N_BYTES>::Entry) * TABLE_SIZE);

    cudaMalloc(&d_found, sizeof(bool));
    cudaMalloc(&d_seed1, N_BYTES);
    cudaMalloc(&d_seed2, N_BYTES);
    cudaMalloc(&d_collision_dp, N_BYTES);
    cudaMemset(d_found, 0, sizeof(bool));

    dp_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_table, d_found, d_seed1, d_seed2, d_collision_dp);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    unsigned long long h_total_hashes = 0;
    unsigned long long h_total_inserts = 0;
    cudaMemcpyFromSymbol(&h_total_hashes, d_total_hashes, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&h_total_inserts, d_total_inserts, sizeof(unsigned long long));

    bool h_found = false;
    uint8_t h_seed1[N_BYTES];
    uint8_t h_seed2[N_BYTES];
    uint8_t h_collision_dp[N_BYTES];
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    if (h_found) {
        cudaMemcpy(h_seed1, d_seed1, N_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_seed2, d_seed2, N_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_collision_dp, d_collision_dp, N_BYTES, cudaMemcpyDeviceToHost);

        printf("Collision found!\n");
        printf("Seed 1: "); for (int i = 0; i < N_BYTES; ++i) printf("%02x", h_seed1[i]); printf("\n");
        printf("Seed 2: "); for (int i = 0; i < N_BYTES; ++i) printf("%02x", h_seed2[i]); printf("\n");
        printf("Collision distinguished point (DP): "); for (int i = 0; i < N_BYTES; ++i) printf("%02x", h_collision_dp[i]); printf("\n");
    } else {
        printf("No collision found.\n");
    }

    if (h_total_inserts > 0) {
        double average_chain_length = (double)h_total_hashes / (double)h_total_inserts;
        printf("Total Distinguished Points (Chains): %llu\n", h_total_inserts);
        printf("Total SHA3 calls (Total steps): %llu\n", h_total_hashes);
        printf("Average chain length: %.2f steps per chain\n", average_chain_length);
    } else {
        printf("No distinguished points were inserted.\n");
    }

    cudaFree(d_table);
    cudaFree(d_found);
    cudaFree(d_seed1);
    cudaFree(d_seed2);
    cudaFree(d_collision_dp);
    return 0;
}
