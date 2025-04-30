#ifndef SHA3_CUH
#define SHA3_CUH

#define ROUNDS 24

__device__ static const uint64_t keccakf_rndc[ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ static const int keccakf_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ static const int keccakf_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16,
    8, 21, 24, 4, 15, 23, 19, 13,
    12, 2, 20, 14, 22, 9, 6, 1
};

__device__ inline uint64_t ROTL64(uint64_t x, int y) {
    return (x << y) | (x >> (64 - y));
}

__device__ inline uint64_t load64(const uint8_t *x) {
    uint64_t r = 0;
    for (int i = 0; i < 8; i++) {
        r |= ((uint64_t)x[i]) << (8 * i);
    }
    return r;
}

__device__ inline void store64(uint8_t *x, uint64_t u) {
    for (int i = 0; i < 8; i++) {
        x[i] = (u >> (8 * i)) & 0xFF;
    }
}

__device__ void keccakf(uint64_t st[25]) {
    uint64_t bc[5], t;
    for (int round = 0; round < ROUNDS; round++) {
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, keccakf_rotc[i]);
            t = bc[0];
        }
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= keccakf_rndc[round];
    }
}

__device__ void sha3_256(const uint8_t *input, size_t inlen, uint8_t *output) {
    uint64_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;

    size_t rate = 136;
    uint8_t temp[136] = {0};

    while (inlen >= rate) {
        for (int i = 0; i < 17; i++) {
            st[i] ^= load64(input + 8 * i);
        }
        keccakf(st);
        input += rate;
        inlen -= rate;
    }

    for (int i = 0; i < inlen; i++)
        temp[i] = input[i];
    temp[inlen] = 0x06;
    temp[rate - 1] |= 0x80;

    for (int i = 0; i < 17; i++)
        st[i] ^= load64(temp + 8 * i);

    keccakf(st);

    for (int i = 0; i < 4; i++)
        store64(output + 8 * i, st[i]);
}

#endif // SHA3_CUH
