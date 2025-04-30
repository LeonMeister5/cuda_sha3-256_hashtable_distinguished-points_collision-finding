// ht.cuh
#ifndef HT_CUH
#define HT_CUH

#include <cuda.h>
#include <stdint.h>

#define HT_EMPTY_MARKER 0xFF

template<int KeySize, int ValueSize>
class SimpleHashTable {
public:
    struct __align__(8) Entry { // <--- 强制8字节对齐，防止misaligned address
        uint8_t key[KeySize];
        uint8_t value[ValueSize];
    };

    __device__ SimpleHashTable(Entry* table, size_t capacity)
        : table_(table), capacity_(capacity) {}

    __device__ bool find(const uint8_t* key, uint8_t* value_out) {
        size_t h = hash(key);
        for (int i = 0; i < capacity_; ++i) {
            size_t idx = (h + i) % capacity_;
            Entry* entry = &table_[idx];
            if (is_empty(entry->key)) return false;
            if (equals(entry->key, key)) {
                for (int j = 0; j < ValueSize; ++j) value_out[j] = entry->value[j];
                return true;
            }
        }
        return false;
    }

    __device__ bool insert(const uint8_t* key, const uint8_t* value) {
        size_t h = hash(key);
        for (int i = 0; i < capacity_; ++i) {
            size_t idx = (h + i) % capacity_;
            Entry* entry = &table_[idx];

            uint64_t* p64 = reinterpret_cast<uint64_t*>(&entry->key[0]);
            if (atomicCAS(p64, 0xFFFFFFFFFFFFFFFFULL, *reinterpret_cast<const uint64_t*>(key)) == 0xFFFFFFFFFFFFFFFFULL) {
                for (int j = 8; j < KeySize; ++j) entry->key[j] = key[j];
                for (int j = 0; j < ValueSize; ++j) entry->value[j] = value[j];
                return true;
            }
            if (equals(entry->key, key)) {
                return false;
            }
        }
        return false;
    }

private:
    Entry* table_;
    size_t capacity_;

    __device__ bool is_empty(const uint8_t* key) const {
        for (int i = 0; i < KeySize; ++i) {
            if (key[i] != HT_EMPTY_MARKER) return false;
        }
        return true;
    }

    __device__ bool equals(const uint8_t* a, const uint8_t* b) const {
        for (int i = 0; i < KeySize; ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    __device__ size_t hash(const uint8_t* key) const {
        size_t h = 0xcbf29ce484222325ULL;
        for (int i = 0; i < KeySize; ++i) {
            h ^= key[i];
            h *= 0x100000001b3ULL;
        }
        return h % capacity_;
    }
};

#endif // HT_CUH
