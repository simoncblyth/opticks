/**
SPM.cu
=======

Grok generated code under test

**/

#include "SPM.hh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/merge.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <vector>
#include <cassert>

using namespace thrust::placeholders;

// ------------------------------------------------------------------
// Helper functors
// ------------------------------------------------------------------
struct key_functor {
    float tw; unsigned select_mask;
    __device__ uint64_t operator()(const sphotonlite& p) const {
        if ((p.flagmask & select_mask) == 0) return ~0ull;
        unsigned id = p.identity() & 0xFFFFu;
        unsigned bucket = static_cast<unsigned>(p.time / tw);
        return (uint64_t(id) << 48) | uint64_t(bucket);
    }
};

struct reduce_op {
    __device__ sphotonlite operator()(const sphotonlite& a, const sphotonlite& b) const {
        sphotonlite r = a;
        r.time = fminf(a.time, b.time);
        r.flagmask |= b.flagmask;
        unsigned hc = a.hitcount() + b.hitcount();
        r.set_hitcount_identity(hc, a.identity());
        return r;
    }
};

struct select_pred {
    unsigned mask;
    __device__ bool operator()(const sphotonlite& p) const { return (p.flagmask & mask) != 0; }
};

// ------------------------------------------------------------------
// merge_partial_select
// ------------------------------------------------------------------
void SPM::merge_partial_select(
        const sphotonlite* d_in, int num_in, sphotonlite** d_out, int* num_out,
        unsigned select_flagmask, float time_window, cudaStream_t stream )
{
    if (num_in == 0) { *d_out = nullptr; if (num_out) *num_out = 0; return; }

    auto policy = thrust::cuda::par_nosync.on(stream);
    auto in = thrust::device_ptr<const sphotonlite>(d_in);

    if (time_window == 0.f) {
        select_pred pred{select_flagmask};
        int kept = thrust::count_if(policy, in, in + num_in, pred);
        sphotonlite* d_filtered = nullptr;
        if (kept > 0) {
            cudaMallocAsync(&d_filtered, kept * sizeof(sphotonlite), stream);
            thrust::copy_if(policy, in, in + num_in,
                            thrust::device_ptr<sphotonlite>(d_filtered), pred);
        }
        *d_out = d_filtered;
        if (num_out) *num_out = kept;
        return;
    }

    uint64_t* d_keys = nullptr; sphotonlite* d_vals = nullptr;
    cudaMallocAsync(&d_keys, num_in * sizeof(uint64_t),   stream);
    cudaMallocAsync(&d_vals, num_in * sizeof(sphotonlite), stream);

    auto keys_it = thrust::make_transform_iterator(in, key_functor{time_window, select_flagmask});
    thrust::copy(policy, keys_it, keys_it + num_in, thrust::device_ptr<uint64_t>(d_keys));
    thrust::copy_n(policy, in, num_in, thrust::device_ptr<sphotonlite>(d_vals));

    thrust::sort_by_key(policy,
                        thrust::device_ptr<uint64_t>(d_keys),
                        thrust::device_ptr<uint64_t>(d_keys + num_in),
                        thrust::device_ptr<sphotonlite>(d_vals));

    uint64_t* d_out_key = nullptr; sphotonlite* d_out_val = nullptr;
    cudaMallocAsync(&d_out_key, num_in * sizeof(uint64_t),   stream);
    cudaMallocAsync(&d_out_val, num_in * sizeof(sphotonlite), stream);

    auto ends = thrust::reduce_by_key(policy,
                thrust::device_ptr<uint64_t>(d_keys),
                thrust::device_ptr<uint64_t>(d_keys + num_in),
                thrust::device_ptr<sphotonlite>(d_vals),
                thrust::device_ptr<uint64_t>(d_out_key),
                thrust::device_ptr<sphotonlite>(d_out_val),
                thrust::equal_to<uint64_t>(),
                reduce_op());

    int merged = ends.first - thrust::device_ptr<uint64_t>(d_out_key);
    while (merged > 0) {
        uint64_t last;
        cudaMemcpyAsync(&last, d_out_key + merged - 1, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (last != ~0ull) break;
        --merged;
    }

    sphotonlite* d_final = nullptr;
    if (merged > 0) {
        cudaMallocAsync(&d_final, merged * sizeof(sphotonlite), stream);
        cudaMemcpyAsync(d_final, d_out_val, merged * sizeof(sphotonlite),
                        cudaMemcpyDeviceToDevice, stream);
    }

    cudaFreeAsync(d_keys, stream);
    cudaFreeAsync(d_vals, stream);
    cudaFreeAsync(d_out_key, stream);
    cudaFreeAsync(d_out_val, stream);

    *d_out = d_final;
    if (num_out) *num_out = merged;
}

// ------------------------------------------------------------------
// Async save_partial – pinned memory + correct callback
// ------------------------------------------------------------------
struct SavePartialData {
    sphotonlite* h_pinned;
    sphotonlite* d_ptr;
    std::string path;
    int count;
};

extern "C" void CUDART_CB save_partial_callback(cudaStream_t stream, cudaError_t status, void* userData)
{
    (void)stream;
    SavePartialData* data = static_cast<SavePartialData*>(userData);
    if (status != cudaSuccess) {
        fprintf(stderr, "save_partial_callback: CUDA error %d\n", status);
    }

    std::ofstream f(data->path, std::ios::binary);
    if (f) {
        f.write(reinterpret_cast<const char*>(data->h_pinned), data->count * sizeof(sphotonlite));
    } else {
        fprintf(stderr, "save_partial: failed to open %s\n", data->path.c_str());
    }

    cudaFree(data->d_ptr);
    cudaFreeHost(data->h_pinned);
    delete data;
}

void SPM::save_partial(
        const sphotonlite* d_partial,
        int                count,
        const std::string& path,
        cudaStream_t       stream )
{
    if (count == 0 || !d_partial) {
        cudaFree(const_cast<sphotonlite*>(d_partial));
        return;
    }

    sphotonlite* h_pinned = nullptr;
    cudaMallocHost(&h_pinned, count * sizeof(sphotonlite));

    cudaMemcpyAsync(h_pinned, d_partial, count * sizeof(sphotonlite),
                    cudaMemcpyDeviceToHost, stream);

    SavePartialData* cb_data = new SavePartialData{h_pinned, const_cast<sphotonlite*>(d_partial), path, count};
    cudaStreamAddCallback(stream, save_partial_callback, cb_data, 0);
}

// ------------------------------------------------------------------
// Async load_partial
// ------------------------------------------------------------------
void SPM::load_partial(
        const std::string& path,
        sphotonlite**      d_out,
        int*               count,
        cudaStream_t       stream )
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    assert(f && "failed to open partial");
    size_t bytes = f.tellg();
    size_t n = bytes / sizeof(sphotonlite);
    f.seekg(0);

    std::vector<sphotonlite> h(n);
    f.read((char*)h.data(), bytes);

    sphotonlite* d = nullptr;
    cudaMallocAsync(&d, n * sizeof(sphotonlite), stream);
    cudaMemcpyAsync(d, h.data(), n * sizeof(sphotonlite),
                    cudaMemcpyHostToDevice, stream);

    *d_out = d;
    if (count) *count = (int)n;
}

// ------------------------------------------------------------------
// merge_incremental – FULLY IMPLEMENTED
// ------------------------------------------------------------------
namespace {
    struct Partial {
        uint64_t*     keys;
        sphotonlite*  hits;
        int           count;
    };

    Partial load_partial_for_merge(const char* path, float time_window, cudaStream_t stream)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        assert(f && "failed to open partial");
        size_t bytes = f.tellg();
        size_t n = bytes / sizeof(sphotonlite);
        f.seekg(0);

        std::vector<sphotonlite> h(n);
        f.read((char*)h.data(), bytes);

        sphotonlite* d_hits = nullptr;
        uint64_t*     d_keys = nullptr;
        cudaMallocAsync(&d_hits, n * sizeof(sphotonlite), stream);
        cudaMallocAsync(&d_keys, n * sizeof(uint64_t),     stream);
        cudaMemcpyAsync(d_hits, h.data(), n * sizeof(sphotonlite),
                        cudaMemcpyHostToDevice, stream);

        auto policy = thrust::cuda::par_nosync.on(stream);

        if (time_window == 0.f) {
            // Use identity as unique key → no merging
            thrust::transform(policy,
                thrust::device_ptr<const sphotonlite>(d_hits),
                thrust::device_ptr<const sphotonlite>(d_hits + n),
                thrust::device_ptr<uint64_t>(d_keys),
                [] __device__ (const sphotonlite& p) -> uint64_t { return p.identity(); });
        } else {
            auto key_gen = [=] __device__ (const sphotonlite& p) -> uint64_t {
                unsigned id = p.identity() & 0xFFFFu;
                unsigned bk = unsigned(p.time / time_window);
                return (uint64_t(id) << 48) | bk;
            };
            thrust::transform(policy,
                thrust::device_ptr<const sphotonlite>(d_hits),
                thrust::device_ptr<const sphotonlite>(d_hits + n),
                thrust::device_ptr<uint64_t>(d_keys),
                key_gen);
        }

        thrust::sort_by_key(policy,
            thrust::device_ptr<uint64_t>(d_keys),
            thrust::device_ptr<uint64_t>(d_keys + n),
            thrust::device_ptr<sphotonlite>(d_hits));

        return {d_keys, d_hits, (int)n};
    }
}

void SPM::merge_incremental(
        const char** partial_paths,
        sphotonlite** d_final,
        int*          final_count,
        float         time_window,
        cudaStream_t  stream )
{
    if (!partial_paths[0]) {
        *d_final = nullptr;
        if (final_count) *final_count = 0;
        return;
    }

    auto policy = thrust::cuda::par_nosync.on(stream);

    Partial acc = load_partial_for_merge(partial_paths[0], time_window, stream);
    int i = 1;

    while (partial_paths[i] != nullptr) {
        Partial next = load_partial_for_merge(partial_paths[i], time_window, stream);
        ++i;

        size_t total = size_t(acc.count) + next.count;

        uint64_t*     d_merge_key = nullptr;
        sphotonlite*  d_merge_val = nullptr;
        cudaMallocAsync(&d_merge_key, total * sizeof(uint64_t),   stream);
        cudaMallocAsync(&d_merge_val, total * sizeof(sphotonlite), stream);

        auto merge_end = thrust::merge_by_key(
                policy,
                thrust::device_ptr<uint64_t>(acc.keys),   thrust::device_ptr<uint64_t>(acc.keys + acc.count),
                thrust::device_ptr<uint64_t>(next.keys),  thrust::device_ptr<uint64_t>(next.keys + next.count),
                thrust::device_ptr<sphotonlite>(acc.hits),
                thrust::device_ptr<sphotonlite>(next.hits),
                thrust::device_ptr<uint64_t>(d_merge_key),
                thrust::device_ptr<sphotonlite>(d_merge_val));

        size_t merged_size = merge_end.first - thrust::device_ptr<uint64_t>(d_merge_key);

        uint64_t*     d_reduced_key = nullptr;
        sphotonlite*  d_reduced_val = nullptr;
        int           reduced_count;

        if (time_window == 0.f) {
            // No reduction → just copy
            cudaMallocAsync(&d_reduced_key, merged_size * sizeof(uint64_t),   stream);
            cudaMallocAsync(&d_reduced_val, merged_size * sizeof(sphotonlite), stream);
            cudaMemcpyAsync(d_reduced_key, d_merge_key, merged_size * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(d_reduced_val, d_merge_val, merged_size * sizeof(sphotonlite),
                            cudaMemcpyDeviceToDevice, stream);
            reduced_count = merged_size;
        } else {
            cudaMallocAsync(&d_reduced_key, total * sizeof(uint64_t),   stream);
            cudaMallocAsync(&d_reduced_val, total * sizeof(sphotonlite), stream);

            auto reduce_end = thrust::reduce_by_key(
                    policy,
                    thrust::device_ptr<uint64_t>(d_merge_key),
                    thrust::device_ptr<uint64_t>(d_merge_key + merged_size),
                    thrust::device_ptr<sphotonlite>(d_merge_val),
                    thrust::device_ptr<uint64_t>(d_reduced_key),
                    thrust::device_ptr<sphotonlite>(d_reduced_val),
                    thrust::equal_to<uint64_t>(),
                    reduce_op());

            reduced_count = reduce_end.first - thrust::device_ptr<uint64_t>(d_reduced_key);
        }

        // Cleanup old
        cudaFreeAsync(acc.keys, stream);
        cudaFreeAsync(acc.hits, stream);
        cudaFreeAsync(d_merge_key, stream);
        cudaFreeAsync(d_merge_val, stream);
        cudaFreeAsync(next.keys, stream);
        cudaFreeAsync(next.hits, stream);

        acc.keys  = d_reduced_key;
        acc.hits  = d_reduced_val;
        acc.count = reduced_count;
    }

    *final_count = acc.count;
    *d_final     = acc.hits;
    cudaFreeAsync(acc.keys, stream);  // keys no longer needed
}
