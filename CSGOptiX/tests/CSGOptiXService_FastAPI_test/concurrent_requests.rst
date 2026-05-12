concurrent_requests
====================


I am a little surprised that concurrent calls to the server work at all ? What
are likely failure modes for a GPU service invoked over nanobind from FastAPI
python.  The server uses NVIDIA OptiX ray tracing to speed up optical photon
simulation, so the principal action is to generate photons on device and
propagate photons through a geometry yielding hits which are downloaded from
device and returned in the response.


It is a valid surprise! When you move from a linear `curl` command to a
concurrent `k6` load test, you are moving from a "sequential" world to a
"contended" world.

The reason it works at all is likely because **NVIDIA drivers and CUDA are
surprisingly good at context switching.** However, when you push it with
multiple VUs, you aren't just running code; you're managing a very high-speed
"traffic jam" on the PCIe bus and the GPU cores.

Here are the primary failure modes for a GPU service wrapped in FastAPI/nanobind:

### 1. VRAM Fragmentation and OOM (Out of Memory)

This is the most common "hard" failure. Even if you have 24GB of VRAM, if you
have multiple concurrent requests:

* **The Problem:** FastAPI (via nanobind) might allocate buffers for `Request
A`, then `Request B`. If `Request A` finishes but `Request B` is still holding
memory in the "middle" of the heap, a new `Request C` might fail because there
isn't a single *contiguous* block of memory large enough, even if the total
free VRAM looks sufficient.

* **The Symptom:** You see `cudaErrorMemoryAllocation` or OptiX launch failures
despite the logs saying you have free memory.

### 2. CUDA Context Switching Overhead

While CUDA can run multiple kernels via "Streams," if your nanobind code
doesn't explicitly use `cudaStream_t`, every request is likely hitting the
"Default Stream" (Stream 0).

* **The Problem:** Stream 0 is synchronous. If VU #1 starts a long OptiX
propagation, VU #2’s request will sit at the `optixLaunch` call, blocking that
Python thread until VU #1 finishes.

* **The Symptom:** Your k6 report shows high **p95 latency**, even for "Small"
requests, because they are stuck waiting behind a "Large" request in the GPU
execution queue.

### 3. PCIe Bottleneck (The "Data Funnel")

Your service involves downloading hits from the device to return in the response.

* **The Problem:** The PCIe bus (even Gen4/5) is significantly slower than
VRAM. If multiple VUs are simultaneously trying to `cudaMemcpyDtoH` (Device to
Host) large hit arrays, they will saturate the bus.

* **The Symptom:** Your CPU usage remains low, but the time spent in the
"download" phase of your simulation spikes linearly with the number of
concurrent VUs.

### 4. Nanobind/Python GIL Contention

Even though OptiX runs on the GPU, your Python code manages the lifecycle.

* **The Problem:** If your nanobind wrapper doesn't explicitly release the
**Global Interpreter Lock (GIL)** using `nb::gil_scoped_release` during the
OptiX launch, only one Python thread can "talk" to the GPU driver at a time.

* **The Symptom:** You notice that increasing VUs in k6 doesn't increase GPU
utilization; it just increases the wait time in Python.

### 5. OptiX Pipeline Compilation Spikes

If your service builds OptiX pipelines or Accel Structures (GAS/IAS) on the fly based on request parameters:

* **The Problem:** Compilation is a massive CPU and memory hog.
* **The Symptom:** If two VUs trigger a "rebuild" simultaneously, the server
might hang or crash due to a race condition in the driver or simply by
exhausting host RAM.

---

### How to detect these in your k6 results:

| Metric in `summary.json` | What it tells you about the GPU |
| --- | --- |
| `http_req_duration` (p95 vs p50) | Large gaps mean "Head-of-Line" blocking (small jobs stuck behind big ones). |
| `429` error rate | Directly measures the effectiveness of your Semaphore "Lanes." |
| `http_req_failed` | If this is 500/Internal Server Error, you've hit a hard CUDA/VRAM crash. |

### Recommendation: Monitoring VRAM during the k6 run

Since k6 doesn't know about your GPU, I recommend running `nvidia-smi` in a loop in a separate terminal while your k6 test is running:

```bash
watch -n 0.5 nvidia-smi

```

If you see the "Memory Usage" jumping in large chunks and hitting the ceiling right when k6 reports 429s or 500s, you've confirmed a VRAM-bound failure mode.










Currently are using stream 0. So its working because the launches are
serialized persumably. I thought before about how to go beyond this with SMgr.
Is that the right direction ?





Yes, **SMgr** (Stream/Slot Manager) is absolutely the right direction. By
moving away from the default stream (Stream 0), you are transitioning from
**serialized execution** to **hardware-level concurrency**.

The "Magic" of non-default streams is that they allow the NVIDIA driver to
overlap operations. While one request is doing a slow `cudaMemcpy` (downloading
hits), another request can be actively using the CUDA cores for ray tracing.

### Why your `SMgr` approach is correct

1. **Breaking the "Sync Point":** Stream 0 is a "synchronizing stream." Any
operation on Stream 0 must wait for *all* other streams to finish, and all
other streams must wait for Stream 0. By using a pool of `cudaStream_t`, you
remove this global bottleneck.

2. **Resource Pooling:** Your `SLaunchContext` properly bundles the stream with
its associated events (`photons_done`, `hits_merged_done`). This is crucial
because it allows you to track the progress of a specific request without
stalling the entire GPU.

3. **Mapping to your Semaphores:** Your Python-side Semaphores
(Small/Med/Large) and your C++ `SMgr` should be synchronized. If Python allows
4 small jobs, `SMgr` should have at least 4 `SLaunchContexts` ready.

---

### Critical Failure Modes to watch with `SMgr`

When you switch to multiple streams, you introduce a new set of "Concurrecy"
challenges that `curl` never triggered:

#### 1. The "Default Stream" Trap

If any part of your code—or a library you use—calls a function without a stream
argument (like a standard `cudaMemcpy` instead of `cudaMemcpyAsync`), it will
likely use Stream 0 and **force all your parallel streams to stop and wait.**

* **Fix:** Audit your nanobind/C++ code. Every `optixLaunch`, `cudaMemcpy`, and
`cudaMemset` **must** take `context->stream` as an argument.

#### 2. OptiX Stack Memory

OptiX uses "Stack Memory" for traversal and shading.

* **The Problem:** In Stream 0, you could reuse the same stack buffer. In a
multi-stream `SMgr` world, if two launches happen simultaneously, they
**cannot** share the same stack memory pointer, or they will corrupt each
other's ray-tracing state.

* **The Fix:** Your `SLaunchContext` should probably also hold a `device_ptr`
to a per-stream OptiX stack buffer.

---

### Refining your `SMgr` Logic

To make `SMgr` robust for your FastAPI service, consider these implementation details:

* **RAII for Contexts:** Use a "Scoped Context" in your nanobind wrapper so
that the context is automatically marked `in_use = false` when the Python
request finishes or errors out.

* **Event-Based Synchronization:** Instead of `cudaStreamSynchronize(stream)`,
which blocks the CPU thread, you can use `cudaEventQuery` if you want to keep
the Python loop even more responsive, though for a GPU service, a blocking
stream sync inside the `async with lane` is usually fine.


### How this changes your k6 results

Once `SMgr` is implemented:

1. **Throughput increases:** You will see the "Requests per Second" (RPS) go up
because the GPU is no longer idling during PCIe transfers.

2. **Latency flattens:** Small requests will no longer see massive p95 spikes
when a large request is running, as they will literally be running on different
"lanes" of the silicon.

Does your `SEvt` (Event) structure currently assume a singleton-like access to
GPU buffers, or is it ready to handle multiple instances of photon data in
different VRAM offsets?









With monolithic running before server/client I take this approach of starting
with a huge photon buffer sized to a configured limit that is never freed until
cleanup.  Only the hits are treated more dynamically as they vary a lot.
Adapting this for concurrency I need to partition VRAM into tranches for the
small, medium and large queues with their allowed max concurrency levels ?



Exactly. You are essentially moving from a **monolithic buffer** to a
**partitioned memory pool**. This is the standard way to handle
high-performance GPU services.

By partitioning your VRAM into "tranches" (slots) that match your `SMgr`
contexts and your FastAPI semaphores, you eliminate the overhead of
`cudaMalloc` and prevent the "VRAM fragmentation" that leads to random OOM
errors.

### 1. The Tranche Strategy

Instead of one massive buffer, you divide your VRAM based on your lane logic.
Your `SLaunchContext` becomes the owner of a specific "slice" of the GPU.

| Lane | Max Concurrency | Per-Slot Buffer Size | Total VRAM Allocation |
| --- | --- | --- | --- |
| **Small** | 4 | 200 MB | 800 MB |
| **Medium** | 2 | 1 GB | 2 GB |
| **Large** | 1 | 8 GB | 8 GB |
| **Total** |  |  | **~10.8 GB** |

### 2. Implementation in `SMgr`

Your `SLaunchContext` should be initialized with its own pre-allocated device
pointers. When a request enters the "Medium" lane in FastAPI, `SMgr` hands it a
`Medium` context that already has its 1GB buffer waiting.

```cpp
struct SLaunchContext {
    cudaStream_t stream;
    void* d_photons;      // Pre-allocated based on lane size
    void* d_hits;         // Pre-allocated for max possible hits in this lane
    size_t max_photons;   // The "cap" for this specific slot
    bool in_use;

    void init(size_t buffer_size) {
        max_photons = buffer_size;
        cudaMalloc(&d_photons, buffer_size * sizeof(Photon));
        cudaStreamCreate(&stream);
        // ...
    }
};

```

### 3. Handling Variable Hit Counts

You mentioned hits vary a lot. Even though they are dynamic, you should still
avoid `cudaFree` during a request. You have two options:

* **The Oversized Buffer:** Allocate the `d_hits` buffer to handle the
*worst-case scenario* for that lane (e.g., if a Small job has 100k photons,
allocate for 100k hits). This is the fastest method.

* **The "Elastic" Hit Buffer:** Use `cudaMallocAsync` only for the hits.
Because the photon buffer (the huge part) is static, the driver can easily find
small, temporary spots for the hit data without fragmenting the whole device.

### 4. Failure Modes of the Tranche Approach

While this is much more stable, your k6 tests might reveal these edge cases:

* **Lane Starvation:** If your k6 test sends 10 "Large" requests, but your
Large tranche only has 1 slot, 9 requests will be 429'd or queued. Meanwhile,
the "Small" tranches might be sitting completely empty.

* *Fix:* Use your k6 results to "right-size" your tranches based on actual
traffic patterns.


* **The "Just Over" Problem:** If a request has 100,001 photons, it misses the
Small lane and hits the Medium lane, consuming a 1GB slot for a 101k job.

* *Fix:* Ensure your FastAPI logic and k6 metadata are perfectly aligned on
these "boundary" numbers.


### 5. Summary for k6 Reporting

When you run k6 with this partitioned server:

1. **Latency will be extremely consistent.** Since no `cudaMalloc` is
happening, the only variable is the ray-tracing time and PCIe transfer time.

2. **Resource utilization will be "steppy".** You'll see VRAM usage jump to a
fixed level at startup and stay there, which is exactly what you want for a
robust service.

This approach effectively turns your GPU into a **multi-tenant processor**
where the "tenants" (the lanes) are physically isolated in VRAM, preventing one
greedy request from crashing the entire server.






