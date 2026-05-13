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






Requests that would exceed allowable VRAM are handled by slicing the input
gensteps and doing multiple launches.  How would that fit into this pattern ?




If you already have logic to slice gensteps and perform multiple launches, this
fits perfectly into the **Tranche Pattern** as a "Virtual Lane." It allows you
to process a workload that is physically larger than the pre-allocated VRAM
"slot" by trading time for memory.

In this scenario, your `SLaunchContext` acts as a **sliding window**.

### 1. The "Sliding Window" Execution

Instead of trying to fit 10 million photons into a 1 million photon tranche,
you use the fixed buffer as a reusable workspace.

* **Step 1:** Upload the first slice of gensteps to the pre-allocated `d_photons` buffer.
* **Step 2:** Launch the OptiX propagation.
* **Step 3:** Download hits (or accumulate them on-device in a separate, perhaps larger, hits buffer).
* **Step 4:** Repeat until all slices are processed.

### 2. Implementation in the `SMgr` Pattern

You should update your `SLaunchContext` to handle the "Batching" logic
internally. The FastAPI layer doesn't need to know about the slicing; it just
knows it handed a large job to a specific lane.

```cpp
void SLaunchContext::process_large_job(const HostGensteps& all_gs) {
    size_t offset = 0;
    while (offset < all_gs.total_photons) {
        // 1. Calculate slice size based on this tranche's max_photons
        size_t slice_count = std::min(max_photons, all_gs.total_photons - offset);

        // 2. Upload slice to the PRE-ALLOCATED buffer
        cudaMemcpyAsync(d_photons, all_gs.data + offset, slice_count * sizeof(GS), stream);

        // 3. Launch OptiX on this stream
        optixLaunch(pipeline, stream, ..., d_photons, ...);

        // 4. Record event to ensure hits are ready before next upload or download
        cudaEventRecord(photons_done, stream);
        
        offset += slice_count;
    }
    // 5. Final Synchronize or download at the very end
    cudaStreamSynchronize(stream);
}

```

### 3. Impact on Server Resources

This "Slicing inside a Tranche" approach creates a very interesting performance
profile in your k6 reports:

* **VRAM Stability:** Your VRAM usage will be a "flat line." Even if a user
sends a gargantuan request, it never exceeds the tranche boundary because it’s
being processed in pieces.

* **Latency Linearity:** You will see `http_req_duration` scale linearly with
the number of slices. A job that requires 4 slices will take almost exactly 4x
longer than a job that fits in 1 slice.

* **The "Slow Lane" Effect:** If you only have 1 "Large" lane, a sliced job
will occupy that lane for a long time. This is where your **429: Too Many
Requests** logic is critical—it prevents other large jobs from queueing up and
timing out while the first one is busy slicing.

### 4. Refining the Lane Logic

To optimize this, you might decide that the "Small" lane never slices (it just
rejects anything too big), while the "Large" lane is the only one equipped with
the slicing logic.

| Lane | Pre-allocated VRAM | Slicing Behavior |
| --- | --- | --- |
| **Small** | 512 MB | **None.** Reject if job > 512MB. |
| **Large** | 4 GB | **Enabled.** If job is 12GB, do 3 passes. |

### 5. What k6 will show you

If you test this with k6, pay close attention to the **correlation between
`x-opticks-count` and `http_req_duration**`.

If slicing is working correctly, the graph of `Count vs. Duration` should be a
series of "Steps." Every time the count exceeds a multiple of your tranche
size, you'll see a jump in latency as the context moves from $N$ to $N+1$
launches.

This architecture is extremely robust because it treats VRAM as a **cached
resource** rather than a **dynamic allocation**, ensuring your server remains
responsive and never crashes due to an unexpectedly large input.




Regarding the 429s how to calculate the "Retry-After" delay.  Need an estimate
of current processing, but there will be other requests coming in too ?



Calculating a meaningful `Retry-After` value is a balancing act. If the delay
is too short, the client retries while the VRAM is still locked, wasting CPU
cycles and network bandwidth. If it’s too long, your GPU sits idle while
clients wait unnecessarily.

To get a smart estimate, you need to combine **deterministic** data (how long a
slice takes) with **probabilistic** data (how many slices are left in the
queue).

### 1. The "Cost Per Slice" Constant

Since you are using fixed VRAM tranches and serialized slicing, the time to
process one "chunk" is remarkably stable. You should benchmark your service to
find the average time ($T_{slice}$) for a full tranche propagation.

### 2. The Calculation Formula

The estimated wait time for a new request depends on how many jobs are already
ahead of it and how far along the *current* job is.

$$RetryAfter = (N_{queued\_jobs} \times T_{job\_avg}) + T_{remaining\_current\_slices}$$

However, since you are using **Semaphores** and **Lanes**, you can simplify
this logic based on the specific lane's state.

### 3. Implementation Strategy: The "Remaining Work" Counter

The most accurate way is to have your `SLaunchContext` or `SMgr` track its progress.

```python
# Pseudo-logic for the FastAPI Layer
@app.post("/simulate")
async def simulate(request: Request):
    count = int(request.headers.get("x-opticks-count", 0))
    lane, tranche_size = select_lane(count)
    
    if lane.locked():
        # 1. How many total slices does the currently running job have?
        # 2. How many has it finished?
        # You can store this in a global/shared dictionary updated by the C++ worker
        remaining_slices = SMgr.get_remaining_slices(lane_id)
        
        # Calculate delay: (Remaining Slices * Time per Slice) + Buffer
        # Example: 5 slices left * 0.2s per slice + 0.5s safety margin
        delay = (remaining_slices * T_SLICE) + 0.5
        
        return Response(
            status_code=429, 
            headers={"Retry-After": str(int(delay))}
        )

```

### 4. Accounting for "Incoming Traffic"

You mentioned that other requests are coming in too. In a standard 429
scenario, you **do not** account for requests that haven't arrived yet.

* **Why?** Because a 429 tells the client when the *current* congestion is
expected to clear. If you add time for "potential future requests," you
artificially throttle your throughput.

* **The "Thundering Herd":** If 10 clients receive `Retry-After: 5`, they will
all hit your server at exactly $T+5$.

* **The Fix:** Add **Jitter**. Instead of a flat estimate, return `delay +
random.uniform(0.5, 2.0)`. This spreads the retries out so they don't all slam
the semaphore at the same millisecond.



### 5. Summary of Factors for your Estimate

| Factor | Source | How to use it |
| --- | --- | --- |
| **Slices Remaining** | `SLaunchContext` | `num_slices - current_slice_index` |
| **Slice Velocity** | Benchmark | The average seconds per `optixLaunch` + `Memcpy`. |
| **Queue Depth** | Semaphore | If you allow a queue (instead of immediate 429), count the waiters. |
| **Overhead** | Fixed Constant | Add ~200ms for FastAPI/Python overhead and networking. |

### k6 Verification

When you implement this, use your k6 script to monitor the **Wait Time**.

* If your `Retry-After` is accurate, you should see the "Retried" requests in
k6 succeeding on their second attempt with very high frequency.

* If you see "Double 429s" (a retry that gets limited again), your estimate is
too aggressive.

* If your GPU utilization drops between requests, your estimate is too conservative.

**Pro-Tip:** If your C++ `SMgr` can't easily communicate "remaining slices"
back to Python, a simpler "exponential backoff" hint (e.g., 2s, 4s, 8s) is
often sufficient, but for Opticks-level precision, the "Remaining Slices"
method is the gold standard.







