request_expense_orchestration
===============================

A problem with the server is that gensteps can correspond to everything from a
few photons to multiple millions of photons and even billions in double muon
events.  So the server needs to somehow handle requests with very different
expenses.  What do you suggest ? Where to do this orchestration in the use of
FastApi based off of metadata with the requests perhaps ?  Also is FastAPI the
best for this ? Any alternatives.  Clearly when there is too much for the
server to handle need to tell clients to Retry-Later.




::

    from fastapi import Request, HTTPException, status

    async def limit_heavy_requests(request: Request):
        photon_count = int(request.headers.get("X-Opticks-Weight", 0))
        current_load = get_gpu_utilization() # Custom function to check your C++ state

        if photon_count > 1_000_000 and current_load > 0.8:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server overloaded with heavy simulation",
                headers={"Retry-After": "30"} # Tells client to wait 30 seconds
            )





FastAPI is excellent for the orchestration because it handles asynchronous I/O
(like waiting for the GPU) without blocking other clients. However, it is not a
task scheduler.

Alternatives for "Extreme" Load:

    Ray: If you are doing distributed scientific computing, Ray is the gold
standard. It allows you to decorate your C++ Opticks functions as "Actors" and
handles the resource-based scheduling (e.g., "only run this if 4GB of VRAM is
free") automatically.

    Celery / Redis Queue: For Tier 3 (billions of photons), you should return a
202 Accepted immediately with a Location header. The client then polls that URL
to see when the hits are ready. This prevents HTTP timeouts on simulations that
take minutes.


Alternatives to FastAPI
--------------------------

* (Anyscale) Ray
* Celery
* Redis
* RQ (Redis Queue)



Ray
----

* https://www.ray.io/

* https://github.com/ray-project/ray

* https://docs.ray.io/en/latest/ray-overview/examples.html


From Ray whitepaper
---------------------

* "Ray Core" looks most appropriate no AI specific "Ray AIR"



Semaphore
-----------

::

    import asyncio
    from fastapi import FastAPI, Request

    app = FastAPI()
    # Only allow 1 massive Opticks simulation at a time on the GPU
    gpu_semaphore = asyncio.Semaphore(1)

    @app.post("/simulate")
    async def simulate(request: Request):
        # The request will "hang" here until the GPU is free
        async with gpu_semaphore:
            # 1. Stream data into NumPy (Zero-Copy)
            # 2. Call your C++ Opticks code via nanobind
            # 3. Return hits
            return make_response(result_arr)



Three queues
--------------

::

    import asyncio
    from fastapi import FastAPI, Request

    app = FastAPI()

    # Server-side Resource Controller
    small_lane = asyncio.Semaphore(4)  # 4 small jobs at once
    med_lane   = asyncio.Semaphore(2)  # 2 medium jobs
    large_lane = asyncio.Semaphore(1)  # 1 heavy job (muons, etc.)

    @app.post("/simulate")
    async def simulate(request: Request):
        photon_count = int(request.headers.get("X-Opticks-Weight", 0))
        
        if photon_count < 100_000:
            lane = small_lane
        elif photon_count < 10_000_000:
            lane = med_lane
        else:
            lane = large_lane

        async with lane:
            # Run Opticks C++ simulation
            return make_response(result)




Future
--------


transitioning to a multi-GPU, multi-server architecture for Opticks moves you
from a "workstation-as-a-service" model to a true **distributed compute
cluster**. To keep the C++ libcurl client simple, the orchestrator must act as
a single, transparent entry point that hides the complexity of the backend.

Here is the architectural roadmap for that transition:

---

## 1. The "Single Entry Point" Orchestrator

You introduce a **Load Balancer / Gateway** (often FastAPI or Nginx) that
handles the incoming libcurl connections. This service does not do any
simulation; its only jobs are:

* **Authentication & Metadata Validation:** Checking the photon counts.
* **Health Checking:** Monitoring which GPU servers are alive.
* **Routing:** Forwarding the request to the specific server that has the "lane" available.

---

## 2. Distributed Resource Management (The Ray Path)

As mentioned before, **Ray** is the industry standard for this transition. In a
multi-server setup, Ray treats all GPUs across all servers as a single pool of
resources.

* **Head Node:** Runs the orchestrator/scheduler.
* **Worker Nodes:** Each server with one or more GPUs runs a Ray worker.
* **Scheduling:** When a request comes in for 100M photons, the orchestrator
asks Ray: *"Which node has 4GB of VRAM free?"* Ray handles the "plumbing" of
sending that task to the correct machine.

---

## 3. Communication Patterns: Forwarding vs. Redirection

You have two choices for how the data actually gets to the client:

### A. The Proxy Model (Simplest for Client)

The orchestrator receives the gensteps from the client, sends them to a GPU
worker, receives the hits back, and then sends them back to the client.

* **Pros:** Client only ever talks to one IP.
* **Cons:** The orchestrator becomes a bandwidth bottleneck (it handles all hit data twice).

### B. The Redirect Model (Most Scalable)

1. Client POSTs gensteps to the Orchestrator.
2. Orchestrator assigns the job to `GPU-Server-05`.
3. Orchestrator returns an **HTTP 307 (Temporary Redirect)** to the client.
4. The `libcurl` client (if `CURLOPT_FOLLOWLOCATION` is set) automatically reconnects to `GPU-Server-05` to finish the transaction.

* **Pros:** Hit data goes directly from the GPU server to the client.
* **Cons:** Every GPU server must be reachable by the client.

---

## 4. Key Infrastructure Components

| Component | Recommendation | Why? |
| --- | --- | --- |
| **Service Discovery** | **Consul** or **Ray Service** | Automatically registers new GPU servers as they come online. |
| **Shared Storage** | **S3 / MinIO** (Optional) | If your "meta" data gets huge, you might store results there and send a download link instead of raw bytes in the response. |
| **Containerization** | **Docker + NVIDIA Container Toolkit** | Vital for ensuring the Opticks/CUDA environment is identical across all servers in the cluster. |

---

## 5. Summary: Evolution of the "Lanes"

In a single-server setup, your "Lanes" are just Python semaphores. In a multi-server setup:

1. **Small Lane:** Can be distributed across all servers; any GPU can take these.
2. **Large Lane:** Might be restricted to specific "Fat Nodes" (e.g., servers with A100/H100 GPUs) that have the massive VRAM required for billion-photon events.

### The "Academic Budget" Approach

If you want to build this today without heavy infrastructure:

1. Use **FastAPI** as a simple reverse proxy.
2. Use **Ray Core** (Apache 2.0) to connect your workstations.
3. Stick with your **Header-based metadata** (X-Opticks-Weight) so the Ray scheduler can make intelligent decisions about which machine is "heavy" enough to handle the request.

This keeps your C++ client code **exactly the same**—it still just POSTs to a URL and waits for hits—while the backend scales from one GPU to a hundred.




