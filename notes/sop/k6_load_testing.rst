k6_load_testing
=================


* https://grafana.com/docs/k6/latest/set-up/install-k6/


Install k6
-----------

::

    sudo dnf install https://dl.k6.io/rpm/repo.rpm
    sudo dnf install k6



Start the server
----------------

::

    ~/j/bin/oj_server.sh


Create small/medium/large gensteps for testing
---------------------------------------------------------

::

   ~/o/sysrap/tests/SEvt__createInputGenstep_configuredTest_SML.sh


Run k6 load-test
------------------

::

    ~/np/tests/np_curl_test/np_curl_test.sh k6_post
    ~/np/tests/np_curl_test/np_curl_test.sh k6_load




Monitoring during tests
-------------------------


## 3. Monitoring Tools (The "Dashboard")

You cannot load test a GPU server effectively without watching the hardware.
While the load test is running, you should have these open in separate
terminals:

* **`nvidia-smi -l 1`**: Watch your VRAM usage. This is where you'll see if your "lanes" and "semaphores" are actually preventing OOMs.
* **`nload` or `iftop**`: Watch the network throughput. Since hit arrays are large, you might saturate your 1Gbps or 10Gbps link before you even max out the GPU.
* **`htop`**: To ensure the FastAPI/Uvicorn process isn't pinning a single CPU core while handling the async I/O.

---

## 4. Testing the "Muon" Problem (Soak Testing)

To test your "Billion Photon" events, you need a **Soak Test**. This involves
sending a heavy request and then immediately sending several tiny requests
while the heavy one is still processing.

**The Test Case:**

1. **Terminal 1:** `curl` a massive muon-level request (Large Lane).
2. **Terminal 2:** Immediately run `ab` with 10 small requests (Small Lane).
3. **The Goal:** The 10 small requests should finish *before* the massive muon request does. If the small ones are stuck waiting, your "lane" logic isn't partitioned correctly.

---

## 5. Metrics to Track

When the test finishes, don't just look at "Success/Failure." Look at:

* **P95 Latency:** How long 95% of requests take. If this spikes, your queue is too deep.
* **Throughput (req/sec):** How many photons/sec the server is actually processing.
* **Error Rate:** Specifically looking for `429` (Too Many Requests) or `504` (Gateway Timeout).

### Summary Recommendation

Start with **`ab`** for simple concurrency checks. If you plan on deploying
this as a production service, move to **`k6`** so you can automate a "Stress
Test" (ramping up users until it breaks) and a "Break Test" (sending a request
so large it tries to exceed VRAM).





