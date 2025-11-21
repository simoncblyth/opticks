async_launch_during_genstep_collection : "Incremental Async Launch"
======================================================================



Overview
---------

Developing CUDA Thrust based hitlite merging SPM.cu
made me take a look at async CUDA using returned future objects.
The simplicity of the immediately returned future objects that
have to be waited on makes async less daunting than juggling callbacks.

Originally, I had only a very vague idea of using async CUDA to
get overlap between multiple SEvt/QEvt. The vagueness is clear in SMgr.h

The amount of code change needed to do that, makes it seem unfeasible.
Plus anyhow I am sceptical of the benefits of event level overlapping
with very large events.

BUT, there are other ways to profit from overlapping, eg within one event.
Can do incremental async launches as the genstep are collected thats a lot
more tractable as it focusses on overlapping Geant4 with Opticks.

Whats missing:

* extending async to full pipeline
* dynamic concurrency control based on VRAM, gs photon counts


::

    Opticks currently collects so called gensteps (that parameterize optical photon
    generation) from a Geant4 simulation and then at the end of the Geant4 event
    performs the optical simulation and hit merging and then populates Geant4 hit
    collections with them. Clearly lauches could be done during the Geant4 event
    once enough gensteps to get to a certain photon count are collected. Any
    thoughts on what is optimum ?


Single threaded incremental launches
-------------------------------------

Thinking about incremental launches. Is the below reasonable:

To make this work, require that during the Geant4 simulation as each genstep is
collected the photons collected since the last clear are checked to see if
the configured launch minimum has been reached. If so the async pipeline is
invoked with the gensteps with a future object returned and collected
and the gensteps vector is cleared. Subsequently at end of event another async
invokation is done with the remainder gensteps yielding the final future
object. Then all those future objects are waited on and merging is done.

At first glance it does not seem that there is need of multiple CPU threads to do this ?


* BIG GORILLA : NEED TO MAKE FULL GS->HIT PIPELINE ASYNC


Grok-pseudo-code (std::move to avoid copies)::

    // One single Geant4 thread does everything
    class OpticksManager {
        std::vector<Future> pending_launches;
        std::vector<Genstep> current_batch;
        size_t photons_in_batch = 0;
        const size_t LAUNCH_THRESHOLD = 400'000'000;  // 400 M photons

        void collectGenstep(const Genstep& gs, size_t num_photons) {
            current_batch.push_back(gs);
            photons_in_batch += num_photons;

            if (photons_in_batch >= LAUNCH_THRESHOLD) {
                // Fire-and-forget async launch
                pending_launches.push_back( async_launch(std::move(current_batch)) );
                current_batch.clear();
                photons_in_batch = 0;
            }
        }

        void endOfEvent() {
            // Launch final remainder (even if small)
            if (!current_batch.empty()) {
                pending_launches.push_back( async_launch(std::move(current_batch)) );
                current_batch.clear();
            }

            // Now wait for all launches to finish
            for (auto& f : pending_launches)
                f.wait();                     // or cudaEventSynchronize

            // Merge all completed hit buffers
            merge_all(pending_launches);
            pending_launches.clear();
        }
    };



Look for high level discussion : async CUDA with future objects that must be waited on
----------------------------------------------------------------------------------------


* https://events.prace-ri.eu/event/989/sessions/3096/attachments/1196/2029/cuda-multiple-gpus.pdf
* https://medium.com/gpgpu/multi-gpu-programming-6768eeb42e2c
* https://www.sintef.no/contentassets/3eb4691190f2451fb21349eb24cb9e8e/part-3-multi-gpu-programming.pdf


