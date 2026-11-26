Opticks_without_OptiX
======================

I would suggest to start with the lbvh package on github

* https://github.com/ToruNiina/lbvh

and related blog posts from Tero Karras

* https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/
* https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
* https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

Tero Karras, had a pivotal role in NVIDIA RTX hardware and software, he is
famous for GPU ray tracing - so his techniques are a good place to start with.

Try to make sense of what the code is doing.
You will need to learn CUDA Thrust, Morton Codes, ...

Then apply it to a very simple geometry, eg a collection of
non overlapping boxes.

Once you are familiar we can more formally consider what
kind of API you should be aiming for in order to replace OptiX
and reuse as much of Opticks as possible.


Below is an edited dialog with Gemini
--------------------------------------

SCB
----

For educational purposes a doctoral student is interested to investigate
implementing Opticks without using NVIDIA OptiX - ie manually creating
acceleration structures and traversal with CUDA. Which open source
projects would be a good starting point for his learning ?


Gemini
------

To investigate building a custom CUDA ray-tracing and acceleration-structure
pipeline without NVIDIA OptiX—and to compare those trade-offs against Opticks'
own architecture—the best open-source starting points fall into a few distinct
categories depending on what part of the pipeline the student wants to study
first.

1. Parallel GPU BVH Construction (Morton Codes & Radix Sort)

When building BVHs directly on the GPU rather than uploading a CPU-built tree,
Tero Karras's Linear BVH (LBVH) algorithm is the standard reference:

https://github.com/ToruNiina/lbvh


Another (might be too triangle-centric)

* nolmoonen/cuda-lbvh (GitHub)

A CMake-based CUDA LBVH implementation using
CUB radix sort alongside a basic path tracer on standard scenes (such as Sponza
and San Miguel), showing build times vs. traversal throughput.



SCB
---

Opticks provides analytic CSG intersection. The work would seek to reuse
Opticks intersection functionality and hence presumably try to replace the
OptiX API. So with that in mind I guess starting learning with BVH construction
would be best ?


Gemini
------

Yes—if the goal is to keep the analytic CSG intersection logic and replace the
OptiX API layer, BVH construction and traversal is the right place to start,
with one important nuance: in an analytic CSG setup, the BVH is built over
primitive/solid bounding boxes (AABBs) rather than triangle primitives, and a
BVH leaf or candidate hit hands control over to the analytic CSG intersection
evaluator.


Why Starting with BVH Construction Makes Sense Here

* Clean seam at AABBs and candidate hits: OptiX manages the acceleration
  structure (AS) build, top/bottom-level traversal, and calls custom intersection
  programs on candidate AABBs. Replacing OptiX means building the BVH in CUDA and
  writing a CUDA traversal loop that invokes the existing CSG intersection
  routines when a ray overlaps a leaf/primitive AABB.

* Tree quality governs how often CSG runs: Analytic CSG intersection is
  typically heavier per test than a triangle ray-box or ray-triangle check. A
  looser BVH or poor traversal order triggers more false AABB hits and more CSG
  evaluations, making BVH structure and traversal pruning immediately visible in
  performance.

* Independent of photon physics: A student can test BVH build time, tree depth,
  memory footprint, and ray-to-AABB traversal correctness on synthetic or
  exported AABBs before wiring up full optical photon transport or simulation
  loops.

  * SCB: Opticks optical physics can be used


What to Watch Out For When Adapting BVH Learning to Opticks

1. Primitives are AABBs, not triangles: Standard LBVH/BVH tutorials assume
   triangles with centroids and vertices. For analytic CSG, the BVH uses the AABBs
   of CSG leaves/solids (or instance transforms), so Morton code generation and
   bottom-up AABB reduction operate directly on primitive bounding boxes.

2. Traversal interface vs. OptiX program model: OptiX abstracts traversal
   behind traversable handles and program types (__raygen__, __intersection__,
   __closesthit__, __anyhit__). In pure CUDA, the student chooses between a single
   monolithic traversal-and-intersection kernel or decoupled stages/queues, and
   manages ray state, hit distance (t_hit / interval updates), and traversal
   stacks explicitly.

2. Instancing and CSG evaluation: Complex detector geometries often combine
   instanced assemblies with CSG boolean trees. Starting with a flat or two-level
   AABB BVH calling the CSG evaluator keeps the first prototype manageable before
   tackling multi-level instancing or CSG-specific optimizations.

   * SCB: two level is sufficient for JUNO geometry


