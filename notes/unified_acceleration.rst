notes/unified_acceleration
============================

Overview
----------

Want a clean way to bring Opticks to lots of
different hardware with minimal duplication. 

Interested to learn from and perhaps use approaches
that expose ray tracing functionality in a unified 
way with multiple backends for different compute 
and ray tracing frameworks. 

Luisa-Render
--------------

* Development mostly from : BNRist, Department of CS&T, Tsinghua University, China

* SCB : interesting, as ROCm/HIP-RT could in principal be another backend to LuisaRender 
  which could bring HIP-RT to HYGON DCU 

  * thats using AMDs open source ray tracing framework HIP-RT on HYGON DCU and controlling
    it at a high level in a unified way with backends including CUDA-OptiX, Metal, DirectX, ISPC, and LLVM 



* https://github.com/LuisaGroup/LuisaRender
* https://github.com/LuisaGroup/LuisaCompute

* https://luisa-render.com/

The advancements in hardware have drawn more attention than ever to
high-quality offline rendering with modern stream processors, both in the
industry and in research fields. However, the graphics APIs are fragmented and
existing shading languages lack high-level constructs such as polymorphism,
which adds complexity to developing and maintaining cross-platform
high-performance renderers. We present LuisaRender, a high-performance
rendering framework for modern stream-architecture hardware. Our main
contribution is an expressive C++-embedded DSL for kernel programming with JIT
code generation and compilation. We also implement a unified runtime layer with
resource wrappers and an optimized Monte Carlo renderer. Experiments on test
scenes show that LuisaRender achieves much higher performance than existing
research renderers on modern graphics hardware. 


* https://luisa-render.com/static/paper/paper.pdf
* ~/opticks_refs/LuisaRender.pdf


::

    Behind the unified DSL and runtime, the backends realize the concrete resource
    management requests and computation tasks on different native APIs.
    Currently, we have implemented 5 backends for different platforms, namely CUDA,
    Metal, DirectX, ISPC, and LLVM.


gpustack
----------

* https://github.com/gpustack/gpustack

Supported Accelerators::

    Apple Metal (M-series chips)
    NVIDIA CUDA (Compute Capability 6.0 and above)
    AMD ROCm
    Ascend CANN
    Moore Threads MUSA
    Hygon DTK

We plan to support the following accelerators in future releases.::

    Intel oneAPI
    Qualcomm AI Engine



