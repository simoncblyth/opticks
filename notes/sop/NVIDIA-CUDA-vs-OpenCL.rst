NVIDIA-CUDA-vs-OpenCL
======================

> On another note, are you aware of the situation with open-source GPU languages
> (OpenCL most prominently I guess) on NVIDIA hardware? Are they compatible
> nowadays? Any caveats?

I think the CUDA:OpenCL split for development and deployment is something like 9:1
So no sensible developer would willingly use OpenCL, they only do so if forced
- as it will make their development much harder.
GPU development is hard enough as it is, deliberately limiting yourself to
a small community and lesser performance and tooling makes no sense.

NVIDIA does not provide OpenCL versions of its acceleration libraries,
so OpenCL developers are forced to use generic open-source alternatives missing
out on countless hours of NVIDIA+community optimizations.  Also OpenCL developers
would not have access to dedicated hardware such as tensor cores or RT cores
for hardware ray tracing.


