vendors : Ray tracing framework vendors
==========================================

Overview
---------

These notes are intended to help with keeping 
track of where the GPU vendors are at with regard to:

* compute frameworks
* ray tracing frameworks
* hardware accelerated ray tracing  


Summary
----------


.. class:: small 

    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+
    | GPU vendor    |  compute framework  | ray trace framework         | hardware RT                     | notes                                        |    
    +===============+=====================+=============================+=================================+==============================================+
    | NVIDIA        |   CUDA(2007-)       |  OptiX(2009-)               | RTX/RT Core (2018-)             |                                              |    
    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+
    | Apple         |   Metal/MPS(2014-)  |  Metal/MPS(2020-)           | From M3 (2023-)                 |                                              |    
    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+
    | AMD           |   ROCm(2016-)       | RadeonRays, HIP-RT(2022-)   | From Radeon RX 6000 (2020-)     |                                              |    
    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+
    | Intel         |   oneAPI(2020-)     |  Embree?                    | From Arc Alchemist (2022-)      | uses SYCL                                    |    
    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+
    | Huawei        |  ?                  |  mobile only                | mobile only                     |                                              |    
    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+
    | Cross-vendor  | Vulkan              | Vulkan                      | NVIDIA/AMD/Intel/?              |                                              |    
    |               | compute shaders     | ray trace extension         |                                 | Depends on vendor drivers                    |    
    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+
    | Cross-vendor  |  OpenCL             |                             |                                 |  dead?                                       |    
    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+
    | Cross-vendor  |  OpenMP             |                             |                                 |  new support for GPU offloading              |    
    +---------------+---------------------+-----------------------------+---------------------------------+----------------------------------------------+



Sources
---------


* https://analyticsindiamag.com/deep-tech/6-alternatives-to-cuda/

* https://en.wikipedia.org/wiki/Ray-tracing_hardware

* https://chipsandcheese.com/p/raytracing-on-amds-rdna-2-3-and-nvidias-turing-and-pascal

* https://www.modular.com/blog/democratizing-ai-compute-part-3-how-did-cuda-succeed


GPU vendors and progress with compute and ray tracing frameworks
-------------------------------------------------------------------

0. NVIDIA, Compute Unified Device Architecture (CUDA)

   * NVIDIA OptiX : industry leading ray tracing API, from 2009, API transition with OptiX 7.0 (2019) 
   * hardware accelerated ray tracing from NVIDIA RTX 1st generation GPUs (2018)

1. AMD, Radeon Open Compute (ROCm) 

   * Radeon Rays
   * https://www.techpowerup.com/324767/several-amd-rdna-4-architecture-ray-tracing-hardware-features-leaked
   * https://www.techspot.com/news/106208-amd-reveals-rdna4-architecture-radeon-rx-9070-gpus.html
   * https://www.techspot.com/news/106772-amd-unveil-rx-9000-series-february-28-leaked.html

   * https://gpuopen.com/learn/introducing-hiprt/

2. Apple Silicon, Metal Performance Shaders (MPS)

   * ray tracing API from Apple M1 (2020), 
   * hardware accelerated ray tracing from Apple M3-Pro (2023)
   * impractical for production use due to high cost, but many developers use Apple laptops  

3. Intel Xe2 GPU, Vulkan 

   * https://wccftech.com/intel-xe2-gpus-50-percent-uplift-new-ray-tracing-cores-lunar-lake-arc-battlemage-discrete/
   * https://www.intel.com/content/www/us/en/content-details/726653/a-quick-guide-to-intel-s-ray-tracing-hardware.html   

4. Huawei, HarmonyOS NEXT, Ascend 910, Da Vinci GPU  

   * cloud ray trace rendering announced
   * https://www.huaweicentral.com/huawei-brings-harmonyos-next-cloud-rendering-for-realistic-gaming-experience/
   * https://www.nextplatform.com/2024/08/13/huaweis-hisilicon-can-compete-with-nvidia-gpus-in-china/
   * https://en.wikipedia.org/wiki/OneAPI_(compute_acceleration)


5. Imagination Tech "IMG CXT GPU" : IP for mobile GPU with hardware ray tracing

   * https://www.imaginationtech.com/news/imagination-launches-the-most-advanced-ray-tracing-gpu/


6. Vulkan Compute Shaders

   * https://en.wikipedia.org/wiki/Vulkan
   * https://github.com/GPSnoopy/RayTracingInVulkan
   * https://vulkan.gpuinfo.org/
   * https://www.khronos.org/blog/ray-tracing-in-vulkan
   * https://github.com/KhronosGroup/MoltenVK/issues/427
   * https://forums.developer.nvidia.com/t/what-are-the-advantages-and-differences-between-optix-7-and-vulkan-raytracing-api/220360/2
   * https://www.khronos.org/blog/vulkan-ray-tracing-final-specification-release






NVIDIA : Blackwell RTX 6000 ? Potential for March Release 
------------------------------------------------------------

* https://www.tomshardware.com/pc-components/gpus/nvidias-rtx-blackwell-workstation-gpu-spotted-with-96gb-gddr7-proviz-gpu-with-a-512-bit-bus

One of the interesting peculiarities of the alleged RTX 6000 Blackwell
Generation graphics board is its 96GB of onboard GDDR7 memory. While memory
requirements for DCC and ProViz applications gradually increase, 96GB may be
overkill for graphics and computer-aided design workloads. However, 96GB of
onboard GDDR7 memory will be particularly useful for AI applications. Nvidia
will likely launch an AI-specific version of the RTX 6000 Blackwell board.

It is unclear when Nvidia plans to release its alleged RTX 6000 'Blackwell
Generation' professional graphics card, but it is natural to expect a high-end
DCC and ProViz solution to be released at the Game Developers Conference (GDC)
in the second half of March in San Jose, California. Alternatively, Nvidia
could introduce professional Blackwell-based graphics solutions at its own GPU
Technology Conference (GTC), which will take place from March 17 to March 21,
2025, but in San Francisco, California.


AMD : HIP-RT 
--------------

* https://gpuopen.com/learn/introducing-hiprt/
* https://gpuopen.com/download/publications/HIPRT-paper.pdf
* ~/opticks_refs/HIPRT-paper.pdf
* https://rocm.docs.amd.com/projects/HIP/en/docs-5.7.1/user_guide/faq.html

* https://www.phoronix.com/news/AMD-HIP-Ray-Tracing-RT-Open

End of Sec 2.1 of HIP-RT paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LuisaRender [Zheng et al. 2022] encapsulates the ray tracing functionality of
ray tracing engines behind a common API using an extended C++ and utilizing
OptiX or DirectX as backends. Similarly to the professional renderers mentioned
above, Rodent and LuisaRender could implement a HIPRT backend for AMD GPUs.
Mitsuba requires an intermediate representation (e.g., PTX), which is not
supported by HIP.

HIP-RT on DCU or NPU 
~~~~~~~~~~~~~~~~~~~~~~~

DCUs are built upon AMD’s open-source ROCm software stack and use
the HIP (Heterogeneous-Compute Interface for Portability)
programming model (AMD 2024). 

Blender/Cycles: HIP-RT for AMD hardware ray-tracing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* https://projects.blender.org/blender/blender/pulls/105538

HIPRT enables AMD hardware ray tracing on RDNA2 and above and fallbacks to
shader implementation for graphic cards that support HIP but not hardware ray
tracing.

The ray tracing feature functions are accessed through HIPRT SDK (available on
GPUOpen). HIPRT SDK allows the developer to build the bvh on the GPU with
different methods and trade-offs (fast, high quality, balanced). It also allows
for user defined primitives and custom intersection functions. The device side
of the SDK provides traversal functions. HIPRT traversal functionality is
pre-compiled in bitcode format and is shipped with HIPRT SDK. Blender kernels
are compiled with hiprt headers and then linked with hiprt bitcode that has the
implementation of traversal and intersection functions.

HIPRTDevice and HIPRTDeviceQueue, derived from HIP, implement new functions or
override existing functions to enable HIP ray tracing on the host side.

HIPRT offers an average improvement of 25% in sample rendering rate.




Huawei NPU (Neural Processing Unit)
-------------------------------------

* :google:`Huawei NPU`

* https://medium.com/huawei-developers/world-of-huawei-ascend-future-with-npus-5843c18993f3


HYGON DCU
----------

* :google:`HYGON DCU`

Optimizing depthwise separable convolution on DCU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://link.springer.com/article/10.1007/s42514-024-00200-3
* ~/opticks_refs/Using_Hygon_DCU_s42514-024-00200-3.pdf 

On the software side, GPUs utilize the CUDA (Compute Unified Device
Architecture) (Guide 2020) platform developed by NVIDIA, while DCUs are built
upon AMD’s open-source ROCm software stack and use the HIP
(Heterogeneous-Compute Interface for Portability) programming model (AMD 2024).
These differences present challenges when porting code from GPUs to DCUs, as we
need to re-implement code in the new programming model and consider the
hardware resources of the new device to optimize performance.


Sec 2.1
~~~~~~~~~

The DCU utilizes the AMD ROCm software stack, which
includes the HIP (Heterogeneous Interface for Portability)
C/C++ based programming model and runtime library
(AMD 2024).


:google:`HIP-RT on "DCU"`
--------------------------

* https://link.springer.com/content/pdf/10.1007/s11390-025-4285-7.pdf




Qualcomm ray tracing : Qualcomm Adreno GPU 
---------------------------------------------

* https://www.qualcomm.com/news/onq/2023/05/hardware-accelerated-ray-tracing-improves-lighting-effects-in-mobile-gaming

2023/3 
    the game War Thunder Mobile now incorporates hardware-accelerated ray tracing
    on a Snapdragon processor, using the Qualcomm Adreno GPU.
    On devices with the Snapdragon 8 Gen 2 mobile platform or higher, users see the
    more-realistic shadows shown on the left of the image above
    ...
    As the main rendering choice for mobile devices, the Vulkan specification uses
    a variety of extensions such as acceleration structures, ray tracing pipelines
    and ray queries. Instead of defining the visible area represented by triangles
    and shading them respectively, rays are generated for each visible pixel on
    screen. 


* https://github.com/SnapdragonStudios/adreno-gpu-vulkan-code-sample-framework

This repository contains a Vulkan Framework designed to enable developers to
get up and running quickly for creating sample content and rapid prototyping.
It is designed to be easy to build and have the basic building blocks needed
for creating an Android APK with Vulkan functionality, input system, as well as
other helper utilities for loading … 


Qualcomm Snapdragon XR2 Gen 2 (Pico 4 Ultra, Meta Quest 3)
-------------------------------------------------------------


Qualcomm Snapdragon XR2 Plus Gen 2 
-------------------------------------

* https://www.theverge.com/2024/1/4/24024480/qualcomm-snapdragon-xr2-plus-gen-2-vr-headset-chipset-samsung-google

XR2 Plus Gen 2 supports 4.3K resolution at 90fps per eye, which is a cut above
the XR2 Gen 2’s 3K-per-eye rendering. It also supports 12 concurrent cameras to
handle passthrough video as well as body and face tracking. Qualcomm says that
the new chipset offers a 15 percent increase in GPU frequency compared to the
standard XR2 Gen 2 and 20 percent greater CPU frequency — all in service of
“spatial computing in 4K.”


Vulkan Ray Tracing
---------------------

* https://www.khronos.org/blog/vulkan-ray-tracing-final-specification-release

2020 
    Khronos® has released the final versions of the set of Vulkan®, GLSL and SPIR-V
    extension specifications that seamlessly integrate ray tracing into the
    existing Vulkan framework. This is a significant milestone as it is the
    industry’s first open, cross-vendor, cross-platform standard for ray tracing
    acceleration - and can be deployed either using existing GPU compute or
    dedicated ray tracing cores. 


LumiBench
---------

* https://people.ece.ubc.ca/aamodt/publications/papers/lumibench.iiswc2023.pdf
* ~/opticks_refs/lumibench_iiswc2023.pdf


Interesting uses of GPU
-------------------------

Deepseek : PTX level use of NVIDIA GPUs

* https://medium.com/@amin32846/unlock-warp-level-performance-deepseeks-practical-techniques-for-specialized-gpu-tasks-a6cf0c68a178



