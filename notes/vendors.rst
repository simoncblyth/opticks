vendors : Ray tracing framework vendors
==========================================





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

2. Apple Silicon, Metal Performance Shaders (MPS)

   * ray tracing API from Apple M1 (2020), 
   * hardware accelerated ray tracing from Apple M3-Pro (2023)
   * impractical for production use due to high cost, but many developers use Apple laptops  

3. Intel Xe2 GPU, Vulkan 

   * https://wccftech.com/intel-xe2-gpus-50-percent-uplift-new-ray-tracing-cores-lunar-lake-arc-battlemage-discrete/
   

4. Huawei, HarmonyOS NEXT, Ascend 910, Da Vinci GPU  

   * cloud ray trace rendering announced
   * https://www.huaweicentral.com/huawei-brings-harmonyos-next-cloud-rendering-for-realistic-gaming-experience/
   * https://www.nextplatform.com/2024/08/13/huaweis-hisilicon-can-compete-with-nvidia-gpus-in-china/


Interesting uses of GPU
-------------------------

Deepseek : PTX level use of NVIDIA GPUs

* https://medium.com/@amin32846/unlock-warp-level-performance-deepseeks-practical-techniques-for-specialized-gpu-tasks-a6cf0c68a178







