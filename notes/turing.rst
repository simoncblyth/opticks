Turing GPU Architecture
=============================

Turing Whitepaper
--------------------

* https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
* ~/opticks_refs/NVIDIA-Turing-Architecture-Whitepaper.pdf


p25

Real-time ray tracing in games and other applications is made possible by
incorporation of multiple new hardware-based ray tracing acceleration engines
called RT Cores in Turing TU102, TU104, and TU106 GPUs, combined with NVIDIA
RTX software technology.

* https://developer.nvidia.com/rtx
* https://developer.nvidia.com/rtx/raytracing

p27

Turing GPUs not only include dedicated ray tracing acceleration hardware, but
also use an advanced acceleration structure described in the next section.
Essentially, an entirely new rendering pipeline is available to enable
real-time ray tracing in games and other graphics applications using a single
Turing GPU (see Figure 17).


p30 TURING RT CORES

At the heart of Turing’s hardware-based ray tracing acceleration is the new RT
Core included in each SM. RT Cores accelerate Bounding Volume Hierarchy (BVH)
traversal and ray/triangle intersection testing (ray casting) functions. (See
Appendix D Ray Tracing Overview on page 68 for more details on how BVH
acceleration structures work). RT Cores perform visibility testing on behalf of
threads running in the SM.

To better understand the function of RT Cores, and what exactly they
accelerate, we should first explain how ray tracing is performed on GPUs or
CPUs without a dedicated hardware ray tracing engine. Essentially, the process
of BVH traversal would need to be performed by shader operations and take
thousands of instruction slots per ray cast to test against bounding box
intersections in the BVH until finally hitting a triangle and the color at the
point of intersection contributes to final pixel color (or if no triangle is
hit, background color may be used to shade a pixel).

Ray tracing without hardware acceleration requires thousands of software
instruction slots per ray to test successively smaller bounding boxes in the
BVH structure until possibly hitting a triangle. It’s a computationally
intensive process making it impossible to do on GPUs in real-time without
hardware-based ray tracing acceleration (see Figure 19).

The RT Cores in Turing can process all the BVH traversal and ray-triangle
intersection testing, saving the SM from spending the thousands of instruction
slots per ray, which could be an enormous amount of instructions for an entire
scene. **The RT Core includes two specialized units. The first unit does bounding
box tests, and the second unit does ray-triangle intersection tests. The SM
only has to launch a ray probe, and the RT core does the BVH traversal and
ray-triangle tests, and return a hit or no hit to the SM. The SM is largely
freed up to do other graphics or compute work**. See Figure 20 or an illustration
of Turing ray tracing with RT Cores.

p31 Fig19/Fig20 Turing RT CORE comparison to pre-Turing   


p79 RT Cores

* Hybrid rendering
* Denoising algorithms
* BVH algorithm

All of the optimizations above helped to improve the efficiency of ray tracing,
but not enough to make it close to real time. However, **once the BVH algorithm
became standard, the opportunity emerged to make a carefully crafted
accelerator that would make this operation dramatically more efficient. RT
cores are that accelerator, making our GPUs 10x faster on ray tracing** and
bringing ray tracing to real time graphics for the first time.


Anandtech Turing Deep Dive
-----------------------------

Bounding Volume Hierarchy - How Computers Test the World
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.anandtech.com/show/13282/nvidia-turing-architecture-deep-dive/3

Perhaps the biggest aspect of NVIDIA’s gamble on ray tracing is that
traditional GPUs just aren’t very good at the task. They’re fast at
rasterization and they’re even fast at parallel computing, however ray tracing
does not map very well to either of those computing paradigms. Instead NVIDIA
**has to add hardware dedicated to ray tracing, which means devoting die space
and power to hardware that cannot help with traditional rasterization**.

A big part of that hardware, in turn, will go into solving the most basic
problem of ray tracing: how do you figure out what a ray is intersecting with?
The most common solution to this problem is to store triangles in a data
structure that is well-suited for ray tracing. And this data structure is
called a Bounding Volume Hierarchy.


* https://www.anandtech.com/show/13282/nvidia-turing-architecture-deep-dive/5

  * https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf 
  * https://research.nvidia.com/publication/maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees
  * https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
  * https://users.aalto.fi/~ailat1/publications/aila2010hpg_paper.pdf


When BVH became a standard of sorts, NVIDIA was able to design a corresponding
fixed function hardware accelerator.


Unlike Tensor Cores, which are better seen as an FMA array alongside the FP and
INT cores, the RT Cores are more like a classic offloading IP block. Treated
very similar to texture units by the sub-cores, instructions bound for RT Cores
are routed out of sub-cores, which is later notified on completion. Upon
receiving a ray probe from the SM, the RT Core proceeds to autonomously
traverse the BVH and perform ray-intersection tests. This type of ‘traversal
and intersection’ fixed function raytracing accelerator is a well-known concept
and has had quite a few implementations over the years, as traversal and
intersection testing are two of the most computationally intensive tasks
involved. In comparison, traversing the BVH in shaders would require thousands
of instruction slots per ray cast, all for testing against bounding box
intersections in the BVH.

* https://pubweb.eng.utah.edu/~cs6958/papers/HWRT-seminar/a160-nah.pdf

* https://www.anandtech.com/Gallery/Album/6660#28

  > 10 Giga Rays

* https://images.anandtech.com/galleries/6660/NV_Turing_Editors_Day_132.png


Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs

* https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf




Anandtech : The NVIDIA GeForce RTX 2080 Ti & RTX 2080 Founders Edition Review: Foundations For A Ray Traced Future
---------------------------------------------------------------------------------------------------------------------

* https://www.anandtech.com/show/13346/the-nvidia-geforce-rtx-2080-ti-and-2080-founders-edition-review/3


Overall, NVIDIA’s grand vision for real-time, hybridized raytracing graphics
means that they needed to make significant architectural investments into
future GPUs. The very nature of the operations required for ray tracing means
that they don’t map to traditional SIMT execution especially well, and while
this doesn’t preclude GPU raytracing via traditional GPU compute, it does end
up doing so relatively inefficiently. Which means that **of the many
architectural changes in Turing, a lot of them have gone into solving the
raytracing problem – some of which exclusively so**.

To that end, on the ray tracing front Turing introduces two new kinds of
hardware units that were not present on its Pascal predecessor: RT cores and
Tensor cores. The former is pretty much exactly what the name says on the tin,
with **RT cores accelerating the process of tracing rays, and all the new
algorithms involved in that**. Meanwhile the tensor cores are technically not
related to the raytracing process itself, however they play a key part in
making raytracing rendering viable, along with powering some other features
being rolled out with the GeForce RTX series.

Starting with the RT cores, these are perhaps NVIDIA’s biggest innovation –
efficient raytracing is a legitimately hard problem – however for that reason
they’re also the piece of the puzzle that NVIDIA likes talking about the least.
The company isn’t being entirely mum, thankfully. But we really only have a
high level overview of what they do, with the secret sauce being very much
secret. How NVIDIA ever solved the coherence problems that dog normal
raytracing methods, they aren’t saying.

At a high level then, the RT cores can essentially be considered a
fixed-function block that is designed specifically to accelerate Bounding
Volume Hierarchy (BVH) searches. BVH is a tree-like structure used to store
polygon information for raytracing, and it’s used here because it’s an innately
efficient means of testing ray intersection. Specifically, by continuously
subdividing a scene through ever-smaller bounding boxes, it becomes possible to
identify the polygon(s) a ray intersects with in only a fraction of the time it
would take to otherwise test all polygons.

NVIDIA’s RT cores then implement a hyper-optimized version of this process.
What precisely that entails is NVIDIA’s secret sauce – in particular the how
NVIDIA came to determine the best BVH variation for hardware acceleration – but
in the end the RT cores are designed very specifically to accelerate this
process. The end product is a collection of two distinct hardware blocks that
constantly iterate through bounding box or polygon checks respectively to test
intersection, to the tune of billions of rays per second and many times that
number in individual tests. All told, NVIDIA claims that the fastest Turing
parts, based on the TU102 GPU, can handle upwards of **10 billion ray
intersections per second** (10 GigaRays/second), ten-times what Pascal can do if
it follows the same process using its shaders.


NVIDIA has not disclosed the size of an individual RT core, but they’re thought
to be rather large. Turing implements just one RT core per SM, which means that
even the massive TU102 GPU in the RTX 2080 Ti only has 72 of the units.
Furthermore because the RT cores are part of the SM, they’re tightly couple to
the SMs in terms of both performance and core counts. As NVIDIA scales down
Turing for smaller GPUs by using a smaller number of SMs, the number of RT
cores and resulting raytracing performance scale down with it as well. So
NVIDIA always maintains the same ratio of SM resources (though chip designs can
very elsewhere).


Worlds First Ray Tracing GPU  
-------------------------------

* https://bgr.com/2018/08/14/nvidia-quadro-rtx-8000-ray-tracing-gpu-release-date-price/
* http://www.moorinsightsstrategy.com/nvidia-doubles-down-on-ray-tracing-with-turing/


