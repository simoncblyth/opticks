
What determines Opticks performance ? 
=======================================


> The results are encouraging, as using Opticks is about 40 times faster. 
> However, I recall that other users have reported speed factors of
> 100 times or more.
> 
> I contacted Hans (CaTS developer), and he has seen similar behavior with the
> current version of Opticks. Upon further inspection, we found that the GPU is
> underutilized; in my case, GPU usage is around 20%. He mentioned that with the
> legacy version of Opticks, resources were fully utilized. Do you have any
> comments or ideas how we can debug this?


Opticks performance is extremely dependent on:

1. source geometry 
2. modelling of source geometry 
3. choices made in the geometry translation (eg instancing criteria)
4. problem solids in the geometry (eg deep CSG trees can kill performance)

By extreme, I mean really extreme : I have seen slowdowns of 100x in the past. 

Because of the extreme geometry sensitivity I suggest you do performance 
tests with a geometry that is close to the one you really care about. 
Experience and optimizations directed at very simple test geometry 
will almost certainly not be relevant to the real geometry. 

If you have contacts with NVIDIA engineers with expertise
in their profiling tools and OptiX you really want to use
this valuable opportunity to profile geometry that you 
care about.  

To find problem geometry I have a procedure (ELV scanning) that dynamically 
changes solids included/excluded within the geometry measuring 
ray trace render performance for each using CSGOptiX/cxr_scan.sh
As simulation performance is limited by finding intersects fast 
rendering corresponds to fast simulation. 

The render scan results allows to create a table 
that orders all the geometry solid inclusions/exclusions
by their render times. If the difference between the 
fastest and slowest is reasonable (<~5x) then you 
can say you do not have any extremely bad solids.  
I plan to do this again for JUNO during July. 
I will write some notes documenting how to do this so 
others can try to do the same. 

A big difference between old and new Opticks is that I was able to spend 
many months dedicated to optimization with the old version. 
Including communications with OptiX experts from NVIDIA.  

So far with the new version I have spent very little time on optimization. 
The ever changing JUNO geometry and the issues that it causes have been taking 
my time. 

To benefit from the little optimization work that I have done, you need to 
use a Release build. See : notes/sop/release_build.rst
With JUNO I only got a ~10% improvement with this. 
I expect there is lots more to be had from kernel simplification, 
in addition to improvements you can get from sorting out any 
geometry issues.

Another thing to consider is the size of your launches. 
If they are too small the overheads will limit you gains.




NVIDIA OptiX forum on GPU utilization
----------------------------------------

* https://forums.developer.nvidia.com/t/optix-low-computational-usage-on-gpu/218442

dhart (David Hart, NVIDIA Optix expert)::

    Hi @hashemi_amirreza, welcome!

    How are you measuring GPU utilization? Be aware that due to the proprietary ray
    tracing cores, some Nvidia tools do not currently show you a complete picture
    of utilization; the ray tracing workload may not appear as either compute or
    memory usage. Often high compute and memory is actually a sign of low
    efficiency, or simply of complex shaders, so low compute & memory usage is not
    necessarily bad.

    It may be worth focusing on the overall performance, and comparing that to the
    expected performance. Have you measured your rendering throughput in rays per
    second? How fast are you expecting it to render, and how fast does it currently
    render?

    –
    David.  







 
