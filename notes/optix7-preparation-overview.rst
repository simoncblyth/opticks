optix7-preparation-overview
==============================

::

    c() {  cd ~/CSG ; git status ; }
    cg() { cd ~/CSG_GGeo; git status ; }
    cx() { cd ~/CSGOptiX; git status ; }
    qu() { quadarap ; }



CMake project dependencies
-----------------------------


::

      CSG : CUDA SysRap 

      CSG_GGeo :  CUDA CSG GGeo  

      CSGOptiX : CUDA OpticksCore CSG 

      QudaRap : GGeo OpticksCUDA



Project summaries
------------------------

CSG
    base model, including CSGFoundary nexus, creates CSGFoundry geometries

CSG_GGeo
    loads GGeo geometry and creates and saves as CSGFoundary 
  
CSGOptiX
    renders CSGFoundary geometry using either OptiX pre-7 or 7 

QudaRap
    pure CUDA photon generation, revolving around GPU side qctx.h 


TODO
-----

* prototype project structure for integrating QudaRap qctx.h with OptiX 7 running like CSGOptiX 

  * new package name ? CSGQuda? 
  * how to split rendering and simulation functionality : with duplication avoided ?
  * from perusing CSGOptiX.h looks like need to pull off common geometry core : CSGOptiXGeo ? 
  * TODO: effect the split : what is render specific ? 

    * separate pipelines ? PIP::init names the raygen/miss/hitgroup programs, easy split based on the names 
    * separate Param.h ? its simple enough that having common param seems not so problematic
    * tuck rendering stuff into separate struct or just separate methods ?


* bring CSG, CSG_GGeo and CSGOptiX under opticks umbrella joining QudaRap  


How much of a separation between rendering and simulation ?
--------------------------------------------------------------

* rendering means the ability to save jpg files viewing geometry, so it adds no dependencies 
* separation is just for clarity of organization, no strong technical need 


rendering
    viewpoint input yields frame of pixels

simulation
    genstep input yields buffer of photons 


optix 7 rdr/sim separation at what level PIP/SBT or within the raygen function ?  
----------------------------------------------------------------------------------

::

     58 PIP::PIP(const char* ptx_path_)
     59     :
     60     max_trace_depth(2),
     61     num_payload_values(8),
     62     num_attribute_values(4),
     63     pipeline_compile_options(CreatePipelineOptions(num_payload_values,num_attribute_values)),
     64     program_group_options(CreateProgramGroupOptions()),
     65     module(CreateModule(ptx_path_,pipeline_compile_options))
     66 {
     67     init();
     68 }



* at first glance would seem having separate PIP "rdr" "sim" instances seems appropriate as different payload attribute values etc..
  
  * but looks like would add lots of code/complexity 
  * SBT takes pip ctor argument, so separate SBT too ?
  * hmm annoying to need 2nd SBT for teeing up different raygen data : when hardly use that 
  * SBT is primarily for geometry and hence common : is there some way to keep it fully common ? 

* simulation performance is much more critical so will optimize for that anyhow
* the purpose of the rendering is as a visual geometry check of the simulation geometry, 
  which is best served by keeping the sim/rdr branches as close as possible  

* hmm having a single raygen with a param rgmode to switch between rendering and simulation looks 
  very attractive for minimizing code divergence

  * i like the radical simplicity of that approach  
  * my rendering is totally minimal, expect simulation will use more resources  
    so this approach may be fine in long run too 


Prototype thoughts
-----------------------

* new package depending on CSGOptiX and QudaRap ?

* start with purely numerical approach : fabricate a torch genstep and check intersects of 
  generated photons with the optix 7 geometry 

* technically how to get access to the qctx "physics context" from optix 7 intersect code ? 
  look at how the geometry data is uploaded 

  * examine the cx optix launch to see how to introduce the qctx ? another param ? 

* CSGOptiX is too render specific need a lower level intermediate struct
  that can be common to both rendering and simulation  



