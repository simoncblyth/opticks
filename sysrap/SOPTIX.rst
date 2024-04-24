sysrap/SOPTIX,SCUDA,SMesh,SGLFW : triangulated machinery 
===========================================================


Overview
----------

The SOPTIX,SCUDA,SMesh,SGLFW structs were developed to learn how to 
implement a triangulated geometry workflow with the NVIDIA OptiX 7+ API
and also to allow OpenGL rendering of the triangulated geometry.  


Structs
---------

* SOPTIX_Scene.h : top level, holds vectors of SCUDA_MeshGroup SOPTIX_MeshGroup and OptixInstance 
* SOPTIX_SBT.h : create sbt from pipeline and scene by uploding the prog and hitgroup records

* SOPTIX.h : OptixDeviceContext + SOPTIX_Properties  
* SOPTIX_Properties.h : optixDeviceContextGetProperty results

* SOPTIX_Module.h : Create OptixModule from PTX loaded from file
* SOPTIX_Pipeline.h : Create OptixPipeline from OptixModule
* SOPTIX_Options.h : module and pipeline compile/link options
* SOPTIX_OPT.h : enum strings
* SOPTIX.cu : RG, CH progs for simple triangle renderer
* SOPTIX_Params.h : render control 

* SOPTIX_Accel.h : builds acceleration structure GAS or IAS from the buildInputs
* SOPTIX_Binding.h : CPU/GPU SBT records
* SOPTIX_getPRD.h : unpackPointer from optixPayload

* SOPTIX_BuildInput_Mesh.h : create OptixBuildInput via "part" indexing into SCUDA_MeshGroup (Used from SOPTIX_MeshGroup)
* SOPTIX_MeshGroup.h : create SOPTIX_BuildInput_Mesh for each part of SCUDA_MeshGroup, use those to form SOPTIX_Accel gas  

* SCUDA_Mesh.h : uploads SMesh tri and holds SCU_Buf 
* SCUDA_MeshGroup.h : collect vectors of NP from each SMeshGroup sub, upload together with SCU_BufferView 

* SMeshGroup.h : collection of SMesh subs and names
* SMesh.h : holds tri,vtx,nrm NP either from original G4VSolid conversion or concatenation


* SGLFW.h : Light touch OpenGL render loop and key handling
* SGLFW_Keys.h : record of keyboard keys currently held down with modifiers bitfield summarization
* SGLFW_Extras.h : Toggle, GLboolean, bool, GLenum, Attrib, Buffer, VAO 


* SGLFW_CUDA.h : Coordinate SCUDAOutputBuffer and SGLDisplay for display of interop buffers
* SCUDAOutputBuffer.h : Allows an OpenGL PBO buffer to be accessed from CUDA 
* SGLDisplay.h : OpenGL shader pipeline that presents PBO to screen

* SGLFW_Program.h : compile and link OpenGL pipeline using shader sources loaded from directory
* SGLFW_Mesh.h : create OpenGL buffers with SMesh and instance data and render
* SGLFW_Scene.h : manage scene data and OpenGL render pipelines 



tests
-------

SGLFW_SOPTIX_Scene_test.{sh,cc}
    TODO: a little bit more encapsulation into SOPTIX_Scene.h 

SOPTIX_Scene_test.{sh,cc}
    TODO: tidy up using enhancements from above 

SOPTIX_Module_test.{sh,cc}
    TODO: CHECK

SOPTIX_Options_test.{sh,cc}
    TODO: CHECK

SOPTIX_Pipeline_test.{sh,cc}
    TODO: CHECK



SOPTIX_SBT::initHitgroup
---------------------------

HMM: with analytic geometry have "boundary" that 
comes from the CSGNode. To do that with triangles 
need to plant the boundary indices into HitgroupData.  
That means need hitgroup records for each sub-SMesh 
(thats equivalent to each CSGPrim)

Need nested loop like CSGOptiX/SBT.cc SBT::createHitgroup::
 
     GAS 
        BuildInput       (actually 1:1 with GAS) 
           sub-SMesh 

So need access to scene data to form the SBT 




