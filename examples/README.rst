Examples of Opticks Usage, especially the CMake machinery  
=============================================================

The examples have go.sh scripts which perform the normal 
config, make, install using a fresh build dir every time.
This is a convenient when the focus is on the CMake machinery. 

UsePLog
    simple executable using PLog via imported interface target 
    from cmake/Modules/FindPLog.cmake

 
UseGLMRaw
   (plain vanilla modern CMake with target import/export)

   * find_package(GLM) succeeds due to cmake/Modules/FindGLM.cmake which 
     exports the Opticks::GLM interface only target 
     (GLM is just headers) 

   * vending targets as of CMake 3.5 takes lots of boilerplate


UseUseGLM
   (plain vanilla modern CMake)

   * find_package(UseGLM) succeed due to the exported target written by UseGLM

   * nice and simple : plain CMake is fine when consuming 
     targets only without vending.  

   * Note that only need to be concerned with direct dependency on UseGLM.
     The dependency of UseGLM on GLM comes along with the imported target with no effort.

   * Also note that the dependency transmission is working across package boundaries, 
     ie UseGLM and UseUseGLM are not tied together in a single CMake build, 
     they are entirely separate projects.



UseGLMViaBCM
    (modern CMake assisted by BCM)

   * much less boilerplate, intent is clear

   * forced(?) to place headers into an include folder in source dir 
   * forced(?) in install at prefix/include rather than current opticks location prefix/include/name/

   * HMM: NOT KEEN ON CO-MINGLING HEADERS OF ALL PROJECTS IN A SINGLE include DIR, OR ON HAVING 
     THE HASSLE OF SEPARATE include DIR for sources 
   

UseUseGLMViaBCM
    
   * similar observations to UseUseGLM

UseSysRap
   vends a library that uses the SysRap target exported by sysrap/CMakeLists.txt  

UseUseSysRap
   uses the UseSysRap exported library, succeeds to auto-find the dependencies (SysRap)
   of its direct dependent UseSysRap 

UseOpticksBoost
   no longer operational exercise for the old variable-centric approach to Boost hookup 

UseBoost 
   Attempt to vend a library target that uses Boost::filesystem::
 
       16 find_package(Boost REQUIRED COMPONENTS filesystem)
       17 
       18 ## kludge that tees up arguments to find_dependency in generated export useboost-config.cmake 
       19 ## so downstream will automatically do the above find_package 
       20 set_target_properties(Boost::filesystem PROPERTIES INTERFACE_FIND_PACKAGE_NAME "Boost REQUIRED COMPONENTS filesystem")


UseUseBoost
   attempt to use the lib target exported from UseBoost, initially failed to auto hookup  
   the non-direct Boost::filesystem dependent, because useboost-config.cmake had::

       include(CMakeFindDependencyMacro)
       # Library: Boost::filesystem
       find_dependency(Boost) 
       include("${CMAKE_CURRENT_LIST_DIR}/useboost-targets.cmake")
       include("${CMAKE_CURRENT_LIST_DIR}/properties-useboost-targets.cmake")

   Suspect problem is that the non-BCM exported targets lack some needed metadata ? YEP, BCM 
   relies on setting target properties in bcm_deploy that get read on generating the exported target
   serialization.  Kludge fix is to misuse that property as shown above, so that the imported target
   automatically does the necessary::

        find_dependency(Boost REQUIRED COMPONENTS filesystem)  
        # this works with cmake_minimum_version set to 3.5 with cmake 3.11 


UseBoostRap
   testing dependency isolation of BoostRap 

UseOpenMesh
   check consumption of BCM exported targets done by cmake/Modules/FindOpenMesh.cmake


UseNPY(needs-revisit)
    old first attempt using raw inclusion of exported targets with 
    the non-standard approach of cmake -DOPTICKS_CONFIG_DIR=/usr/local/opticks/config
    rather than using the standard find_package mechanism 

    * exporting was done in opticksnpy/CMakeLists.txt
      BUT: this approach is jumping into the thicket of the dependency tree.  Better to 
      get experience out in the leaves, before tackling the interior of the bush.  




UseOptiX
   really minimal usage of OptiX C API, checking creation of context and buffer, 
   no kernel launching

UseOptiXProgram
   OptiX C API creates raygen program and launches it, just dumping launch index  

UseOptiXProgramPP
   OptiX C++ API variant of the above : provides a command line interface to quickly run 
   simple OptiX code (no buffers in context).

UseOptiXBuffer
    OptiX C API creates raygen program that just writes constant values to a buffer

UseOptiXBufferPP
   OptiX C++ API : creates in and out buffers from NPY arrays and launches a program that 
   simply copies from in to out.  Provides a command line interface to quickly run variants
   of the buffer accessing GPU code. 

UseOptiXGeometry
   Minimally demonstrate OptiX geometry without using OXRAP, performs a "standalone" raytrace
   of a box with normal shader coloring.
 
UseOptiXGeometryTriangles
   Minimally demonstrate the use of optix::GeometryTriangles introduced in OptiX 6.0.0. 
   Raytraces an octahedron writing a PPM file. 
   Based on NPY and SYSRAP for buffer and PPM handling. No OXRAP.

   * https://raytracing-docs.nvidia.com/optix/api/html/group___geometry_triangles.html

UseOContextBufferPP
   Use the OptiXRap.OContext to reimplement UseOptiXBufferPP in a higher level style, 
   hoping to approach close enough to UseOptiXRap for the problem to manifest.  
   But it hasnt.

UseOptiXRap
   Uses Opticks higher level OptiXRap API to test changing the sizes of buffers.  

   Issue with OptiX 6.0.0 : the buffer manipulations seem to work but the rtPrintf 
   output does not appear unless the buffer writing is commented out.

   Huh, now rtPrintf seems to be working without any clear fix.  
   Now not working.
   Now working again, immediately after an oxrap--  

   Perhaps a problem of host code being updated and PTX not, because the
   PTX is from oxrap ?

   Can change the progname via envvar::

       USEOPTIXRAP_PROGNAME="bufferTest_2" UseOptiXRap   



UseOpticksGLFW
   minimal use of OpenGL via GLFW, pops up a window and renders a colorful rotating triangle. 
   Key presses cause the GLFW_KEY_XX enum name to be emitted to stdout. Press ESCAPE to exit.

UseOpticksGLFWSnap
   variant of UseOpticksGLFW adding the capability to save screen images to PPM files

UseOpticksGLFWSPPM
   variant of UseOpticksGLFWSnap with the PPM handling from reusable sysrap/SPPM 

UseShader
   Formerly named UseOpticksGLFWShader

   * adapted GLFW example, modified to use GLEW and GLM : it ran giving a black screen.
   * adding a VAO makes the coloured triangle appear      
   * added error checking and compilation log output 

   This is a good starting point for creating self contained minimal reproducers. 

UseOGLRapMinimal
   Creates red-green-blue axes that can interact with using the usual controls. 
   Tests the Rdr axis renderer in isolation using just Composition, Frame and Interactor
   (no Scene).

UseGeometryShader
   Creates red-green-blue axes

   Implemented in standalone single file fashion that sets up a geometry shader 
   pipeline using the same shader strings as the Rdr axis renderer 
   as used by UseOGLRapMinimal.  All the mat4 have been matched with
   UseOGLRapMinimal.

   Features a monolithic standalone getMVP, providing the ModelViewProjection matrix, which 
   is useful for demo code.::

       glm::mat4 getMVP(int width, int height, bool verbose)

   Actually it was the comparison of the mat4 between
   UseOGLRapMinimal which uses View::getTransforms 
   and my standalone reimplementation of the matrix manipulations 
   in UseGeometryShader that led to finding the "uninitialized forth row bug" 
   that has been lurking for years ready to bite just at the wrong time 
   following a Linux kernel and driver update and OptiX update.
    
   See the mis-named: notes/issues/OGLRap_GLFW_OpenGL_Linux_display_issue_with_new_driver.rst 


UseOGLRap
   same as OGLRap AxisAppCheck 

UseOpticksGL
   OAxisTest appears to be trying to change things with OptiX launches whilsy displaying with OpenGL

UseOpticksGLEW
    Just dumping version numbers from header. CMake machinery test.





Standalone-ish OptiX 7 Examples
-----------------------------------

UseOptiX7
    Basic check of CMake machinery, finding OptiX 7

UseOptiX7GeometryStandalone
    Start from the SDK optixSphere example, which uses custom(aka analytic) geometry.
    Follows the monolithic layout of optixSphere, just adapting to use glm for 
    viewpoint math.

UseOptiX7GeometryModular
    Apply wrecking ball to the monolith, splitting into: 

    Engine
       context, control
    Binding 
       common types between CPU and GPU 
    PIP
       pipeline of programs creation and updating  
    GAS
       geometry acceleration structure building 

UseOptiX7GeometryInstanced
    Attempting to switch UseOptiX7GeometryModular to use an
    instanced custom geometry for lots of spheres.





Standalone-ish OptiX 6 Texture Examples
----------------------------------------

UseOptiXTexture
    C API 3D texture creation, with pullback test into out_buffer

UseOptiXTextureLayered
    Switch from 3D to layered 2D texture, *exfill* attempt to fill with MapEx failed 

UseOptiXTextureLayeredPP
    Convert to use OptiX 6 C++ API 

UseOptiXTextureLayeredOK
    Start encapsulation into Make2DLayeredTexture

UseOptiXTextureLayeredOKImg
    Use ImageNPY::LoadPPM to load images into textures 
    First try at 2d layered tex failed, so reverted to 2d textures.

UseOptiXTextureLayeredOKImgGeo
    Ray-traced theta-phi texture mapping onto a sphere, when using an Earth texture this provides 
    Earth view PPM images centered around any latitude-longitude position.
    This example was used to develop the watertight OptiX OCtx wrapper (C opaque pointer style) 
    which does not leak any optix types into its interface.

UseOptiXGeometryInstanced
    start from UseOptiXGeometryInstancedStandalone, plan:
    
    1. DONE: Opticks packages to reduce the amount of code
    2. DONE: adopt OCtx watertight wrapper, adding whats needed for instancing  
    3. DONE: add optional switch from box to sphere 
    4. get a layered texture to work with instances, such that 
       different groups of instances use different layers  
    5. DONE: generate PPM of thousands of textured Earths  

UseOptiXGeometryInstancedOCtx
    start from UseOptiXGeometryInstanced, using just OCtx 
   
    * test use of multiple layers 
    * 1d layered float textures

UseOptiXGeometryOCtx
    start from UseOptiXGeometry to investigate why getting problem with instanced spheres in OptiX 6.5



