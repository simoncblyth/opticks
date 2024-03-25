SOPTIX_Scene_test_needs_some_image_handling_before_leap_to_OpenGL
====================================================================


::

    epsilon:UseOptiX7GeometryInstanced blyth$ opticks-f SIMG.h 
    ./CSGOptiX/Frame.cc:#include "SIMG.hh"
    ./sysrap/CMakeLists.txt:    list(APPEND HEADERS   SIMG.hh  )
    ./sysrap/STTF.hh:as STTF.hh and SIMG.hh are otherwise purely header-only.  
    ./sysrap/tests/SIMGStandaloneTest.cu:#include "SIMG.hh"
    ./sysrap/tests/STTFTest.cc:#include "SIMG.hh"
    ./sysrap/tests/SIMGTest.cc:#include "SIMG.hh"
    ./qudarap/tests/QTexRotateTest.cc:#include "SIMG.hh"
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/Frame.cc:#include "SIMG.hh"
    ./examples/UseSysRapSIMG/UseSysRapSIMG.cc:#include "SIMG.hh"
    ./optixrap/OContext.cc:#include "SIMG.hh"
    epsilon:opticks blyth$ 

    epsilon:opticks blyth$ opticks-f SPPM.h 
    ./okop/OpTracer.cc:#include "SPPM.hh"
    ./sysrap/CMakeLists.txt:    SPPM.hh
    ./sysrap/tests/SPPMTest.cc:#include "SPPM.hh"
    ./sysrap/SPPM.cc:#include "SPPM.hh"
    ./examples/UseOptiX7GeometryInstancedGAS/Engine.cc:#include "SPPM.h"
    ./examples/UseOptiX7GeometryInstancedGASComp/Engine.cc:#include "SPPM.h"
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/Frame.cc:#include "SPPM.h"
    ./examples/UseOptiXGeometry/UseOptiXGeometry.cc:#include "SPPM.hh"
    ./examples/UseOptiXGeometryOCtx/UseOptiXGeometryOCtx.cc:#include "SPPM.hh"
    ./examples/UseOptiXGeometryTriangles/UseOptiXGeometryTriangles.cc:#include "SPPM.hh"
    ./examples/UseOpticksGLFWSPPM/UseOpticksGLFWSPPM.cc:#include "SPPM.hh"
    ./optixrap/OContext.cc:#include "SPPM.hh"
    ./npy/ImageNPY.cpp:#include "SPPM.hh"
    ./oglrap/Frame.cc:#include "SPPM.hh"
    ./oglrap/Pix.hh:#include "SPPM.hh"
    epsilon:opticks blyth$ 

