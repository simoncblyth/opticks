DONE : OKConf_header_only_to_simplify_STTF_usage
==================================================

Changes::

   SIMG.hh -> SIMG.h
   STTF.hh -> STTF.h 
   removed SLOG::ttf 



STTF.hh usage needs simplification, it is smeared around too much:: 

    OKConf
    SLOG 
    SIMG

In turn that complicates SIMG.hh and Frame also. 


::


    [blyth@localhost opticks]$ opticks-f STTF | grep -v STTF.hh | grep -v STTFTest 
    ./bin/opticks-setup-minimal.sh:   export OPTICKS_STTF_PATH=/usr/local/opticks/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf
    ./externals/g4.bash:    STTF::GetFontPath dpath /home/simon/local/opticks/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf epath  

    ./okconf/OKConf.cc:    std::cout << std::setw(50) << "OKConf::DefaultSTTFPath()      "    << OKConf::DefaultSTTFPath() << std::endl ; 
    ./okconf/OKConf.cc:const char* OKConf::DefaultSTTFPath()  // static
    ./okconf/OKConf.h:       static const char* DefaultSTTFPath();  
    ./okconf/OKConf.h:    std::cout << std::setw(50) << "OKConf::DefaultSTTFPath()      "    << OKConf::DefaultSTTFPath() << std::endl ; 
    ./okconf/OKConf.h:inline const char* OKConf::DefaultSTTFPath()  // static
    ./okconf/OKConf.hh:       static const char* DefaultSTTFPath();  

    ./optixrap/OContext.cc:#define STTF_IMPLEMENTATION 1 
    ./optixrap/OContext.hh:struct STTF ; 
    ./optixrap/OContext.hh:            STTF*             m_ttf ; 

    ./sysrap/CMakeLists.txt:set(WITH_STTF YES)
    ./sysrap/CMakeLists.txt:if(WITH_STTF)
    ./sysrap/CMakeLists.txt:if(WITH_STTF)
    ./sysrap/CMakeLists.txt:target_compile_definitions( ${name} PUBLIC WITH_STTF)

    ./sysrap/SIMG.hh:#define STTF_IMPLEMENTATION 1 
    ./sysrap/SIMG.hh:    STTF* ttf = SLOG::instance ? SLOG::instance->ttf : nullptr ; 
    ./sysrap/SIMG.hh:    STTF* ttf = nullptr ; 

    ./sysrap/SLOG.cc:#define STTF_IMPLEMENTATION 1 
    ./sysrap/SLOG.cc:    ttf(new STTF),
    ./sysrap/SLOG.cc:    ttf(new STTF),
    ./sysrap/SLOG.hh:struct STTF ; 
    ./sysrap/SLOG.hh:    STTF*       ttf ;    // truetypefont

    ./sysrap/tests/CMakeLists.txt:if(WITH_STTF)
    ./opticks.bash:export OPTICKS_STTF_PATH=\$OPTICKS_PREFIX/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf
    [blyth@localhost opticks]$ 



See failing "simple" build::

    [blyth@localhost ~]$ ~/o/sysrap/tests/SIMGTest.sh 
                name : SIMGTest 
                FOLD : /tmp/SIMGTest 
                 bin : /tmp/SIMGTest/SIMGTest 
    In file included from ../OPTICKS_LOG.hh:175,
                     from SIMGTest.cc:9:
    ../SLOG.hh:33:10: fatal error: plog/Log.h: No such file or directory
       33 | #include <plog/Log.h>
          |          ^~~~~~~~~~~~
    compilation terminated.


Fixed by moving to SIMG.h STTF.h::

    [blyth@localhost tests]$ IMGPATH=/tmp/flower.jpg ~/o/sysrap/tests/SIMGTest.sh
                    name : SIMGTest 
                    FOLD : /tmp/SIMGTest 
                     bin : /tmp/SIMGTest/SIMGTest 
    SIMG width 640 height 427 channels 3 loadpath /tmp/flower.jpg loadext .jpg
    [blyth@localhost tests]$ 




WIP : Replace all use of STTF.hh with STTF.h and remove ttf from SLOG
-----------------------------------------------------------------------

::

    [blyth@localhost opticks]$ opticks-f STTF.hh
    ./optixrap/OContext.cc:#include "STTF.hh"
    DEAD CODE : AND LOOKS WAS UNUSED ANYHOW

    ./sysrap/CMakeLists.txt:    list(APPEND HEADERS STTF.hh)
    // REMOVED

    ./sysrap/SIMG.hh:#include "STTF.hh"
    // REMOVED

    ./sysrap/SLOG.cc:#include "STTF.hh"
    // REMOVED

    ./sysrap/STTF.hh:STTF.hh
    ./sysrap/STTF.hh:as STTF.hh and SIMG.hh are otherwise purely header-only.  

    ./sysrap/tests/STTFTest.cc:#include "STTF.hh"
    DONE : NOW BUILDS FROM INSTALLED SYSRAP HEADERS : NOT THE LIB 

    ./sysrap/STTF.h:STTF.h : try for simpler usage than STTF.hh
    ./opticks.bash:## see sysrap/STTF.hh still needed for binary release
    [blyth@localhost opticks]$ 




WIP : Replace all use of SIMG.hh with SIMG.h and profit from that to simplify CSGOptiX.cc Frame usage
------------------------------------------------------------------------------------------------------

::

    [blyth@localhost opticks]$ opticks-f SIMG.hh
    ./CSGOptiX/Frame.cc:#include "SIMG.hh"
    DONE

    ./examples/UseOptiX7GeometryInstancedGASCompDyn/Frame.cc:#include "SIMG.hh"
    SKIP : decided not to update as this is using copied in SIMG.hh not the sysrap one 

    ./examples/UseSysRapSIMG/UseSysRapSIMG.cc:#include "SIMG.hh"
    DONE

    ./optixrap/OContext.cc:#include "SIMG.hh"
    DEAD CODE 

    ./qudarap/tests/QTexRotateTest.cc:#include "SIMG.hh"
    DONE

    ./sysrap/CMakeLists.txt:    list(APPEND HEADERS   SIMG.hh  )
    ./sysrap/SIMG.hh:SIMG.hh : DEPRECATED : MOVING TO USE SAME FUNCTIONALITY BUT LESS DEPENDENCY SIMG.h 

    ./sysrap/STTF.hh:as STTF.hh and SIMG.hh are otherwise purely header-only.  

    ./sysrap/tests/STTFTest.cc:#include "SIMG.hh"
     DONE

    ./sysrap/SIMG.h:SIMG.h : trying to make SIMG.hh simpler to use by cutting dependencies

    [blyth@localhost opticks]$ 




