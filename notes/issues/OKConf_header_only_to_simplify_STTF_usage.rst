OKConf_header_only_to_simplify_STTF_usage
==========================================



STTF usage needs simplification, it is smeared around too much:: 

    OKConf
    SLOG 
    SIMG


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




