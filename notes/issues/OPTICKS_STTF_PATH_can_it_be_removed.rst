OPTICKS_STTF_PATH_can_it_be_removed.rst
========================================


Can the envvar be removed from setup ?  YES : looks so


::

    epsilon:issues blyth$ opticks-f OPTICKS_STTF_PATH
    ./bin/opticks-setup-minimal.sh:   export OPTICKS_STTF_PATH=/usr/local/opticks/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf
    ./sysrap/STTF.hh:    static constexpr const char* KEY = "OPTICKS_STTF_PATH" ; 
    ./sysrap/tests/STTFTest.sh:#export OPTICKS_STTF_PATH=/Library/Fonts/Arial.ttf 
    ./sysrap/tests/STTFTest.sh:echo OPTICKS_STTF_PATH $OPTICKS_STTF_PATH
    ./opticks.bash:export OPTICKS_STTF_PATH=\$OPTICKS_PREFIX/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf
    epsilon:opticks blyth$ 



::

    epsilon:sysrap blyth$ opticks-f STTF.hh
    ./sysrap/SIMG.hh:#include "STTF.hh"
    ./sysrap/CMakeLists.txt:    list(APPEND HEADERS STTF.hh stb_truetype.h)
    ./sysrap/STTF.hh:STTF.hh
    ./sysrap/STTF.hh:as STTF.hh and SIMG.hh are otherwise purely header-only.  
    ./sysrap/tests/STTFTest.cc:#include "STTF.hh"
    ./sysrap/SLOG.cc:#include "STTF.hh"
    ./opticks.bash:## see sysrap/STTF.hh
    ./optixrap/OContext.cc:#include "STTF.hh"
    epsilon:opticks blyth$ 


::

     84 inline const char* STTF::GetFontPath() // static
     85 {
     86     const char* dpath = OKConf::DefaultSTTFPath() ;
     87     const char* epath = getenv(KEY) ;
     88     //printf("STTF::GetFontPath dpath %s epath %s \n", ( dpath ? dpath : "" ), ( epath ? epath : "" ) );    
     89     return epath ? epath : dpath ;
     90 }

::

    epsilon:issues blyth$ OKConfTest | grep STTF
                       OKConf::DefaultSTTFPath()      /usr/local/opticks/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf

::

    /**
    OKConf::OpticksInstallPrefix
    -----------------------------

    NB resolution order:

    1. envvar OPTICKS_INSTALL_PREFIX
    2. internally set OKCONF_OPTICKS_INSTALL_PREFIX 
       coming from the CMAKE_INSTALL_PREFIX used 
       for the installation

    **/
    const char* OKConf::OpticksInstallPrefix()
    {
    #ifdef OKCONF_OPTICKS_INSTALL_PREFIX
       const char* evalue = getenv("OPTICKS_INSTALL_PREFIX") ;   
       return evalue ? evalue : OKCONF_OPTICKS_INSTALL_PREFIX ;
    #else 
       return "MISSING" ; 
    #endif    
    }

    278 const char* OKConf::DefaultSTTFPath()  // static
    279 {
    280     std::stringstream ss ;
    281     ss << OKConf::OpticksInstallPrefix()
    282        << "/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf"
    283        ;
    284     std::string shaderdir = ss.str();
    285     return strdup(shaderdir.c_str());
    286 }

