#pragma once
/**
OKConf.h : header only version of OKConf.hh that aims to replace OKConf.hh
============================================================================

Static methods providing installation constants and
version number integers of externals.


**/


#include "OKConf_Config.hh"


#ifdef WITH_CUDA

#include <fstream>
#include <string>
#include <iostream>
#include <regex>


#ifdef OKCONF_OPTIX_VERSION_INTEGER

#define OKCONF_OPTIX_VERSION_MAJOR (OKCONF_OPTIX_VERSION_INTEGER / 10000)
#define OKCONF_OPTIX_VERSION_MINOR ((OKCONF_OPTIX_VERSION_INTEGER % 10000) / 100)
#define OKCONF_OPTIX_VERSION_MICRO (OKCONF_OPTIX_VERSION_INTEGER % 100)

#else

#define OKCONF_OPTIX_VERSION_INTEGER 0
#define OKCONF_OPTIX_VERSION_MAJOR 0
#define OKCONF_OPTIX_VERSION_MINOR 0
#define OKCONF_OPTIX_VERSION_MICRO 0

#endif
#endif


struct OKConf
{
    static int Check();
    static void Dump(const char* msg="OKConf::Dump");
    static const char* OpticksInstallPrefix();
    static const char* OpticksGitHash();
    static const char* CMAKE_CXX_FLAGS();

#ifdef WITH_CUDA
    // DESPITE APPEARING TO NEED CUDA THE BELOW ALL COME FROM MACROS AT CMake LEVEL
    // THE CUDA API IS NOT USED
    static const char* OptiXInstallDir();
    //static const char* CUDA_NVCC_FLAGS();
    static int ComputeCapabilityInteger();
    static int OptiXVersionInteger() ;

    static int OptiXVersionMajor() ;
    static int OptiXVersionMinor() ;
    static int OptiXVersionMicro() ;
    static int CUDAVersionInteger() ;
    static const char* PTXPath( const char* cmake_target, const char* cu_name, const char* ptxrel=nullptr );

    static const char* NvidiaDriverVersion();

#endif

    // static unsigned CLHEPVersionInteger();    see x4/tests/CLHEPVersionInteger.cc
    static int Geant4VersionInteger() ;

    static int OpticksVersionInteger();
    static int OpticksVersionInteger_Gymnastically();

    static const char* ShaderDir();
    static const char* DefaultSTTFPath();
};


#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

#include "OpticksVersionNumber.hh"





inline int OKConf::Check()
{
   int rc = 0 ;

   if(OpticksVersionInteger() == 0)
   {
       rc += 1 ;
   }
   if(OpticksVersionInteger_Gymnastically() == 0)
   {
       rc += 1 ;
   }



#ifdef WITH_CUDA
   if(CUDAVersionInteger() == 0)
   {
       rc += 1 ;
   }

   if(OptiXVersionInteger() == 0)
   {
       rc += 1 ;
   }
   if(ComputeCapabilityInteger() == 0)
   {
       rc += 1 ;
   }
   /*
   // this setup is now downstream of OKConf, actually from OKConf TOPMATTER
   if(CUDA_NVCC_FLAGS() == 0)
   {
       rc += 1 ;
   }
   */

   if(OptiXInstallDir() == 0)
   {
       rc += 1 ;
   }
#endif


   if(OpticksGitHash() == 0)
   {
       rc += 1 ;
   }

   if(CMAKE_CXX_FLAGS() == 0)
   {
       rc += 1 ;
   }

   if(Geant4VersionInteger() == 0)
   {
       rc += 1 ;
   }
   return rc ;
}


inline void OKConf::Dump(const char* msg)
{
    std::cout << msg << std::endl ;
    std::cout << std::setw(50) << "OKConf::OpticksVersionInteger() "   << OKConf::OpticksVersionInteger() << std::endl ;
    std::cout << std::setw(50) << "OKConf::OpticksVersionInteger_Gymnastically() "  << OKConf::OpticksVersionInteger_Gymnastically() << std::endl ;
    std::cout << std::setw(50) << "OKConf::OpticksInstallPrefix() "    << OKConf::OpticksInstallPrefix() << std::endl ;
    std::cout << std::setw(50) << "OKConf::OpticksGitHash() "          << OKConf::OpticksGitHash() << std::endl ;
    std::cout << std::setw(50) << "OKConf::CMAKE_CXX_FLAGS() "         << OKConf::CMAKE_CXX_FLAGS() << std::endl ;


#ifdef WITH_CUDA
    std::cout << std::setw(50) << "OKConf::CUDAVersionInteger() "      << OKConf::CUDAVersionInteger() << std::endl ;
    std::cout << std::setw(50) << "OKConf::NvidiaDriverVersion() "     << OKConf::NvidiaDriverVersion() << std::endl ;
    std::cout << std::setw(50) << "OKConf::ComputeCapabilityInteger() "<< OKConf::ComputeCapabilityInteger() << std::endl ;
    //std::cout << std::setw(50) << "OKConf::CUDA_NVCC_FLAGS() "         << OKConf::CUDA_NVCC_FLAGS() << std::endl ;

    std::cout << std::setw(50) << "OKConf::OptiXInstallDir() "         << OKConf::OptiXInstallDir() << std::endl ;
    std::cout << std::setw(50) << "OKCONF_OPTIX_VERSION_INTEGER "      << OKCONF_OPTIX_VERSION_INTEGER << std::endl ;
    std::cout << std::setw(50) << "OKConf::OptiXVersionInteger() "     << OKConf::OptiXVersionInteger() << std::endl ;
    std::cout << std::setw(50) << "OKCONF_OPTIX_VERSION_MAJOR   "      << OKCONF_OPTIX_VERSION_MAJOR << std::endl ;
    std::cout << std::setw(50) << "OKConf::OptiXVersionMajor() "       << OKConf::OptiXVersionMajor() << std::endl ;
    std::cout << std::setw(50) << "OKCONF_OPTIX_VERSION_MINOR   "      << OKCONF_OPTIX_VERSION_MINOR << std::endl ;
    std::cout << std::setw(50) << "OKConf::OptiXVersionMinor() "       << OKConf::OptiXVersionMinor() << std::endl ;
    std::cout << std::setw(50) << "OKCONF_OPTIX_VERSION_MICRO   "      << OKCONF_OPTIX_VERSION_MICRO << std::endl ;
    std::cout << std::setw(50) << "OKConf::OptiXVersionMicro() "       << OKConf::OptiXVersionMicro() << std::endl ;
#endif


    std::cout << std::setw(50) << "OKConf::Geant4VersionInteger() "    << OKConf::Geant4VersionInteger() << std::endl ;
    std::cout << std::setw(50) << "OKConf::ShaderDir()            "    << OKConf::ShaderDir() << std::endl ;
    std::cout << std::setw(50) << "OKConf::DefaultSTTFPath()      "    << OKConf::DefaultSTTFPath() << std::endl ;
    std::cout << std::endl ;
}









#ifdef WITH_CUDA

inline int OKConf::CUDAVersionInteger()
{
#ifdef OKCONF_CUDA_API_VERSION_INTEGER
   return OKCONF_CUDA_API_VERSION_INTEGER ;
#else
   return 0 ;
#endif
}

inline int OKConf::OptiXVersionInteger()
{
#ifdef OKCONF_OPTIX_VERSION_INTEGER
   return OKCONF_OPTIX_VERSION_INTEGER ;
#else
   return 0 ;
#endif
}
inline int OKConf::OptiXVersionMajor()
{
#ifdef OKCONF_OPTIX_VERSION_MAJOR
   return OKCONF_OPTIX_VERSION_MAJOR ;
#else
   return 0 ;
#endif
}
inline int OKConf::OptiXVersionMinor()
{
#ifdef OKCONF_OPTIX_VERSION_MINOR
   return OKCONF_OPTIX_VERSION_MINOR ;
#else
   return 0 ;
#endif
}
inline int OKConf::OptiXVersionMicro()
{
#ifdef OKCONF_OPTIX_VERSION_MICRO
   return OKCONF_OPTIX_VERSION_MICRO ;
#else
   return 0 ;
#endif
}

inline int OKConf::ComputeCapabilityInteger()
{
#ifdef OKCONF_COMPUTE_CAPABILITY_INTEGER
   return OKCONF_COMPUTE_CAPABILITY_INTEGER ;
#else
   return 0 ;
#endif
}

inline const char* OKConf::OptiXInstallDir()
{
#ifdef OKCONF_OPTIX_INSTALL_DIR
   return OKCONF_OPTIX_INSTALL_DIR ;
#else
   return "MISSING" ;
#endif
}

/*
inline const char* OKConf::CUDA_NVCC_FLAGS()
{
#ifdef OKCONF_CUDA_NVCC_FLAGS
   return OKCONF_CUDA_NVCC_FLAGS ;
#else
   return "MISSING" ;
#endif
}
*/

/**
OKConf::PTXPath
-----------------

The path elements configured here must match those from the CMakeLists.txt
that compiles the <name>.cu to <target>_generated_<name>.cu.ptx eg in optixrap/CMakeLists.txt::

    091 set(CU_SOURCES
    092
    093     cu/pinhole_camera.cu
    094     cu/constantbg.cu
    ...
    120     cu/intersect_analytic_test.cu
    121     cu/Roots3And4Test.cu
    122 )
    ...
    131 CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${CU_SOURCES} )
    132 CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )
    133
    134
    135 add_library( ${name} SHARED ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )
    136 #[=[
    137 The PTX are not archived in the lib, it is just expedient to list them as sources
    138 of the lib target so they get hooked up as dependencies, and thus are generated before
    139 they need to be installed
    140 #]=]
    ...
    170 install(FILES ${_generated_PTX_files} DESTINATION installcache/PTX)

The form of the PTX filename comes from the FindCUDA.cmake file for example at
/usr/share/cmake3/Modules/FindCUDA.cmake

**/
inline const char* OKConf::PTXPath( const char* cmake_target, const char* cu_name, const char* ptxrel )
{
    std::stringstream ss ;
    ss << OKConf::OpticksInstallPrefix()
       << "/installcache/PTX/"
       ;

    if(ptxrel) ss << ptxrel << "/" ;

    ss
       << cmake_target
       << "_generated_"
       << cu_name
       << ".ptx"
       ;
    std::string ptxpath = ss.str();
    return strdup(ptxpath.c_str());
}




/**
OKConf::NvidiaDriverVersion
----------------------------

Linux only parsing of /proc/driver/nvidia/version file::

    NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  610.43.02  Release Build  (root@localhost.localdomain)  Mon Jun  1 06:02:53 PM CST 2026
    GCC version:  gcc version 11.5.0 20240719 (Red Hat 11.5.0-5) (GCC)

That looks for double dotted version string on the first line.

**/

inline const char* OKConf::NvidiaDriverVersion()
{
    std::ifstream fp("/proc/driver/nvidia/version");
    if (!fp.is_open()) return "unknown";

    std::string firstline;
    std::getline(fp, firstline);

    // Match 1 or more digits, a literal dot, 1+ digits, a literal dot, 1+ digits
    std::regex version_pattern(R"(\d+\.\d+\.\d+)");
    std::smatch match;

    bool found = std::regex_search(firstline, match, version_pattern); // equivalent to Python re.search()

    std::string version = found ? match[0].str() : "unknown-pattern-mismatch" ;
    return strdup(version.c_str());
}
#endif



inline int OKConf::Geant4VersionInteger()
{
#ifdef OKCONF_GEANT4_VERSION_INTEGER
   return OKCONF_GEANT4_VERSION_INTEGER ;
#else
   return 0 ;
#endif
}

inline const char* OKConf::OpticksInstallPrefix()
{
#ifdef OKCONF_OPTICKS_INSTALL_PREFIX
   const char* evalue = getenv("OPTICKS_INSTALL_PREFIX") ;
   return evalue ? evalue : OKCONF_OPTICKS_INSTALL_PREFIX ;
#else
   return "MISSING" ;
#endif
}

inline const char* OKConf::OpticksGitHash()
{
#ifdef OKCONF_OPTICKS_GIT_HASH
   return OKCONF_OPTICKS_GIT_HASH ;
#else
   return "OPTICKS_GIT_HASH_MISSING" ;
#endif
}


inline const char* OKConf::CMAKE_CXX_FLAGS()
{
#ifdef OKCONF_CMAKE_CXX_FLAGS
   return OKCONF_CMAKE_CXX_FLAGS ;
#else
   return "MISSING" ;
#endif
}



inline const char* OKConf::ShaderDir() // static
{
    std::stringstream ss ;
    ss << OKConf::OpticksInstallPrefix()
       << "/gl"
       ;
    std::string shaderdir = ss.str();
    return strdup(shaderdir.c_str());
}

inline const char* OKConf::DefaultSTTFPath()  // static
{
    std::stringstream ss ;
    ss << OKConf::OpticksInstallPrefix()
       << "/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf"
       ;
    std::string shaderdir = ss.str();
    return strdup(shaderdir.c_str());
}


inline int OKConf::OpticksVersionInteger()
{
#ifdef OPTICKS_VERSION_NUMBER
    return OPTICKS_VERSION_NUMBER;
#else
    return 0;
#endif
}



// ------------ WHY THE GYMNASTICS ?
// Actually no need for handling an integer macro - but there is potential utility
// for manipulation of macros containing strings that want to pass from preprocessor
// macro into strings.
//
// converts preprocessor macro into a string
#define OK_XSTR(s) OK_STR(s)  // Layer 1: Forces argument expansion
#define OK_STR(s) #s          // Layer 2: Stringifies the result - the hash is stringification operator

inline int OKConf::OpticksVersionInteger_Gymnastically()
{
    const char* s_version = OK_XSTR(OPTICKS_VERSION_NUMBER);
    int i_version = atoi(s_version);
    return i_version ;
}

#undef OK_XSTR
#undef OK_STR



