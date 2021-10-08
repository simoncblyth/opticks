#[=[
cmake/Modules/OpticksCXXFlags.cmake
=====================================

Geant4 1100 uses std::string_view in G4String.hh forcing 
all code that includes that header to use at least c++17 
For some years Opticks on Linux has been using c++14 
configured in cmake/Modules/OpticksCXXFlags.cmake 

Bumping to c++17 restricts the supported compilers to gcc 5+ 
which is not available by default on older redhat/centos/sl nodes.
Older compilers will give om-conf/om-cleaninstall errors like::

    CMake Error in CMakeLists.txt:
      Target "OKConf" requires the language dialect "CXX17" , but CMake does not
      know the compile flags to use to enable it.

The redhat/centos/sl workaround allowing use of a newer gcc than the OS default
is to use devtoolset and add the below in eg .bashrc/.local:: 

    devtoolset-notes(){ cat << EON
    When enabling/disabling/changing devtoolset
    ---------------------------------------------

    1. start a new session and exit the old sessions for clarity
    2. must do this after any absolute PATH settings as it prefixes PATH and LD_LIBRARY_PATH

    * https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
    * https://www.softwarecollections.org/en/scls/rhscl/devtoolset-8/

    EON
    }

    # default gcc is 4.8.5 
    #source /opt/rh/devtoolset-9/enable    ## gcc 9.3.1 : cannot be used with CUDA 10.1
    source /opt/rh/devtoolset-8/enable    ## gcc 8.3.1  : works with CUDA 10.1
    #source /opt/rh/devtoolset-7/enable    ## gcc 7.3.1 


Note that using a non-default compiler for your OS is a dangerous situation 
as vendors such as NVIDIA typically only develop packages such as CUDA/nvcc
against the default compiler for the OS.

After changing the standard or the compiler it is necessary to om-cleaninstall
and possibly do a deeper clean with  om-prefix-clean.  

okconf/tests/CPPVersionInteger.cc::

    [simon@localhost okconf]$ CPPVersionInteger
    201703

#]=]

# start from nothing, so repeated inclusion of this into CMake context doesnt repeat the flags 
set(CMAKE_CXX_FLAGS)

if(WIN32)

  # HMM: need to detect compiler not os?
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4") # overall warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd4996")   # disable  C4996: 'strdup': The POSIX name for this item is deprecated.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0")

else()

  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
     set(CMAKE_CXX_STANDARD 14)
     set(CMAKE_CXX_STANDARD_REQUIRED on)
  else ()
     #set(CMAKE_CXX_STANDARD 14)
     set(CMAKE_CXX_STANDARD 17)   ## Geant4 1100 forcing c++17 : BUT that restricts to gcc 5+ requiring 
     set(CMAKE_CXX_STANDARD_REQUIRED on)
  endif ()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden") ## avoid boostrap visibility warning at link 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-show-option") 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")

  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")
  else()
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")
  endif()

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")


endif()


if(FLAGS_VERBOSE)
   # https://cmake.org/Wiki/CMake_Useful_Variables
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_RELWITHDEBINFO= ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD : ${CMAKE_CXX_STANDARD} " )
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD_REQUIRED : ${CMAKE_CXX_STANDARD_REQUIRED} " )
endif()



