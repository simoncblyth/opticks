NLJSON_review
==============


Current Opticks makes little use of JSON, but still sysrap/CMakeLists.txt::

     077 find_package(NLJSON REQUIRED MODULE)
     ...
     733 if(NLJSON_FOUND)
     734    message(STATUS "SYSRAP.NLJSON_FOUND")
     735    list(APPEND SOURCES  SMeta.cc)
     736    list(APPEND HEADERS  SMeta.hh)
     737 endif()
     ...
     794 if(NLJSON_FOUND)
     795 target_link_libraries(${name}  Opticks::NLJSON)
     796 endif()


ITS HEADER ONLY - LINK LINE NOT NEEDED

* traditionally yes, but CMake targets can do more than just linking

  * they carry things like INTERFACE_INCLUDE_DIRECTORIES


::

    [lo] A[blyth@localhost notes]$ opticks-fl NLJSON
    ./cmake/Modules/FindNLJSON.cmake
    ./cmake/CMakeLists.txt
    ./examples/UseNLJSON/CMakeLists.txt
    ./examples/UseNLJSON/UseNLJSON.cc
    ./externals/nljson.bash
    ./npy/NMeta.hpp
    ./sysrap/CMakeLists.txt
    ./sysrap/tests/CMakeLists.txt



cmake/Modules/FindNLJSON.cmake::

    set(NLJSON_MODULE "${CMAKE_CURRENT_LIST_FILE}")
    #set(NLJSON_VERBOSE OFF)

    find_path(
        NLJSON_INCLUDE_DIR
        NAMES "json.hpp"
        PATHS "${OPTICKS_PREFIX}/externals/include/nljson"
    )

    if(NLJSON_INCLUDE_DIR)
      set(NLJSON_FOUND "YES")
    else()
      set(NLJSON_FOUND "NO")
    endif()


    if(NLJSON_VERBOSE OR NOT NLJSON_FOUND)
      message(STATUS "OPTICKS_PREFIX           : ${OPTICKS_PREFIX}")
      message(STATUS "NLJSON_MODULE            : ${NLJSON_MODULE}")
      message(STATUS "NLJSON_INCLUDE_DIR       : ${NLJSON_INCLUDE_DIR} ")
      message(STATUS "NLJSON_FOUND             : ${NLJSON_FOUND}")
    endif()

    if(NOT NLJSON_FOUND)
      message(FATAL_ERROR "NLJSON NOT FOUND")
    endif()

    set(_tgt Opticks::NLJSON)
    if(NLJSON_FOUND AND NOT TARGET ${_tgt})
        add_library(${_tgt} INTERFACE IMPORTED)
        set_target_properties(${_tgt} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NLJSON_INCLUDE_DIR}"
            INTERFACE_PKG_CONFIG_NAME "NLJSON"
        )
        set(NLJSON_targets "NLJSON")
    endif()

::

    [lo] A[blyth@localhost opticks]$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    plog
    nljson


::

    nljson-vi(){       vi $BASH_SOURCE ; }
    nljson-usage(){ cat << EOU

    * https://github.com/nlohmann/json
    * https://github.com/nlohmann/json#json-as-first-class-data-type

    EOU
    }

    nljson-env(){ olocal- ;  }
    nljson-url(){ echo https://github.com/nlohmann/json/releases/download/v3.9.1/json.hpp ; }

    nljson-prefix(){ echo $(opticks-prefix)/externals ; }
    nljson-path(){   echo $(opticks-prefix)/externals/include/nljson/json.hpp ; }
    nljson-dist(){   echo $(nljson-path) ; }
    nljson-get()
    {
       local msg="=== $FUNCNAME :"
       local iwd=$PWD
       local path=$(nljson-path)
       local dir=$(dirname $path) &&  mkdir -p $dir && cd $dir

       local url=$(nljson-url)
       local name=$(basename $url)

       [ ! -s "$name" ] && opticks-curl $url
       [ ! -s "$name" ] && echo $msg FAILED TO DOWNLOAD $name

       cd $iwd

       [ -s "$path" ]   # set rc
    }

    nljson--(){
       nljson-get
       #nljson-pc
    }

    nljson-r(){ vim -R $(nljson-path) ; }


    nljson-pc-(){ cat << EOP




