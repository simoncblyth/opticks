# === func-gen- : network/asiozmq/asiozmq fgp externals/asiozmq.bash fgn asiozmq fgh network/asiozmq
asiozmq-src(){      echo externals/asiozmq.bash ; }
asiozmq-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(asiozmq-src)} ; }
asiozmq-vi(){       vi $(asiozmq-source) ; }
asiozmq-env(){      elocal- ; }
asiozmq-usage(){ cat << EOU

Asio ZMQ 
=========

Providing the BOOST/ASIO interfaces for ZeroMQ.

* https://github.com/yayj/asio-zmq

* header only, so below functions build the examples


potential problem from c++11 requirement
--------------------------------------------

::

    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -stdlib=libc++")
    else ()
      set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
    endif ()


cmake fixes to build examples
------------------------------

cmake call needs option to locate my FindZMQ.cmake::

   -DCMAKE_MODULE_PATH=$ENV_HOME/cmake/Modules

Plus a few changes to use::

    delta:example blyth$ git diff CMakeLists.txt 
    diff --git a/example/CMakeLists.txt b/example/CMakeLists.txt
    index ff23762..d3fdb33 100644
    --- a/example/CMakeLists.txt
    +++ b/example/CMakeLists.txt
    @@ -10,17 +10,23 @@ endif ()
     add_definitions(-DBOOST_ASIO_HAS_STD_CHRONO)
     
     find_package(Boost REQUIRED COMPONENTS system)
    -find_library(ZMQ_LIBRARY zmq REQUIRED)
    +
    +#find_library(ZMQ_LIBRARY zmq REQUIRED)
    +find_package(ZMQ REQUIRED)
     
     file(GLOB example_SRCS "${CMAKE_SOURCE_DIR}/*.cpp")
     
     include_directories(
         ${CMAKE_SOURCE_DIR}/../include
         ${Boost_INCLUDE_DIRS}
    +    ${ZMQ_INCLUDE_DIRS}
         )
     
    +
    +
     foreach(SRC ${example_SRCS})
       get_filename_component(EXE ${SRC} NAME_WE)
       add_executable(${EXE} ${SRC})
    -  target_link_libraries(${EXE} ${ZMQ_LIBRARY} ${Boost_LIBRARIES})
    +  #target_link_libraries(${EXE} ${ZMQ_LIBRARY} ${Boost_LIBRARIES})
    +  target_link_libraries(${EXE} ${ZMQ_LIBRARIES} ${Boost_LIBRARIES})
     endforeach()


examples
-------------


identity
~~~~~~~~~

Output perplexing until you realise are seeing
the internal multipart structure of ZMQ messages 
with an identifier then empty, then the body.

::

    delta:example.build blyth$ ./identity 
    ----------------------------------------
    ?A?

    ROUTER uses a generated UUID
    ----------------------------------------
    PEER2

    ROUTER socket uses REQ's socket identity
    delta:example.build blyth$ 


rrbroker/rrclient/rrworker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* rrbroker

  * instantiation binds sockets and attaches async handers to invoke handle_recv

    * frontend_.bind("tcp://*:5559")   

      * rrclient REQ write_message/read_message to/from this socket)

    * backend_.bind("tcp://*:5560")    

      * rrworker REP async_read_message with handle_req which write_message back
        and then async_read_message with handle_req to keep going 

  * handle_recv

    * grabs msg into tmp via a swap
    * forwards message to the other one
    * invokes async_recv_message on the receiver to keep the ball rolling



FIXED : getting duplicate symbols with these three
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    boost::asio::error::zmq_category()
    boost::asio::error::make_error_code(boost::asio::error::zmq_error)
    std::to_string(boost::asio::zmq::frame const&)


Whats special with those ?

* symbols defined outside of class or struct definitions 
  which do not have an inline

* FIXED : by adding "inline" to those three::

    -const system::error_category& zmq_category()
    +inline const system::error_category& zmq_category()

    -system::error_code make_error_code(zmq_error e)
    +inline system::error_code make_error_code(zmq_error e)

    -std::string to_string(boost::asio::zmq::frame const& frame)
    +inline std::string to_string(boost::asio::zmq::frame const& frame)


C++ duplicate symbols at linking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* pragma once does not help, that is to avoid duplication 
  of headers within one compilation unit, not between units

* methods defined inside the class definition are implicitly 
  inline and do not cause dyplicate symbols 

* sometimes can use extern to avoid, how ?

* as above shows adding "inline" can be an easy fix sometimes








EOU
}
asiozmq-dir(){  echo $(local-base)/env/network/asiozmq ; }
asiozmq-idir(){ echo $(asiozmq-dir)/include ; }
asiozmq-sdir(){ echo $(asiozmq-dir)/example ; }
asiozmq-bdir(){ echo $(asiozmq-dir)/example.build ; }
asiozmq-edir(){ echo $(opticks-home)/network/asiozmq; }

asiozmq-cd(){   cd $(asiozmq-dir) ; }
asiozmq-scd(){  cd $(asiozmq-sdir)  ; }
asiozmq-bcd(){  cd $(asiozmq-bdir) ; }
asiozmq-icd(){  cd $(asiozmq-idir)/asio-zmq ; }
asiozmq-ecd(){  cd $(asiozmq-edir) ; }


asiozmq-get(){
   local dir=$(dirname $(asiozmq-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d asiozmq ] && git clone https://github.com/yayj/asio-zmq.git asiozmq 
}
asiozmq-wipe(){
   local bdir=$(asiozmq-bdir)  ;
   rm -rf $bdir
}
asiozmq-cmake(){
   local iwd=$PWD
   local sdir=$(asiozmq-sdir) ;
   local bdir=$(asiozmq-bdir)  ;
   mkdir -p $bdir
   asiozmq-bcd
   cmake $sdir -DCMAKE_MODULE_PATH=$ENV_HOME/cmake/Modules
   cd $iwd
}
asiozmq-make(){
   local iwd=$PWD
   asiozmq-bcd
   make $*
   cd $iwd
}
asiozmq--(){
   asiozmq-wipe
   asiozmq-cmake
   asiozmq-make
}



