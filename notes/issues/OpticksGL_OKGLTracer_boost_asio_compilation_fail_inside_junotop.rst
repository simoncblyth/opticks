OpticksGL_OKGLTracer_boost_asio_compilation_fail_inside_junotop
=================================================================

Below was pilot error due to trying to do the opticks update build 
without first getting into jre (j-runtime-env).  But the disceptive 
thing is that the update build looked just fine for many projects
before failing in okgl.


JUNOTOP/opticks okgl compilation fail for lack of boost headers
---------------------------------------------------------------------

Following a successful "bash junoenv opticks", update building JUNOTOP/opticks with oo fails::

    L7[blyth@lxslc709 opticks]$ t oo
    oo () 
    { 
        opticks-;
        cd $(opticks-home);
        om-;
        om--
    }


    102 OpticksViz::OpticksViz(OpticksHub* hub, OpticksIdx* idx, bool immediate)
    103     :
    104     SCtrl(),
    105     m_preinit(preinit()),
    106     m_log(new SLog("OpticksViz::OpticksViz", "", LEVEL)),
    107 #ifdef WITH_BOOST_ASIO
    108     m_io_context(),
    109     m_listen_udp(new BListenUDP<OpticksViz>(m_io_context,this)),   // UDP messages -> ::command calls to this
    110 #endif
    111     m_hub(hub),
    112     m_umbrella_cfg(hub->getUmbrellaCfg()),
    113     m_ok(hub->getOpticks()),




    -- Build files have been written to: /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/build/opticksgl
    Scanning dependencies of target OpticksGL
    [ 50%] Building CXX object CMakeFiles/OpticksGL.dir/OKGL_LOG.cc.o
    [ 66%] Building CXX object CMakeFiles/OpticksGL.dir/OAxisTest.cc.o
    [ 66%] Building CXX object CMakeFiles/OpticksGL.dir/OFrame.cc.o
    [ 66%] Building CXX object CMakeFiles/OpticksGL.dir/ORenderer.cc.o
    [ 83%] Building CXX object CMakeFiles/OpticksGL.dir/OKGLTracer.cc.o
    In file included from /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgl/OKGLTracer.cc:44:0:
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OGLRap/OpticksViz.hh:133:10: error: ‘io_context’ in namespace ‘boost::asio’ does not name a type
              boost::asio::io_context m_io_context ;    
              ^
    In file included from /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OGLRap/OpticksViz.hh:174:0,
                     from /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgl/OKGLTracer.cc:44:
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/BoostRap/BListenUDP.hh:46:43: error: expected ‘)’ before ‘&’ token
             BListenUDP(boost::asio::io_context& io_context, T* ctrl_);
                                               ^
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/BoostRap/BListenUDP.hh:53:9: error: ‘io_context’ in namespace ‘boost::asio’ does not name a type
             boost::asio::io_context& io_context_ ; 
             ^
    In file included from /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OGLRap/OpticksViz.hh:174:0,
                     from /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgl/OKGLTracer.cc:44:
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/BoostRap/BListenUDP.hh:86:26: error: expected constructor, destructor, or type conversion before ‘(’ token
     BListenUDP<T>::BListenUDP(boost::asio::io_context& io_context, T* ctrl_ )
                              ^
    make[2]: *** [CMakeFiles/OpticksGL.dir/OKGLTracer.cc.o] Error 1
    make[1]: *** [CMakeFiles/OpticksGL.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all make : non-zero rc 2
    === om-all om-make : ERROR bdir /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/build/opticksgl : non-zero rc 2
    === om-one-or-all make : non-zero rc 2
    L7[blyth@lxslc709 opticks]$ 


Reproduce with::

    okgl 
    om 


export VERBOSE=1::

    /usr/bin/c++ 
    -DBOOST_ALL_NO_LIB 
    -DBOOST_FILESYSTEM_DYN_LINK 
    -DBOOST_PROGRAM_OPTIONS_DYN_LINK  
    -DBOOST_REGEX_DYN_LINK 
    -DBOOST_SYSTEM_DYN_LINK 
    -DGUI_
    -DOPTICKS_BRAP
    -DOPTICKS_CUDARAP
    -DOPTICKS_GGEO
    -DOPTICKS_NPY
    -DOPTICKS_OGLRAP
    -DOPTICKS_OKCONF
    -DOPTICKS_OKCORE
    -DOPTICKS_OKGEO
    -DOPTICKS_OKGL
    -DOPTICKS_OKOP
    -DOPTICKS_OXRAP
    -DOPTICKS_SYSRAP
    -DOPTICKS_THRAP
    -DOpticksGL_EXPORTS
    -DWITH_BOOST_ASIO
    -I/hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgl
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OGLRap
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/externals/include
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OpticksGeo
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OpticksCore
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/NPY
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/externals/glm/glm
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/SysRap
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OKConf
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/BoostRap
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/externals/include/nljson
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/GGeo
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OKOP
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OptiXRap
    -isystem
    /hpcfs/juno/junogpu/blyth/local/OptiX_650/include
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/ThrustRap
    -isystem
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/CUDARap
    -isystem
    /usr/local/cuda/include
    -isystem
    /usr/local/cuda/samples/common/inc
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -fdiagnostics-show-option
    -Wall
    -Wno-unused-function
    -Wno-comment
    -Wno-deprecated
    -Wno-shadow
    -g
    -fPIC
    -std=gnu++1y
    -o
    CMakeFiles/OpticksGL.dir/OKGLTracer.cc.o
    -c
    /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgl/OKGLTracer.cc
    In file included from /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgl/OKGLTracer.cc:44:0:
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OGLRap/OpticksViz.hh:133:10: error: ‘io_context’ in namespace ‘boost::asio’ does not name a type
              boost::asio::io_context m_io_context ;    
              ^
    In file included from /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OGLRap/OpticksViz.hh:174:0,
                     from /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgl/OKGLTracer.cc:44:
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/BoostRap/BListenUDP.hh:46:43: error: expected ‘)’ before ‘&’ token
             BListenUDP(boost::asio::io_context& io_context, T* ctrl_);
                                               ^
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/BoostRap/BListenUDP.hh:53:9: error: ‘io_context’ in namespace ‘boost::asio’ does not name a type
             boost::asio::io_context& io_context_ ; 
             ^
    In file included from /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/OGLRap/OpticksViz.hh:174:0,
                     from /hpcfs/juno/junogpu/blyth/junotop/opticks/opticksgl/OKGLTracer.cc:44:
    /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/opticks/head/include/BoostRap/BListenUDP.hh:86:26: error: expected constructor, destructor, or type conversion before ‘(’ token
     BListenUDP<T>::BListenUDP(boost::asio::io_context& io_context, T* ctrl_ )
                              ^


Compare with succeeding line::


    epsilon:opticksgl blyth$ touch OKGLTracer.cc
    epsilon:opticksgl blyth$ export VERBOSE=1
    epsilon:opticksgl blyth$ om


    /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -DBOOST_ALL_NO_LIB
    -DBOOST_FILESYSTEM_DYN_LINK
    -DBOOST_PROGRAM_OPTIONS_DYN_LINK
    -DBOOST_REGEX_DYN_LINK
    -DBOOST_SYSTEM_DYN_LINK
    -DGUI_
    -DOPTICKS_BRAP
    -DOPTICKS_CUDARAP
    -DOPTICKS_GGEO
    -DOPTICKS_NPY
    -DOPTICKS_OGLRAP
    -DOPTICKS_OKCONF
    -DOPTICKS_OKCORE
    -DOPTICKS_OKGEO
    -DOPTICKS_OKGL
    -DOPTICKS_OKOP
    -DOPTICKS_OXRAP
    -DOPTICKS_SYSRAP
    -DOPTICKS_THRAP
    -DOpticksGL_EXPORTS
    -DWITH_BOOST_ASIO
    -I/Users/blyth/opticks/opticksgl
    -isystem
    /usr/local/opticks/include/OGLRap
    -isystem
    /usr/local/opticks/externals/include
    -isystem
    /usr/local/opticks/include/OpticksGeo
    -isystem
    /usr/local/opticks/include/OpticksCore
    -isystem
    /usr/local/opticks/include/NPY
    -isystem
    /usr/local/opticks/externals/glm/glm
    -isystem
    /usr/local/opticks/include/SysRap
    -isystem
    /usr/local/opticks/externals/plog/include
    -isystem
    /usr/local/opticks/include/OKConf
    -isystem
    /usr/local/opticks/include/BoostRap
    -isystem
    /usr/local/opticks_externals/boost/include
    -isystem
    /usr/local/opticks/externals/include/nljson
    -isystem
    /usr/local/opticks/include/GGeo
    -isystem
    /usr/local/opticks/include/OKOP
    -isystem
    /usr/local/opticks/include/OptiXRap
    -isystem
    /usr/local/optix/include
    -isystem
    /usr/local/opticks/include/ThrustRap
    -isystem
    /usr/local/opticks/include/CUDARap
    -isystem
    /usr/local/cuda/include
    -isystem
    /usr/local/cuda/samples/common/inc

    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -fdiagnostics-show-option
    -Wall
    -Wno-unused-function
    -Wno-unused-private-field
    -Wno-shadow
    -g
    -fPIC


    -std=gnu++14
    -o
    CMakeFiles/OpticksGL.dir/OKGLTracer.cc.o
    -c
    /Users/blyth/opticks/opticksgl/OKGLTracer.cc


Differences the /usr/local/opticks_externals/boost/include line between BoostRap and nljson::
 
    -isystem
    /usr/local/opticks/include/BoostRap
    -isystem
    /usr/local/opticks_externals/boost/include
    -isystem
    /usr/local/opticks/externals/include/nljson
    -isystem


This might be related to finding of an old system boost preventing the intended boost include dir 
from being there. 


Note that there are no opticks_externals for the cluster build. This needs to grab 
the JUNOTOP boost includes::

   /hpcfs/juno/junogpu/blyth/junotop/ExternalLibs/Boost/1.75.0/include


Curious : many other boost using Opticks projects built OK ? What is special about okgl ?

* must have been dumb luck to get so far before failing, because updates did not touch anything needing boost headers ?

 
