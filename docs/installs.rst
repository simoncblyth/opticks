Opticks Installations
========================



Example .bash_profile
--------------------------

::

        # .bash_profile

        # Get the aliases and functions
        if [ -f ~/.bashrc ]; then
                . ~/.bashrc
        fi

        # User specific environment and startup programs

        PATH=$PATH:$HOME/bin



        env-(){     . $HOME/env/env.bash && env-env $* ; }
        opticks-(){ . $HOME/opticks/opticks.bash && opticks-env $* ; }

        alias vip='vi ~/.bash_profile'
        alias ini='. ~/.bash_profile'
        alias t='typeset -f'
        alias o='cd ~/opticks ; hg status'
        alias l='ls -l'
        alias h='history'

        export LOCAL_BASE=/usr/local 
        export OPTICKS_HOME=$HOME/opticks
        export NODE_TAG=SDUGPU
        export PYTHONPATH=$HOME

        PATH=$PATH:$LOCAL_BASE/opticks/lib:$OPTICKS_HOME/bin

        op(){ op.sh $* ; }


        export PATH






SDU GPU Notes
----------------


::

    op --j1707 --gdml2gltf 
         # once only for a geometry conversion 
         # convert g4_00.gdml GDML into serialized OpticksCSG trees coordinated by g4_00.gltf 

    op --j1707 --gltf 3 --G



::


        2017-09-15 15:48:53.103 INFO  [111500] [OFunc::convert@28] OFunc::convert ptxname solve_callable.cu.ptx ctxname solve_callable funcnames  SolveCubicCallable num_funcs 1
        [New Thread 0x7fffec9eb700 (LWP 111505)]
        [New Thread 0x7fffebfea700 (LWP 111506)]
        [New Thread 0x7fffeb5e9700 (LWP 111507)]
        [New Thread 0x7fffeabe8700 (LWP 111508)]
        2017-09-15 15:48:53.971 INFO  [111500] [OContext::createProgram@190] OContext::createProgram START  filename solve_callable.cu.ptx progname SolveCubicCallable
        2017-09-15 15:48:53.971 VERB  [111500] [OConfig::createProgram@65] OConfig::createProgram path /usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx
        2017-09-15 15:48:53.971 DEBUG [111500] [OConfig::createProgram@71] OConfig::createProgram /usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx:SolveCubicCallable
        terminate called after throwing an instance of 'optix::Exception'
          what():  Parse error (Details: Function "RTresult _rtProgramCreateFromPTXFile(RTcontext, const char*, const char*, RTprogram_api**)" caught exception: /usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx: error: Failed to parse input PTX string
        /usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx, line 10; fatal   : Unsupported .target 'sm_37'
        Cannot parse input PTX string
        )


        (gdb) bt
        #0  0x000000356a432925 in raise () from /lib64/libc.so.6
        #1  0x000000356a434105 in abort () from /lib64/libc.so.6
        #2  0x000000382ecbea8d in __gnu_cxx::__verbose_terminate_handler() () from /usr/lib64/libstdc++.so.6
        #3  0x000000382ecbcbe6 in ?? () from /usr/lib64/libstdc++.so.6
        #4  0x000000382ecbcc13 in std::terminate() () from /usr/lib64/libstdc++.so.6
        #5  0x000000382ecbcd32 in __cxa_throw () from /usr/lib64/libstdc++.so.6
        #6  0x00007ffff37eb159 in optix::ContextObj::checkError (this=0xbb75b50, code=RT_ERROR_INVALID_SOURCE) at /home/simon/NVIDIA-OptiX-SDK-4.1.1-linux64/include/optixu/optixpp_namespace.h:1831
        #7  0x00007ffff37edaf2 in optix::ContextObj::createProgramFromPTXFile (this=0xbb75b50, filename="/usr/local/opticks/installcache/PTX/OptiXRap_generated_solve_callable.cu.ptx", 
            program_name="SolveCubicCallable") at /home/simon/NVIDIA-OptiX-SDK-4.1.1-linux64/include/optixu/optixpp_namespace.h:2165
        #8  0x00007ffff37ef0b7 in OConfig::createProgram (this=0xbb33df0, filename=0xbc54190 "solve_callable.cu.ptx", progname=0xbc541e8 "SolveCubicCallable")
            at /home/simon/opticks/optixrap/OConfig.cc:72
        #9  0x00007ffff37f41c6 in OContext::createProgram (this=0xba6f2e0, filename=0xbc54190 "solve_callable.cu.ptx", progname=0xbc541e8 "SolveCubicCallable")
            at /home/simon/opticks/optixrap/OContext.cc:195
        #10 0x00007ffff38174c8 in OFunc::convert (this=0xbc53f80) at /home/simon/opticks/optixrap/OFunc.cc:41
        #11 0x00007ffff3805c6e in OScene::init (this=0xb86c800) at /home/simon/opticks/optixrap/OScene.cc:117
        #12 0x00007ffff380582b in OScene::OScene (this=0xb86c800, hub=0x663f30) at /home/simon/opticks/optixrap/OScene.cc:84
        #13 0x00007ffff2cea262 in OpEngine::OpEngine (this=0xb5d39d0, hub=0x663f30) at /home/simon/opticks/okop/OpEngine.cc:44
        #14 0x00007ffff2cee5b4 in OpPropagator::OpPropagator (this=0xb9209a0, hub=0x663f30, idx=0xb91da80) at /home/simon/opticks/okop/OpPropagator.cc:39
        #15 0x00007ffff2ceb3c3 in OpMgr::OpMgr (this=0x7fffffffda50, argc=13, argv=0x7fffffffdba8, argforced=0x0) at /home/simon/opticks/okop/OpMgr.cc:82
        #16 0x0000000000400b08 in main (argc=13, argv=0x7fffffffdba8) at /home/simon/opticks/okop/tests/OpTest.cc:10
        (gdb) 



        2017-09-15 15:59:28.030 INFO  [111945] [OGeo::convert@169] OGeo::convert START  numMergedMesh: 5
        mm i   0 geocode   A                  numSolids     290276 numFaces        9392 numITransforms           1 numITransforms*numSolids      290276
        mm i   1 geocode   A                  numSolids          5 numFaces        1584 numITransforms       36572 numITransforms*numSolids      182860
        mm i   2 geocode   A                  numSolids          6 numFaces        4704 numITransforms       17739 numITransforms*numSolids      106434
        mm i   3 geocode   A                  numSolids          1 numFaces         192 numITransforms         480 numITransforms*numSolids         480
        mm i   4 geocode   A                  numSolids          1 numFaces        1856 numITransforms         480 numITransforms*numSolids         480
         num_total_volumes 290276 num_instanced_volumes 290254 num_global_volumes 22
        2017-09-15 15:59:28.030 INFO  [111945] [OContext::createProgram@190] OContext::createProgram START  filename intersect_analytic.cu.ptx progname intersect
        2017-09-15 15:59:28.030 VERB  [111945] [OConfig::createProgram@65] OConfig::createProgram path /usr/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic.cu.ptx
        2017-09-15 15:59:28.030 DEBUG [111945] [OConfig::createProgram@71] OConfig::createProgram /usr/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic.cu.ptx:intersect
        terminate called after throwing an instance of 'optix::Exception'
          what():  Parse error (Details: Function "RTresult _rtProgramCreateFromPTXFile(RTcontext, const char*, const char*, RTprogram_api**)" caught exception: /usr/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic.cu.ptx: error: Failed to parse input PTX string
        /usr/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic.cu.ptx, line 10; fatal   : Unsupported .target 'sm_37'
        Cannot parse input PTX string
        )

        Program received signal SIGABRT, Aborted.
        0x000000356a432925 in raise () from /lib64/libc.so.6
        Missing separate debuginfos, use: debuginfo-install glibc-2.12-1.132.el6.x86_64 keyutils-libs-1.4-4.el6.x86_64 keyutils-libs-1.4-5.el6.x86_64 krb5-libs-1.10.3-10.el6_4.6.x86_64 krb5-libs-1.10.3-65.el6.x86_64 libcom_err-1.41.12-18.el6.x86_64 libcom_err-1.41.12-23.el6.x86_64 libgcc-4.4.7-17.el6.x86_64 libgcc-4.4.7-18.el6.x86_64 libselinux-2.0.94-5.3.el6_4.1.x86_64 libselinux-2.0.94-7.el6.x86_64 libstdc++-4.4.7-17.el6.x86_64 libstdc++-4.4.7-18.el6.x86_64 openssl-1.0.1e-57.el6.x86_64 zlib-1.2.3-29.el6.x86_64
        (gdb) 



    -- Generating /usr/local/opticks/build/optixrap/OptiXRap_generated_generate.cu.ptx
    /usr/local/cuda-8.0/bin/nvcc /home/simon/opticks/optixrap/cu/generate.cu -ptx -o /usr/local/opticks/build/optixrap/OptiXRap_generated_generate.cu.ptx -ccbin /opt/rh/devtoolset-2/root/usr/bin/cc -m64 -Xcompiler -fPIC -gencode=arch=compute_37,code=sm_37 -std=c++11 -O2 --use_fast_math -DNVCC -I/usr/local/cuda-8.0/include -I/home/simon/opticks/optixrap -I/usr/local/opticks/externals/plog/include -I/usr/local/include -I/home/simon/opticks/sysrap -I/home/simon/opticks/boostrap -I/usr/local/opticks/externals/glm/glm -I/home/simon/opticks/opticksnpy -I/home/simon/opticks/optickscore -I/home/simon/NVIDIA-OptiX-SDK-4.1.1-linux64/include -I/usr/local/cuda-8.0/include -I/usr/local/opticks/externals/include -I/home/simon/opticks/assimprap -I/home/simon/opticks/ggeo -I/home/simon/opticks/opticksgeo -I/home/simon/opticks/cudarap -I/home/simon/opticks/thrustrap
    /home/simon/NVIDIA-OptiX-SDK-4.1.1-linux64/include/optixu/optixpp_namespace.h(590): warning: overloaded virtual function "optix::APIObj::checkError" is only partially overridden in class "optix::ContextObj"


