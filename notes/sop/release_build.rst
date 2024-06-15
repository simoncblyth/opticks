Release Build
==============

Debug vs Release builds
-------------------------

Release adds:: 

   PRODUCTION

Removes:: 

   DEBUG_TAG, DEBUG_PIDX,...

* debug array collection (eg photon step point records)
* debug code from GPU kernels


The compilation definitions applied to the kernel code have potential to improve 
performance significantly. By reducing code, you reduce resources that can mean 
more threads in flight. 
For more on Release optimization see:

* https://simoncblyth.bitbucket.io/env/presentation/opticks_20231211_profile.html?p=14

In that presentation I only saw only ~10% improvement from Release compared to Debug
I expect more work on this will get more.



How to configure a Release build
-----------------------------------

To do a Release build::

       #export  OPTICKS_BUILDTYPE=Debug
       export  OPTICKS_BUILDTYPE=Release
       export OPTICKS_PREFIX=/usr/local/opticks_${OPTICKS_BUILDTYPE}

I recommend you use a different prefix as mixing build types is a bad idea.  
Also it makes it fast to change between the build types.::

    epsilon:CSGOptiX blyth$ t opticks-buildtype
    opticks-buildtype ()
    {
        echo ${OPTICKS_BUILDTYPE:-Debug}
    }

    epsilon:CSGOptiX blyth$ t om-cmake
    om-cmake ()
    {
        local sdir=$1;
        local bdir=$PWD;
        [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000;
        local rc;
        cmake $sdir -G "$(om-cmake-generator)" -DCMAKE_BUILD_TYPE=$(opticks-buildtype) -DOPTICKS_PREFIX=$(om-prefix) -DCMAKE_INSTALL_PREFIX=$(om-prefix) -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules;
        rc=$?;
        return $rc
    }


The mechanics of how the CMAKE_BUILD_TYPE Release flips 
the PRODUCTION switch happens in  sysrap/CMakeLists.txt line 744 below. 
Because that definition is PUBLIC it should be impacting all pkg that 
depend on sysrap.::

    732 target_compile_definitions( ${name}
    733     PUBLIC
    734       $<$<CONFIG:Debug>:CONFIG_Debug>
    735       $<$<CONFIG:RelWithDebInfo>:CONFIG_RelWithDebInfo>
    736       $<$<CONFIG:Release>:CONFIG_Release>
    737       $<$<CONFIG:MinSizeRel>:CONFIG_MinSizeRel>
    738
    739       OPTICKS_SYSRAP
    740       WITH_CHILD
    741       PLOG_LOCAL
    742       $<$<CONFIG:Debug>:DEBUG_TAG>
    743       $<$<CONFIG:Debug>:DEBUG_PIDX>
    744       $<$<CONFIG:Release>:PRODUCTION>
    745 )
     

The CUDA compilation controlled by CSGOptiX/CMakeLists.txt
is a bit separate, but checking as shown below indicates 
the flags are felt by ptx compilation.
You can check the flags that CMake is using with::

        cx
        export VERBOSE=1
        touch CSGOptiX.cu
        om

If you see debug switches in the ptx compilation then you will need to do a 
fresh Release build.


