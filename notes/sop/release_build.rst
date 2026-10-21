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

To do a Release build, adjust some envvars::

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



How to flip the config with the local.sh machinery ? ~/j/opticks_config.sh
-----------------------------------------------------------------------------

1. change default config in ~/j/opticks_config.sh to "Release_g411"
2. get into environmnent and do full build::

   lo
   oid          ## dump identity vars including OPTICKS_PREFIX
   opticks-full

3. to avoid confusion (such as building a pkg in an old tab and not impacting the downstream pkg in new tab)
   exit all tabs with the old config


::

     01 # .bash_profile > .bashrc > .opticks_config   (workstation/simon)
      2 usage(){ cat << EOU
      3 ~/j/opticks_config.sh
      4 ========================
      5 
      6 Caution this is now sourced first - with almost no environment - it constitutes
      7 the inputs that control the opticks build
      8 
      9 Build and usage config, build config should provide the environment to complete opticks-full.
     10 Note how sourcing the bashrc rather than using opticks-prepend-prefix
     11 allows more intuitive ordering.
     12 
     13 Q: What sources this ?
     14 A: ~/j/local.sh:local_OPTICKS_CONFIG bash func which is used from local_ok_build and local_oj_build_setup
     15 
     16 EOU
     17 }
     18 
     19 
     20 ### TOP CONFIG RELEVANT TO BOTH BUILD AND USAGE ###
     21 
     22 export OPTICKS_HOME=$HOME/opticks
     23 
     24 ## HOP : TO DIFFERENT CONFIG/BUILD_DIR/INSTALL_DIR BY CHANGING THIS - AND OPENING NEW SESSION
     25 #config=Debug        ## gcc11
     26 config=Debug_g411   ## gcc15
     27 
     28 export OPTICKS_CONFIG=${OPTICKS_CONFIG:-$config}
     29 : NB THIS ALSO CONTROLS JUNO_OPTICKS_CONFIG SEE ~/j/local.sh:local_oj_build_setup
     30 
     31 buildtype=Debug
     32 
     33 if [[ "$OPTICKS_CONFIG" =~ Debug ]]; then
     34     buildtype=Debug
     35 elif [[ "$OPTICKS_CONFIG" =~ Client ]]; then
     36     buildtype=Debug
     37 elif [[ "$OPTICKS_CONFIG" =~ Release ]]; then
     38     buildtype=Release
     39 else
     40     buildtype=ERROR_UNEXPECTED_OPTICKS_CONFIG
     41 fi
     42 
     43 
     44 [   -z "$VIP_LOCAL_BASE" ] && echo $BASH_SOURCE - FATAL MISSING VIP_LOCAL_BASE $VIP_LOCAL_BASE && return 1
     45 [ ! -d "$VIP_LOCAL_BASE" ] && echo $BASH_SOURCE - FATAL NON-EXISTING VIP_LOCAL_BASE $VIP_LOCAL_BASE && return 1
     46 
     47 export OPTICKS_BUILDTYPE=$buildtype
     48 export OPTICKS_PREFIX=$VIP_LOCAL_BASE/local/opticks_${OPTICKS_CONFIG}    # Apr23 2026 : Generalized from OPTICKS_BUILDTYPE to OPTICKS_CONFIG
     49 



::

    [lo] A[blyth@localhost sop]$ t opticks-buildtype
    opticks-buildtype () 
    { 
        echo ${OPTICKS_BUILDTYPE:-Debug}
    }

    [lo] A[blyth@localhost sop]$ t opticks-prefix
    opticks-prefix () 
    { 
        echo ${OPTICKS_PREFIX:-/usr/local/opticks}
    }






Whats the difference between Debug and Release builds
------------------------------------------------------

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


