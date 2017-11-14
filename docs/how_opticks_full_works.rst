How opticks-full works
========================

*opticks-full* is a bash function from the opticks.bash file that does the below: 

1. installs externals *opticks-externals-install*
2. configures the build using CMake *opticks-configure*
3. invokes the build *opticks--*
4. prepares installcache *opticks-prepare-installcache*

The *opticks-vi* bash function allows you to examine/edit the functions.

opticks-full
---------------

::

    opticks-full()
    {
        local msg="=== $FUNCNAME :"
        echo $msg START $(date)
        opticks-info

        if [ ! -d "$(opticks-prefix)/externals" ]; then

            echo $msg installing the below externals into $(opticks-prefix)/externals
            opticks-externals
            opticks-externals-install


        else
            echo $msg using preexisting externals from $(opticks-prefix)/externals
        fi

        opticks-configure

        opticks--

        opticks-prepare-installcache

        echo $msg DONE $(date)
    }


opticks-configure
---------------------

*opticks-configure* wipes the build directory and then invokes *opticks-configure-system-boost* 
which in turn calls *opticks-cmake*

::

    opticks-wipe(){
      local msg="=== $FUNCNAME : "
       local bdir=$(opticks-bdir)
       echo $msg wiping build dir $bdir
       rm -rf $bdir
    }

    opticks-configure()
    {
       opticks-wipe

       case $(opticks-cmake-generator) in
           "Visual Studio 14 2015") opticks-configure-local-boost $* ;;
                                 *) opticks-configure-system-boost $* ;;
       esac
    }

    opticks-configure-system-boost()
    {
       opticks-cmake $*
    }


opticks-cmake
----------------

::

    opticks-cmake(){
       local msg="=== $FUNCNAME : "
       local iwd=$PWD
       local bdir=$(opticks-bdir)

       echo $msg configuring installation

       mkdir -p $bdir
       [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use opticks-configure to wipe build dir and re-configure && return

       opticks-bcd

       g4-
       xercesc-

       opticks-cmake-info

       cmake \
            -G "$(opticks-cmake-generator)" \
           -DCMAKE_BUILD_TYPE=Debug \
           -DCOMPUTE_CAPABILITY=$(opticks-compute-capability) \
           -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
           -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
           -DGeant4_DIR=$(g4-cmake-dir) \
           -DXERCESC_LIBRARY=$(xercesc-library) \
           -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
           $* \
           $(opticks-sdir)

       cd $iwd
    }



opticks-cmake-info
---------------------

*opticks-cmake-info* dumps the values of the variables that are input to cmake

::

    opticks-cmake-info(){ cat << EOI

    $FUNCNAME
    ======================

           NODE_TAG                   :  $NODE_TAG

           opticks-sdir               :  $(opticks-sdir)
           opticks-bdir               :  $(opticks-bdir)
           opticks-cmake-generator    :  $(opticks-cmake-generator)
           opticks-compute-capability :  $(opticks-compute-capability)
           opticks-prefix             :  $(opticks-prefix)
           opticks-optix-install-dir  :  $(opticks-optix-install-dir)
           g4-cmake-dir               :  $(g4-cmake-dir)
           xercesc-library            :  $(xercesc-library)
           xercesc-include-dir        :  $(xercesc-include-dir)

    EOI
    }


Example of running::

    simon:docs blyth$ opticks-cmake-info 

    opticks-cmake-info
    ======================

           NODE_TAG                   :  D

           opticks-sdir               :  /Users/blyth/opticks
           opticks-bdir               :  /usr/local/opticks/build
           opticks-cmake-generator    :  Unix Makefiles
           opticks-compute-capability :  30
           opticks-prefix             :  /usr/local/opticks
           opticks-optix-install-dir  :  /Developer/OptiX_380
           g4-cmake-dir               :  /usr/local/opticks/externals/lib/Geant4-10.2.1
           xercesc-library            :  /opt/local/lib/libxerces-c.dylib
           xercesc-include-dir        :  /opt/local/include

    simon:docs blyth$ 


All of the input variables come from other bash functions such as, 

1. *opticks-optix-install-dir*
2. *opticks-compute-capability*

These yield different results depending on the setting of the NODE_TAG envvar.


::

    opticks-optix-install-dir(){
        local t=$NODE_TAG
        case $t in
           D_400) echo /Developer/OptiX_400 ;;
           D) echo /Developer/OptiX_380 ;;
        RYAN) echo /Developer/OptiX_380 ;;
         GTL) echo ${MYENVTOP}/OptiX ;;
        H5H2) echo ${MYENVTOP}/OptiX ;;
           X) echo /usr/local/optix-3.8.0/NVIDIA-OptiX-SDK-3.8.0-linux64 ;;
        #SDUGPU) echo /root/NVIDIA-OptiX-SDK-4.1.1-linux64 ;;
        SDUGPU) echo /home/simon/NVIDIA-OptiX-SDK-4.1.1-linux64 ;;
           *) echo /tmp ;;
        esac
    }


    opticks-compute-capability(){
        local t=$NODE_TAG
        case $t in
           D) echo 30 ;;
        RYAN) echo 30 ;;
         GTL) echo 30 ;;
        H5H2) echo 50 ;;
           X) echo 52 ;;
      SDUGPU) echo 30 ;;
           *) echo  0 ;;
        esac
    }



