glew-is-only-external-other-that-geant4-installing-into-lib64
==================================================================


Context
----------

* :doc:`packaging-opticks-and-externals-for-use-on-gpu-cluster`


Issue
--------

* only G4 libs and GLEW are installed in externals/lib64/
* everything else goes into externals/lib/

* DONE : make the split cleaner, move glew into "externals/lib"


::

   externals/lib64/

    [blyth@localhost opticks]$ l externals/lib64/
    total 469264
    drwxrwxr-x. 3 blyth blyth       194 Aug 29  2018 Geant4-10.4.2
    -rwxr-xr-x. 1 blyth blyth   2904120 Aug 17  2018 libG4GMocren.so
    ...
    -rwxr-xr-x. 1 blyth blyth   4648040 Aug 17  2018 libG4clhep.so
    drwxrwxr-x. 3 blyth blyth       203 Jul  6  2018 Geant4-10.2.1
    drwxr-xr-x. 2 blyth blyth        21 Jul  5  2018 pkgconfig
    -rw-r--r--. 1 blyth blyth   1237814 Jul  5  2018 libGLEW.a
    lrwxrwxrwx. 1 blyth blyth        17 Jul  5  2018 libGLEW.so -> libGLEW.so.1.13.0
    lrwxrwxrwx. 1 blyth blyth        17 Jul  5  2018 libGLEW.so.1.13 -> libGLEW.so.1.13.0
    -rw-r--r--. 1 blyth blyth    773480 Jul  5  2018 libGLEW.so.1.13.0


    [blyth@localhost opticks]$ l externals/lib/*.so
    -rwxr-xr-x. 1 blyth blyth  222680 Apr 10 16:06 externals/lib/libDualContouringSample.so
    -rwxr-xr-x. 1 blyth blyth  539064 Apr 10 16:05 externals/lib/libImplicitMesher.so
    -rwxr-xr-x. 1 blyth blyth 1043096 Jul  7  2018 externals/lib/libImGui.so
    -rwxr-xr-x. 1 blyth blyth 7446072 Jul  5  2018 externals/lib/libYoctoGL.so
    lrwxrwxrwx. 1 blyth blyth      23 Jul  5  2018 externals/lib/libOpenMeshTools.so -> libOpenMeshTools.so.6.3
    lrwxrwxrwx. 1 blyth blyth      22 Jul  5  2018 externals/lib/libOpenMeshCore.so -> libOpenMeshCore.so.6.3
    lrwxrwxrwx. 1 blyth blyth      14 Jul  5  2018 externals/lib/libassimp.so -> libassimp.so.3


Note this means must cleaninstall all subs using GLEW which are oglrap and above, ie oglrap:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::
 
   cd ~/opticks
   opticks-deps   # list dependencies

   om-visit oglrap:    # just to list the projects are about to clean install 
   om-cleaninstall oglrap:





glew, what controls the lib64 with install ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* glew is installed by a Makefile which includes config/Makefile.linux which sets LIBDIR
  used by the install

* https://stackoverflow.com/questions/49626164/make-install-installs-libraries-in-usr-lib-instead-of-usr-lib64


Unchanged::

    [blyth@localhost glew-1.13.0]$ glew--
    install -d -m 0755 "/home/blyth/local/opticks/externals/include/GL"
    install -m 0644 include/GL/wglew.h "/home/blyth/local/opticks/externals/include/GL/"
    install -m 0644 include/GL/glew.h "/home/blyth/local/opticks/externals/include/GL/"
    install -m 0644 include/GL/glxew.h "/home/blyth/local/opticks/externals/include/GL/"
    sed \
        -e "s|@prefix@|/home/blyth/local/opticks/externals|g" \
        -e "s|@libdir@|/home/blyth/local/opticks/externals/lib64|g" \
        -e "s|@exec_prefix@|/home/blyth/local/opticks/externals/bin|g" \
        -e "s|@includedir@|/home/blyth/local/opticks/externals/include/GL|g" \
        -e "s|@version@|1.13.0|g" \
        -e "s|@cflags@||g" \
        -e "s|@libname@|GLEW|g" \
        -e "s|@requireslib@|glu|g" \
        < glew.pc.in > glew.pc
    install -d -m 0755 "/home/blyth/local/opticks/externals/lib64"
    install -m 0644 lib/libGLEW.so.1.13.0 "/home/blyth/local/opticks/externals/lib64/"
    ln -sf libGLEW.so.1.13.0 "/home/blyth/local/opticks/externals/lib64/libGLEW.so.1.13"
    ln -sf libGLEW.so.1.13.0 "/home/blyth/local/opticks/externals/lib64/libGLEW.so"
    install -m 0644 lib/libGLEW.a "/home/blyth/local/opticks/externals/lib64/"
    install -d -m 0755 "/home/blyth/local/opticks/externals/lib64"
    install -d -m 0755 "/home/blyth/local/opticks/externals/lib64/pkgconfig"
    install -m 0644 glew.pc "/home/blyth/local/opticks/externals/lib64/pkgconfig/"
    [blyth@localhost glew]$ 


With LIBDIR override works::

    [blyth@localhost glew]$ t glew-make
    glew-make is a function
    glew-make () 
    { 
        local target=${1:-install};
        local iwd=$PWD;
        glew-scd;
        local gen=$(opticks-cmake-generator);
        case $gen in 
            "Visual Studio 14 2015")
                glew-install-win
            ;;
            *)
                make $target GLEW_PREFIX=$(glew-prefix) GLEW_DEST=$(glew-prefix) LIBDIR=$(glew-prefix)/lib
            ;;
        esac;
        cd_func $iwd
    }


Manually remove from lib64::

    [blyth@localhost lib64]$ rm libGLEW.so libGLEW.so.1.13 libGLEW.so.1.13.0 libGLEW.a pkgconfig/glew.pc
    [blyth@localhost lib64]$ rm -rf pkgconfig/




