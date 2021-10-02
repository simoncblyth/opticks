opticks_setup_cannot_be_moved
===============================

::

    (base) [simon@localhost ~]$ echo $OPTICKS_PREFIX
    /data/simon/local/opticks
    (base) [simon@localhost ~]$ opticks-setup
    === opticks-setup.sh : build time OPTICKS_PREFIX /home/simon/local/opticks is not consistent with HERE_OPTICKS_PREFIX /data/simon/local/opticks
    === opticks-setup.sh : opticks setup scripts cannot be moved
    (base) [simon@localhost ~]$ echo $OPTICKS_PREFIX
    /home/simon/local/opticks
    (base) [simon@localhost ~]$ 

    (base) [simon@localhost ~]$ echo $OPTICKS_PREFIX
    /home/simon/local/opticks
    (base) [simon@localhost ~]$ opticks-setup-generate
    === opticks-check-compute-capability : OPTICKS_COMPUTE_CAPABILITY 70 : looking good it is an integer expression of 30 or more
    === opticks-check-geant4 : ERROR no g4_prefix : failed to find Geant4Config.cmake along CMAKE_PREFIX_PATH
    (base) [simon@localhost ~]$ 
    (base) [simon@localhost ~]$ echo $OPTICKS_PREFIX
    /home/simon/local/opticks
    (base) [simon@localhost ~]$ 


    (base) [simon@localhost ~]$ echo $OPTICKS_PREFIX
    /home/simon/local/opticks
    (base) [simon@localhost ~]$ vip
    4 files to edit
    (base) [simon@localhost ~]$ ini
    (base) [simon@localhost ~]$ opticks-setup-generate
    === opticks-check-compute-capability : OPTICKS_COMPUTE_CAPABILITY 70 : looking good it is an integer expression of 30 or more
    === opticks-setup-generate : writing /data/simon/local/opticks/bin/opticks-setup.sh
    === opticks-setup-generate : post opticks-setup-hdr- rc 0
    === opticks-setup-generate : post opticks-setup-geant4- rc 0
    === opticks-externals-setup
    === opticks-ext-setup : bcm
    === opticks-ext-setup : glm
    === opticks-ext-setup : glfw
    === opticks-ext-setup : glew
    === opticks-ext-setup : gleq
    === opticks-ext-setup : imgui
    === opticks-ext-setup : plog
    === opticks-ext-setup : opticksaux
    === opticks-ext-setup : nljson
    === opticks-setup-generate : post opticks-externals-setup rc 0
    === opticks-preqs-setup
    === opticks-ext-setup : cuda
    === opticks-ext-setup : optix
    === opticks-setup-generate : post opticks-preqs-setup rc 0
    (base) [simon@localhost ~]$ 

    (base) [simon@localhost ~]$ echo $OPTICKS_PREFIX
    /data/simon/local/opticks



om-conf fail::

    -- Configuring done
    CMake Error in CMakeLists.txt:
      Imported target "Opticks::SysRap" includes non-existent path

        "/home/simon/local/opticks/externals/glm/glm"

      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:

      * The path was deleted, renamed, or moved to another location.

      * An install or uninstall procedure did not complete successfully.

      * The installation package was faulty and references files it does not
      provide.



Take a look at the exported targets, they contain absolute paths and hence are not movable::

    (base) [simon@localhost opticks]$ cd $OPTICKS_PREFIX/lib/cmake/sysrap
    -bash: cd: /data/simon/local/opticks/lib/cmake/sysrap: No such file or directory
    (base) [simon@localhost opticks]$ cd $OPTICKS_PREFIX/lib64/cmake/sysrap
    (base) [simon@localhost sysrap]$ grep glm *.cmake
    sysrap-targets.cmake:  INTERFACE_INCLUDE_DIRECTORIES "/home/simon/local/opticks/externals/glm/glm;${_IMPORT_PREFIX}/include/SysRap"
    (base) [simon@localhost sysrap]$ 

::

    (base) [simon@localhost sysrap]$ l $OPTICKS_PREFIX/lib64/cmake/
    total 0
    drwxrwxr-x. 2 simon simon 179 Oct  1 22:00 boostrap
    drwxrwxr-x. 2 simon simon 159 Oct  1 22:00 cfg4
    drwxrwxr-x. 2 simon simon 174 Oct  1 22:00 cudarap
    drwxrwxr-x. 2 simon simon 164 Oct  1 22:00 extg4
    drwxrwxr-x. 2 simon simon 159 Oct  1 22:00 g4ok
    drwxrwxr-x. 2 simon simon 159 Oct  1 22:00 ggeo
    drwxrwxr-x. 2 simon simon 154 Oct  1 22:00 npy
    drwxrwxr-x. 2 simon simon 169 Oct  1 22:00 oglrap
    drwxrwxr-x. 2 simon simon 149 Oct  1 22:00 ok
    drwxrwxr-x. 2 simon simon 169 Oct  1 22:00 okconf
    drwxrwxr-x. 2 simon simon 159 Oct  1 22:00 okg4
    drwxrwxr-x. 2 simon simon 159 Oct  1 22:00 okop
    drwxrwxr-x. 2 simon simon 194 Oct  1 22:00 optickscore
    drwxrwxr-x. 2 simon simon 189 Oct  1 22:00 opticksgeo
    drwxrwxr-x. 2 simon simon 184 Oct  1 22:00 opticksgl
    drwxrwxr-x. 2 simon simon 179 Oct  1 22:00 optixrap
    drwxrwxr-x. 2 simon simon 169 Oct  1 22:00 sysrap
    drwxrwxr-x. 2 simon simon 184 Oct  1 22:00 thrustrap
    drwxrwxr-x. 2 simon simon 169 Oct  1 22:00 useglm

    (base) [simon@localhost sysrap]$ rm -rf $OPTICKS_PREFIX/lib64/cmake


Hence have to blow away the OPTICKS_PREFIX and rerun *opticks-full* to get the externals again.

