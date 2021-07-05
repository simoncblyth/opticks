somehow_spontaneous_missing_symbol_G4MTHepRandom_getTheEngine
================================================================

All of a sudden (some enviroment change perhaps?) X4 not compiling with.::

    [ 71%] Linking CXX executable X4GDMLParserTest
    [ 72%] Linking CXX executable X4GDMLBalanceTest
    ../libExtG4.so: undefined reference to `G4MTHepRandom::getTheEngine()'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/X4DumpTest] Error 1
    make[1]: *** [tests/CMakeFiles/X4DumpTest.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....
    ../libExtG4.so: undefined reference to `G4MTHepRandom::getTheEngine()'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/X4EntityTest] Error 1
    make[1]: *** [tests/CMakeFiles/X4EntityTest.dir/all] Error 2

Something has flipped an MT switch.

::

    O[blyth@localhost lib64]$ pwd
    /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64
    O[blyth@localhost lib64]$ nm libG4global.so | c++filt | grep G4MTHepRandom
    O[blyth@localhost lib64]$ 


/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/Geant4-10.4.2/Geant4Config.cmake::

    051 # The components available generally correspond to configurations of
     52 # the Geant4 libraries or optional extras that Geant4 can be built with.
     53 #
     54 # Library Configuration
     55 # ---------------------
     56 #  static            (Static libraries available. Using this component
     57 #                     when static libraries are available will result in
     58 #                     Geant4_LIBRARIES being populated with the static
     59 #                     versions of the Geant4 libraries. It does not
     60 #                     guarantee the use of static third party libraries.)
     61 #  multithreaded     (Libraries have multithreading support. Using this
     62 #                     component will add the compiler definitions needed
     63 #                     to activate multithread mode to Geant4_DEFINITIONS,
     64 #                     if the libraries support it.)
     65 #
     66 #  usolids           (Geant4 solids are replaced with USolids equivalents)
     67 #


    311 # - Multithreading
    312 set(Geant4_multithreaded_FOUND ON)
    313 if(Geant4_multithreaded_FOUND)
    314   list(REMOVE_ITEM Geant4_FIND_COMPONENTS multithreaded)
    315   list(APPEND Geant4_DEFINITIONS -DG4MULTITHREADED)
    316 
    317   # - Define variable to indicate TLS model used
    318   set(Geant4_TLS_MODEL "global-dynamic")
    319 endif()
    320 



Maybe JUNO upstream has flipped the switch, causing inconsistency
-----------------------------------------------------------------------

* Did you use the flag -DGEANT4_BUILD_MULTITHREADED=ON when building Geant4 itself



::

    137 
    138 function juno-ext-libs-geant4-conf-10 {
    139     local msg="===== $FUNCNAME: "
    140     cmake .. -DCMAKE_INSTALL_PREFIX=$(juno-ext-libs-geant4-install-dir) \
    141         -DGEANT4_USE_GDML=ON \
    142         -DGEANT4_INSTALL_DATA=ON \
    143         -DGEANT4_USE_OPENGL_X11=ON \
    144         -DGEANT4_USE_RAYTRACER_X11=ON \
    145         -DGEANT4_BUILD_MULTITHREADED=ON \
    146         -DGEANT4_BUILD_TLS_MODEL=global-dynamic \
    147         -DXERCESC_ROOT_DIR=$(juno-ext-libs-xercesc-install-dir) \
    148         -DGEANT4_USE_SYSTEM_CLHEP=ON
    149 
    150 


::

    O[blyth@localhost packages]$ svn log geant4.sh 
    ------------------------------------------------------------------------
    r4550 | lintao | 2021-05-12 23:23:19 +0800 (Wed, 12 May 2021) | 1 line

    WIP: add the optional dataset G4TENDL.1.3.2.tar.gz. Note: in G4 10.07, there is an option to enable TENDL in cmake. But for this version, just use the patch way to install the dataset and set the envvar. 
    ------------------------------------------------------------------------
    r4371 | lintao | 2021-03-21 00:16:57 +0800 (Sun, 21 Mar 2021) | 1 line

    WIP: then, patch the data files after the installation. 
    ------------------------------------------------------------------------
    r4370 | lintao | 2021-03-20 23:42:09 +0800 (Sat, 20 Mar 2021) | 1 line

    WIP: the first step is adding the patches of source code.
    ------------------------------------------------------------------------
    r4199 | lintao | 2020-11-20 22:49:52 +0800 (Fri, 20 Nov 2020) | 1 line

    WIP: fix the prefix of geant4 during genrating of .pc file
    ------------------------------------------------------------------------
    r3912 | blyth | 2020-05-21 21:29:37 +0800 (Thu, 21 May 2020) | 1 line

    Opticks integration requires ROOT to not use an ancient builtin GLEW, and Geant4 to not use builtin G4clhep 
    ------------------------------------------------------------------------
    r3886 | lintao | 2020-04-29 09:14:27 +0800 (Wed, 29 Apr 2020) | 1 line

    The default Geant4 is roll back to 10.04.p02
    ------------------------------------------------------------------------
    r3555 | lintao | 2019-08-20 00:03:00 +0800 (Tue, 20 Aug 2019) | 1 line

    update Geant4 to 10.05.p01.
    ------------------------------------------------------------------------
    r3549 | lintao | 2019-08-19 09:44:26 +0800 (Mon, 19 Aug 2019) | 1 line

    WIP: merge branch J18v2r1 back to trunk. so the trunk is based on ROOT 6.x and Geant4 10.x by default now.





Check on laptop where the symbol should be 
-----------------------------------------------

::

    epsilon:issues blyth$ g4-cls G4MTHepRandom
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02
    vi -R source/global/HEPRandom/include/G4MTHepRandom.hh source/global/HEPRandom/include/G4MTHepRandom.icc source/global/HEPRandom/src/G4MTHepRandom.cc
    3 files to edit
    epsilon:issues blyth$ 

    epsilon:issues blyth$ echo $DYLD_LIBRARY_PATH | tr ":" "\n"
    /usr/local/opticks_externals/g4_1042/lib
    /usr/local/opticks_externals/clhep/lib
    /usr/local/opticks_externals/xercesc/lib
    /usr/local/opticks_externals/boost/lib
    /usr/local/opticks/lib
    /usr/local/opticks/externals/lib
    /usr/local/cuda/lib
    /usr/local/optix/lib64
    epsilon:issues blyth$ 




just try a reinstall of externals
-------------------------------------

::

    jlibs(){
        cd $JUNOTOP/junoenv
        local libs=$(bash junoenv libs list | perl -ne 'm, (\S*)@, && print "$1\n"' -)
        for lib in $libs ; do 
            echo $lib 
            bash junoenv libs all $lib || return 1 
        done  
    }    

