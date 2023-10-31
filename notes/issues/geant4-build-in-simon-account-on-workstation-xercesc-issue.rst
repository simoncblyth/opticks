geant4-build-in-simon-account-on-workstation-xercesc-issue
===========================================================

* "blyth" account on workstation accessed with "N" 
  is used for opticks+junosw tests

* "simon" account on workstation accessed with "ssh R"
  is used to test opticks without junosw


But::

    g4-
    g4--

    [ 89%] Building CXX object source/persistency/CMakeFiles/G4persistency.dir/mctruth/src/G4VPDigitsCollectionIO.cc.o
    [ 89%] Building CXX object source/persistency/CMakeFiles/G4persistency.dir/mctruth/src/G4VPEventIO.cc.o
    [ 89%] Building CXX object source/persistency/CMakeFiles/G4persistency.dir/mctruth/src/G4VPHitIO.cc.o
    [ 89%] Building CXX object source/persistency/CMakeFiles/G4persistency.dir/mctruth/src/G4VPHitsCollectionIO.cc.o
    gmake[2]: *** No rule to make target `/home/simon/local/opticks_externals/xercesc/lib/libxerces-c.so', needed by `BuildProducts/lib64/libG4persistency.so'.  Stop.
    gmake[1]: *** [source/persistency/CMakeFiles/G4persistency.dir/all] Error 2
    gmake: *** [all] Error 2
    Tue Oct 31 15:13:30 CST 2023
    === g4-- : build FAIL
    (base) [simon@localhost geant4.10.04.p02.Debug.build]$ 
    (base) [simon@localhost geant4.10.04.p02.Debug.build]$ 
    (base) [simon@localhost geant4.10.04.p02.Debug.build]$ pwd
    /data/simon/local/opticks_externals/g4_1042.build/geant4.10.04.p02.Debug.build
    (base) [simon@localhost geant4.10.04.p02.Debug.build]$ 

Forgot to first::

   xercesc-
   xercesc--

After doing that the below completes::

    g4-
    g4--



