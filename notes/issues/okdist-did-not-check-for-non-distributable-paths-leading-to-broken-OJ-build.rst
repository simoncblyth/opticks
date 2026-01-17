okdist-did-not-check-for-non-distributable-paths-leading-to-broken-OJ-build
==============================================================================


Issue : broken gitlab OJ CI build for  Opticks-v0.5.7
------------------------------------------------------

::

    CMake Error in Simulation/GenTools/CMakeLists.txt:
      Imported target "Opticks::G4CX" includes non-existent path
        "/data1/blyth/local/custom4_Debug/0.1.8/include/Custom4"
      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
      * The path was deleted, renamed, or moved to another location.
      * An install or uninstall procedure did not complete successfully.
      * The installation package was faulty and references files it does not
      provide.
    CMake Error in Simulation/DetSimV2/PMTSim/CMakeLists.txt:
      Imported target "Opticks::G4CX" includes non-existent path
        "/data1/blyth/local/custom4_Debug/0.1.8/include/Custom4"
      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
      * The path was deleted, renamed, or moved to another location.
      * An install or uninstall procedure did not complete successfully.
      * The installation package was faulty and references files it does not
      provide.
    CMake Error in Simulation/DetSimV2/PhysiSim/CMakeLists.txt:
      Imported target "Opticks::G4CX" includes non-existent path
        "/data1/blyth/local/custom4_Debug/0.1.8/include/Custom4"
      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
      * The path was deleted, renamed, or moved to another location.
      * An install or uninstall procedure did not complete successfully.
      * The installation package was faulty and references files it does not
      provide.
    CMake Error in Simulation/SimSvc/MultiFilmLUTMakerSvc/CMakeLists.txt:
      Imported target "Opticks::G4CX" includes non-existent path
        "/data1/blyth/local/custom4_Debug/0.1.8/include/Custom4"
      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
      * The path was deleted, renamed, or moved to another location.
      * An install or uninstall procedure did not complete successfully.
      * The installation package was faulty and references files it does not
      provide.


Problem Obvious : I forgot to return to distributable paths when making release
--------------------------------------------------------------------------------

Can detect issue when there are any paths in CMAKE_PREFIX_PATH that do not
start with the OPTICKS_PREFIX OR /cvmfs::

    (ok) A[blyth@localhost ~]$ echo $CMAKE_PREFIX_PATH  | tr ":" "\n"
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/python-numpy/1.26.4
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Python/3.11.10
    /data1/blyth/local/custom4_Debug/0.1.8
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/CLHEP/2.4.7.1
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Xercesc/3.2.4

    /data1/blyth/local/opticks_Debug
    /data1/blyth/local/opticks_Debug/externals
    /cvmfs/opticks.ihep.ac.cn/external/OptiX_800

Added protection in okdist--::

    okdist--(){

       if ! ( om- ; om-cmake-prefix-path-check ) ; then
           echo "$BASH_SOURCE : ABORT AS om-cmake-prefix-path-check SHOWS THAT CMAKE_PREFIX_PATH CONTAINS NON-DISTRIBUTABLE PATHS"
           return 1
       fi

       okdist-install-update
       okdist-install-extras
       okdist-tarball-create
       okdist-tarball-extract
       okdist-tarball-extract-plant-latest-link
       okdist-ls

       #echo $msg okdist-deploy-opticks-site
       #okdist-deploy-opticks-site
    }


So now cannot make distribution with non-distributable paths::

    (ok) A[blyth@localhost ~]$ okdist-
    (ok) A[blyth@localhost ~]$ okdist--
    NON-STANDARD: /data1/blyth/local/custom4_Debug/0.1.8
    /home/blyth/opticks/bin/okdist.bash : ABORT AS om-cmake-prefix-path-check SHOWS THAT CMAKE_PREFIX_PATH CONTAINS NON-DISTRIBUTABLE PATHS
    (ok) A[blyth@localhost ~]$ rc
    RC 1





