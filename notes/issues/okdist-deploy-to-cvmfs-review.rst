okdist-deploy-to-cvmfs-review
================================

Overview
---------

Current approach works, but has deficiencies:

0. not automated - tarball creation and deployment to cvmfs scripts are run manually when making Opticks releases
1. current scripts non-ideal for multi-config usage, eg same tarball names containing different configs,
   plus have to jump between configs and repeat machinery in each
2. multiple extraction scripts on stratum-zero when one would do
3. used non-standard arch string with g411::

    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc15_g411/Opticks-vLatest


TODO
-----

1. DONE : include full relative paths
2. DONE : include ENV.bash
3. DONE : include the Opticks-vLatest symbolic link within the tarball (like junosw/.gitlab-ci/oj_helper.sh:make_tar does)
4. DONE : make name include the full slug (again like junosw/.gitlab-ci/oj_helper.sh:make_tar)


Review
--------

OK tarball currently only using partial relative paths::

    [lo] A[blyth@localhost opticks]$ tar tvf /data1/blyth/local/opticks_Debug/Opticks-v0.6.7.tar | head -5
    -rw-r--r-- blyth/blyth    1335 2026-08-25 14:28 el9_amd64_gcc11/Opticks-v0.6.7/envset.sh
    -rw-r--r-- blyth/blyth    6342 2026-08-25 14:28 el9_amd64_gcc11/Opticks-v0.6.7/bashrc
    -rw-r--r-- blyth/blyth      41 2026-08-25 14:28 el9_amd64_gcc11/Opticks-v0.6.7/metadata/okdist-revision.txt
    -rw-r--r-- blyth/blyth    1572 2026-08-25 14:28 el9_amd64_gcc11/Opticks-v0.6.7/metadata/okdist-info.txt
    -rwxr-xr-x blyth/blyth    4744 2026-06-09 17:13 el9_amd64_gcc11/Opticks-v0.6.7/bin/G4CXOpticks_setGeometry_Test.sh
    [lo] A[blyth@localhost opticks]$
    [lo] A[blyth@localhost opticks]$ tar tvf /data1/blyth/local/opticks_Debug/Opticks-v0.6.7.tar | tail -5
    -rw-r--r-- blyth/blyth      189 2025-07-02 11:11 el9_amd64_gcc11/Opticks-v0.6.7/externals/share/bcm/cmake/BCMIgnorePackage.cmake
    -rw-r--r-- blyth/blyth    13881 2025-07-02 11:11 el9_amd64_gcc11/Opticks-v0.6.7/externals/share/bcm/cmake/BCMPkgConfig.cmake
    -rw-r--r-- blyth/blyth     5952 2025-07-02 11:11 el9_amd64_gcc11/Opticks-v0.6.7/externals/share/bcm/cmake/BCMProperties.cmake
    -rw-r--r-- blyth/blyth      367 2025-07-02 11:11 el9_amd64_gcc11/Opticks-v0.6.7/externals/share/bcm/cmake/version.hpp
    -rw-r--r-- blyth/blyth     1406 2025-07-02 11:11 el9_amd64_gcc11/Opticks-v0.6.7/externals/share/bcm/cmake/BCMConfig.cmake
    [lo] A[blyth@localhost opticks]$



