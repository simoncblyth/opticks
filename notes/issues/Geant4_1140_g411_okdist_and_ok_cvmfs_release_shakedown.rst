Geant4_1140_g411_okdist_and_ok_cvmfs_release_shakedown
========================================================


DONE : okdist-deploy-to-cvmfs generalization for gcc + geant4 version flexibility
-----------------------------------------------------------------------------------

1. DONE : g4 1042 release v0.6.7 "/cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.7"
2. DONE : g4 1140 release v0.6.7 "/cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc15_g411/Opticks-v0.6.7" SHAKEDOWN

   * DONE : same Opticks release built against different gcc, Geant4 and installed to different cvmfs path
   * DONE : review okdist tarball creation on workstation and tarball exploder script on stratum0


cvmfs stratum0
--------------

::

    [lo] A[blyth@localhost releases]$ pwd
    /cvmfs/opticks.ihep.ac.cn/ok/releases
    [lo] A[blyth@localhost releases]$
    [lo] A[blyth@localhost releases]$
    [lo] A[blyth@localhost releases]$ l
    total 2
    1 drwxrwxr-x. 33 cvmfs cvmfs 63 Jun 18 14:28 el9_amd64_gcc11
    1 drwxrwxr-x.  3 cvmfs cvmfs 37 Jun  6  2025 .
    1 drwxrwxr-x.  3 cvmfs cvmfs 30 Jun  6  2025 ..
    1 -rw-r--r--.  1 cvmfs cvmfs  0 Jun  6  2025 .cvmfscatalog
    [lo] A[blyth@localhost releases]$ pwd
    /cvmfs/opticks.ihep.ac.cn/ok/releases
    [lo] A[blyth@localhost releases]$ l el9_amd64_gcc11/
    total 18
    1 drwxrwxr-x. 13 cvmfs cvmfs 260 Jun 18 14:28 Opticks-v0.6.6
    1 drwxrwxr-x. 33 cvmfs cvmfs  63 Jun 18 14:28 .
    1 lrwxrwxrwx.  1 cvmfs cvmfs  14 Jun 18 14:28 Opticks-vLatest -> Opticks-v0.6.6
    1 drwxrwxr-x. 13 cvmfs cvmfs 260 Jun  4 15:50 Opticks-v0.6.5
    1 drwxrwxr-x. 13 cvmfs cvmfs 239 Jun  2 16:37 Opticks-v0.6.4
    1 drwxrwxr-x. 13 cvmfs cvmfs 239 Mar 26 15:27 Opticks-v0.6.3
    1 drwxrwxr-x. 13 cvmfs cvmfs 239 Mar 25 16:30 Opticks-v0.6.2
    1 drwxrwxr-x. 13 cvmfs cvmfs 239 Mar 20 17:43 Opticks-v0.6.1
    1 drwxrwxr-x. 13 cvmfs cvmfs 239 Mar 20 16:32 Opticks-v0.6.0
    1 drwxrwxr-x. 13 cvmfs cvmfs 239 Jan 26  2026 Opticks-v0.5.9




FIXED : Unexpected dirlabel with gcc15
-----------------------------------------

::

    [lob] A[blyth@localhost opticks]$ opticks-okdist-dirlabel
    el9_amd64_gcc1510_g411



