git_pull_10min
=================


Bizarre the 10min pull time quite repeatable !


::

    N[blyth@localhost opticks_bitbucket]$ ./pull.sh 
    Thu Aug  3 02:09:31 CST 2023
    remote: Enumerating objects: 22, done.
    remote: Counting objects: 100% (22/22), done.
    remote: Compressing objects: 100% (14/14), done.
    remote: Total 14 (delta 11), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (14/14), 2.47 KiB | 148.00 KiB/s, done.
    From https://bitbucket.org/simoncblyth/opticks
       982b14a23..36c722de0  master     -> origin/master
    Updating 982b14a23..36c722de0
    Fast-forward
     bin/rsync_put.sh                                                 | 10 ++++++++--
     notes/issues/rsync_put_repo_macOS_to_Linux_case_irregularity.rst | 66 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     pull.sh                                                          |  2 ++
     qudarap/tests/{qstateTest.cc => QStateTest.cc}                   |  0
     4 files changed, 76 insertions(+), 2 deletions(-)
     create mode 100644 notes/issues/rsync_put_repo_macOS_to_Linux_case_irregularity.rst
     rename qudarap/tests/{qstateTest.cc => QStateTest.cc} (100%)
    Thu Aug  3 02:19:23 CST 2023
    N[blyth@localhost opticks_bitbucket]$ 




    N[blyth@localhost opticks_bitbucket]$ ./pull.sh 
    origin	https://bitbucket.org/simoncblyth/opticks (fetch)
    origin	https://bitbucket.org/simoncblyth/opticks (push)
    Thu Aug  3 03:07:52 CST 2023
    remote: Enumerating objects: 38, done.
    remote: Counting objects: 100% (38/38), done.
    remote: Compressing objects: 100% (21/21), done.
    remote: Total 21 (delta 16), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (21/21), 3.98 KiB | 110.00 KiB/s, done.
    From https://bitbucket.org/simoncblyth/opticks
       36c722de0..aa7fa5f3f  master     -> origin/master
    Updating 36c722de0..aa7fa5f3f
    Fast-forward
     CSG/CSGFoundry.cc                    | 28 ++--------------------------
     CSG/CSGFoundry.h                     |  1 -
     qudarap/QEvent.cc                    | 15 +++++++++------
     qudarap/QEvent.hh                    |  6 +++++-
     sysrap/CMakeLists.txt                |  1 +
     sysrap/NP.hh                         | 27 +++++++++++++++++++++++++--
     sysrap/SEvt.cc                       | 91 ++++++++++++++++++++++++++-----------------------------------------------------------------
     sysrap/SEvt.hh                       |  4 ----
     sysrap/smeta.h                       | 54 ++++++++++++++++++++++++++++++++++++++++++++++++++++++
     sysrap/tests/SEvt_AddEnvMeta_Test.cc |  9 +++------
     sysrap/tests/smeta_test.cc           | 11 +++++++++++
     sysrap/tests/smeta_test.sh           | 29 +++++++++++++++++++++++++++++
     u4/U4Recorder.cc                     | 17 ++++++++++++-----
     u4/U4Recorder.hh                     |  5 ++++-
     14 files changed, 181 insertions(+), 117 deletions(-)
     create mode 100644 sysrap/smeta.h
     create mode 100644 sysrap/tests/smeta_test.cc
     create mode 100755 sysrap/tests/smeta_test.sh
    Thu Aug  3 03:17:44 CST 2023
    N[blyth@localhost opticks_bitbucket]$ 



