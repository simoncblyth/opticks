release_workflow
==================

How the ordinary daily dated releases are deployed into CVMFS
---------------------------------------------------------------

The "OJ" opticks build is configured with ~/junosw/.gitlab-ci.yml
which both builds, makes tarball and deploys the tarball to CVMFS 
via the ~/junosw/oj_initialize.sh script using the below lines::

     40 source .gitlab-ci/oj_helper.sh MAKE_TAR
     41 source .gitlab-ci/oj_helper.sh DEPLOY_TAR

Both the above steps will do nothing unless the OPTICKS_DEPLOY_HOST envvar is defined.
That is defined for the gitlab runner by the gitlab secrets mechanism.

DEPLOY_TAR works by setting up an ssh-agent, copying the tarball to remote host
and running a remote command to explode the tarball into cvmfs::

    161    echo scp $OJ_DIST OJ:${OPTICKS_DEPLOY_FOLD}
    162    scp $OJ_DIST OJ:${OPTICKS_DEPLOY_FOLD}
    163    date
    164    echo ssh OJ ./oj_deploy_to_cvmfs.sh ${OPTICKS_DEPLOY_FOLD}/$OJ_DIST
    165    ssh OJ ./oj_deploy_to_cvmfs.sh ${OPTICKS_DEPLOY_FOLD}/$OJ_DIST
    166    date


oj_deploy_to_cvmfs.sh 
~~~~~~~~~~~~~~~~~~~~~~

Publishes an exploded tarball directory using "cvmfs_server publish"
into a directory with the name of the day, eg::

    /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.4/el9_amd64_gcc11/Tue/





oj_reference_deploy_to_cvmfs.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is very similar to oj_deploy_to_cvmfs.sh by the release is placed
into a dated folder instead of the name of the day::

    l /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.1/el9_amd64_gcc11/
    total 7
    1 drwxrwxr-x 10 optickspub optickspub  43 Jul 30 17:40 .
    1 lrwxrwxrwx  1 optickspub optickspub   3 Jul 30 17:40 Latest -> Wed
    1 drwxr-xr-x  7 optickspub optickspub 181 Jul 30 17:34 Wed
    1 drwxr-xr-x  7 optickspub optickspub 181 Jul 29 17:33 Tue
    1 drwxr-xr-x  7 optickspub optickspub 181 Jul 28 17:36 Mon
    1 drwxr-xr-x  7 optickspub optickspub 181 Jul 27 17:35 Sun
    1 drwxr-xr-x  7 optickspub optickspub 181 Jul 26 17:34 Sat
    1 drwxr-xr-x  7 optickspub optickspub 181 Jul 25 17:37 Fri
    1 drwxr-xr-x  7 optickspub optickspub 181 Jul 24 18:29 Thu
    1 lrwxrwxrwx  1 optickspub optickspub  10 Jul 17 16:32 LastRef -> 2025_07_17
    1 drwxrwxr-x  3 optickspub optickspub  37 Jul 17 15:46 ..
    1 -rw-r--r--  1 optickspub optickspub   0 Jul 17 15:46 .cvmfscatalog
    1 drwxr-xr-x  7 optickspub optickspub 181 Jul 17 15:39 2025_07_17


HMM : how to automate dated reference deployment ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Could delect the first OJ install following an Opticks release by
the emptyness of the directory, and switch to dated folder in that case::

    [optickspub@cvmfs-stratum-zero-b ~]$ l /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.4.0_Opticks-v0.5.4/el9_amd64_gcc11/
    total 3
    1 drwxrwxr-x 3 optickspub optickspub  43 Sep 30 12:18 .
    1 lrwxrwxrwx 1 optickspub optickspub   3 Sep 30 12:18 Latest -> Tue
    1 drwxr-xr-x 7 optickspub optickspub 181 Sep 30 12:11 Tue
    1 -rw-r--r-- 1 optickspub optickspub   0 Sep 30 10:12 .cvmfscatalog
    1 drwxrwxr-x 3 optickspub optickspub  37 Sep 30 10:12 ..
    [optickspub@cvmfs-stratum-zero-b ~]$ 


Yes, BUT would not want to make a reference build from a MR branch CI run.

Added some CI variables to the context capture done by .gitlab-ci/oj_helper.sh
eg to allow conditional dated reference deployment for the first non-MR 
OJ build of a new Opticks release. 
     


