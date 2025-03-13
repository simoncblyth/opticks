docker_junosw_opticks_container_build_shakedown
==================================================


Workflow::

   ~/sandbox/docker-mock-gitlab-ci.sh run   ## start container 
   ~/sandbox/docker-mock-gitlab-ci.sh exec  ## exec build into container



Config
--------


::

    A[blyth@localhost ~]$ ~/sandbox/docker-mock-gitlab-ci.sh info
    docker-mock-gitlab-ci.sh:info
    ================================

       docker_ref : junosw/cuda:12.4.1-runtimeplus-rockylinux9
       docker_nam : runtimeplus


    +--------------+-----------+------------------------------------+
    | nam          | size      |  notes                             |
    +==============+===========+====================================+
    | base         |   2.51GB  |  no CUDA                           |
    +--------------+-----------+------------------------------------+
    | runtime      |   5.81GB  |  misses headers                    |
    +--------------+-----------+------------------------------------+
    | runtimeplus  |   7.5GB   |  cherrypick devel                  |      
    +--------------+-----------+------------------------------------+
    | devel        |  10GB+?   |  might be too big for GHA VM  ?    | 
    +--------------+-----------+------------------------------------+


Using::

    export OPTICKS_PREFIX=/cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-v0.2.1/x86_64-CentOS7-gcc1120-geant4_10_04_p02-dbg


This old Opticks release is bound to fail, but check dev cycle anyhow. 


Issue 1 : need Opticks release using consistent CUDA 
------------------------------------------------------

::

    A[blyth@localhost ~]$ opticks-
    A[blyth@localhost ~]$ opticks-okdist-dirlabel
    x86_64--gcc11-geant4_10_04_p02-dbg



Review old okdist and cvmfs release machinery : overhaul needed for containers and generally
---------------------------------------------------------------------------------------------

::

   okdist-
   okdist-vi


::

    A[blyth@localhost local]$ okdist-info
    okdist-info
    =============

       date             : Thu Mar 13 11:37:27 AM CST 2025
       epoch            : 1741837047
       uname -a         : Linux localhost.localdomain 5.14.0-427.16.1.el9_4.x86_64 #1 SMP PREEMPT_DYNAMIC Thu May 9 18:15:59 EDT 2024 x86_64 x86_64 x86_64 GNU/Linux
       okdist-revision  : bbd1eacd3283673140ce0bfcb58ffe3de79ab5ee   

       okdist-ext    : .tar
       okdist-prefix : Opticks-v0.3.1/x86_64--gcc11-geant4_10_04_p02-dbg
       okdist-name   : Opticks-v0.3.1.tar
       okdist-path   : /data1/blyth/local/opticks_Debug/Opticks-v0.3.1.tar

       opticks-dir   : /data1/blyth/local/opticks_Debug
           Opticks installation directory 


       okdist-release-dir-default : /data1/blyth/local/opticks_Debug
       OKDIST_RELEASE_DIR         :  

       okdist-release-dir : /data1/blyth/local/opticks_Debug
            Directory holding binary release, from which tarballs are exploded   

       okdist-install-tests 
            Creates /data1/blyth/local/opticks_Debug/tests populated with CTestTestfile.cmake files 

       okdist-create
            Creates distribution tarball in the installation directory  

       okdist-explode
            Explode distribution tarball from the release directory 

       okdist-release-prefix : /data1/blyth/local/opticks_Debug/Opticks-v0.3.1/x86_64--gcc11-geant4_10_04_p02-dbg 
            Absolute path to exploded release distribution

       okdist--
           From the installation directory, creates tarball with 
           all paths starting with the okdist-prefix  

    A[blyth@localhost local]$ 


