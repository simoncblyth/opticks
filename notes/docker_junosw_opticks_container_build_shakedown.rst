docker_junosw_opticks_container_build_shakedown
==================================================


Workflow::

   ~/sandbox/docker-mock-gitlab-ci.sh run        ## start container 

   cd ~/junosw && sudo rm -rf build InstallArea  ## for clean build test 

   ~/sandbox/docker-mock-gitlab-ci.sh exec       ## exec build into container



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


FIXED : Issue 1 : need Opticks release using consistent CUDA 
--------------------------------------------------------------

::

    A[blyth@localhost ~]$ opticks-
    A[blyth@localhost ~]$ opticks-okdist-dirlabel
    x86_64--gcc11-geant4_10_04_p02-dbg



Review old okdist and cvmfs release machinery : overhaul needed for containers and generally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   okdist-
   okdist-vi
   okdist-info


Do a scratch release with okdist-- and tarball extraction onto /cvmfs/opticks.ihep.ac.cn/
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try okdist::

    okdist-
    okdist--

    list tarball : /data1/blyth/local/opticks_Debug/Opticks-v0.3.1.tar
    -rw-r--r--. 1 blyth blyth 329356800 Mar 13 15:42 /data1/blyth/local/opticks_Debug/Opticks-v0.3.1.tar
    315M	/data1/blyth/local/opticks_Debug/Opticks-v0.3.1.tar
    === okdist-tarball-extract
    [2025-03-13 15:42:47,117] p3514318 {/home/blyth/opticks/bin/oktar.py:251} INFO - extracting tarball with common prefix Opticks-v0.3.1/x86_64--gcc11-geant4_10_04_p02-dbg into base /data1/blyth/local/opticks_Debug 
    okdist-ls
    -rw-r--r--. 1 blyth blyth 329356800 Mar 13 15:42 /data1/blyth/local/opticks_Debug/Opticks-v0.3.1.tar
    315M	/data1/blyth/local/opticks_Debug/Opticks-v0.3.1.tar
    A[blyth@localhost opticks_Debug]$ 


Perhaps assuming that just tagged. Yep, proceed anyway. 

Follow hcvmfs- instructions to copy tarball to cvmfs-stratum-zero and extract/commit into cvmfs
This takes ~5min to do and ~5min for it to appear. 

 

FIXED : Issue 2 : glm include path inconsistency : fixed with CMake BUILD_INTERFACE INSTALL_INTERFACE directives 
------------------------------------------------------------------------------------------------------------------

:doc:`issues/build_time_prefix_for_glm_headers_leaking_into_install_tree`


FIXED : Issue 3 : junosw/Simulation/SimSvc/MultiFilmLUTMakerSvc/src/MultiFilmLUTMakerSvc.cc:61:5 error: 
-----------------------------------------------------------------------------------------------------------

This is within "WITH_G4CXOPTICKS" preprocessor macro block. 

::

   blyth-MultiFilmLUTMakerSvc-fix-WITH_G4CXOPTICKS-compilation-fail


* https://code.ihep.ac.cn/JUNO/offline/junosw/-/merge_requests/820



::

    deleting object of abstract class type 'C4IPMTAccessor' which has non-virtual destructor 
    will cause undefined behavior [-Werror=delete-non-virtual-dtor]


With rockylinux9 based container build::

      | ^~~~~~~~~~~~~~~~~~~~
      /home/juno/junosw/Simulation/SimSvc/MultiFilmLUTMakerSvc/src/MultiFilmLUTMakerSvc.cc: In destructor 'virtual MultiFilmLUTMakerSvc::~MultiFilmLUTMakerSvc()':
      /home/juno/junosw/Simulation/SimSvc/MultiFilmLUTMakerSvc/src/MultiFilmLUTMakerSvc.cc:61:5: error: deleting object of abstract class type 'C4IPMTAccessor' which has non-virtual destructor will cause undefined behavior [-Werror=delete-non-virtual-dtor]
   61 |     delete m_accessor;
      |     ^~~~~~~~~~~~~~~~~



::

    088         C4IPMTAccessor* m_accessor;


    059 MultiFilmLUTMakerSvc::~MultiFilmLUTMakerSvc()
     60 {
     61     //delete m_accessor;
     62     delete m_table;
     63     delete m_interp_res;
     64 }

    122 void MultiFilmLUTMakerSvc::create_table(int wv_sample , int aoi_sample ){
    123 
    124 
    125     set_table_resolution(wv_sample ,aoi_sample);
    126     //int size = m_caculate_vec.size();
    127     //std::cout<<"size == "<<size<<'\n';
    128     //assert(size == 2 ) ;// PMT model just 2 
    129     assert(m_pspd);
    130     // Note: need to new PMTAccessor after Material construction, plese see the PMTAccessor.h
    131     if(!m_accessor){
    132         m_accessor = new PMTAccessor(m_pspd) ;
    133     }


* https://stackoverflow.com/questions/47702776/how-to-properly-delete-pointers-when-using-abstract-classes
* https://stackoverflow.com/questions/461203/when-to-use-virtual-destructors


Workaround "// delete m_accessor;" is fine until upstream Custom4 release:: 

    epsilon:customgeant4 blyth$ git diff C4IPMTAccessor.h
    diff --git a/C4IPMTAccessor.h b/C4IPMTAccessor.h
    index e5914e9..08c7869 100644
    --- a/C4IPMTAccessor.h
    +++ b/C4IPMTAccessor.h
    @@ -19,6 +19,8 @@ struct C4IPMTAccessor
         virtual int    get_implementation_version() const = 0 ; 
         virtual void   set_implementation_version(int v) = 0 ; 
     
    +    virtual ~C4IPMTAccessor(){} ; 
    +
     };
     

Issue 4 : ownership changes in container are inconvenient, HMM unless add juno user 
--------------------------------------------------------------------------------------

After the container build, get git error "fatal: detected dubious ownership in repository"::

    A[blyth@localhost junosw]$ git status
    fatal: detected dubious ownership in repository at '/home/blyth/junosw'
    To add an exception for this directory, call:

        git config --global --add safe.directory /home/blyth/junosw
    A[blyth@localhost junosw]$ 

* https://www.baeldung.com/linux/file-ownership-docker-container




Dockerfile-junosw-cuda-runtimeplus-rl9::

     33 RUN useradd juno
     34 RUN usermod -G wheel -a juno
     ## append "wheel" to the groups that user juno is a member of 

     35 RUN echo -n "assumeyes=1" >> /etc/yum.conf

     83 USER juno
     84 WORKDIR /home/juno


* https://www.docker.com/blog/understanding-the-docker-user-instruction/

The USER instruction in a Dockerfile is a fundamental tool that determines
which user will execute commands both during the image build process and when
running the container. By default, if no USER is specified, Docker will run
commands as the root user, which can pose significant security risks. 


DONE : MR adding Dockerfile to junoenv
----------------------------------------

* https://code.ihep.ac.cn/JUNO/offline/junoenv/-/merge_requests/74


Try tests within container
---------------------------






