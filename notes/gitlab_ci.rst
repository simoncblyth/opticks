gitlab_ci
===========

Which approach for Opticks CI ?
---------------------------------

* Opticks and OptiX from /cvmfs not Docker image because that fits the pattern of other junosw externals
* CUDA from Docker image based FROM junosw/base:el9 to create  opticks/junosw-cuda-el9

  * junotop/junoenv/docker/Dockerfile-junosw-opticks-cuda-el9 
  * 



Approach:


1. start with https://code.ihep.ac.cn/JUNO/offline/junoenv/-/blob/main/docker/Dockerfile-junosw-base-el9?ref_type=heads

   * that is "FROM almalinux:9"

Try::

   FROM junosw/base:el9


2. build image in GHA, check size
3. draw on nvidia/cuda Dockerfile to add whats needed for CUDA
4. add the /cvmfs config


Docker image size will be big ... so
--------------------------------------

* aim for the image to be very seldom updated (eg once per year)

  * eg only when updating CUDA/OptiX versions 
 
* OptiX headers have no rpm pkg, so could just grab from /cvmfs
  using some symbolic link to pick the version ?  
  
  * /cvmfs/opticks.ihep.ac.cn/external/OptiX_800

* essentially aim for the image to just capture 

  * standard repo sourced pkgs 
  * cvmfs config

* things like Opticks needing more frequent updates can come from releases onto /cvmfs

  * use a Jlatest symbolic link to select the Opticks 

* worrying about how to shrink Docker image size can be deferred

  * *not such a big deal for JUNO CI anyhow, aim for yearly image updates*




.gitlab-ci.yml : specify the CI pipeline comprising a set of jobs
-------------------------------------------------------------------

* https://docs.gitlab.com/ci/

General structure : 

* a few keys 

  * workflow:when to trigger the pipeline
  * stages: eg build, test    
 
* templates start with . 
* the rest are jobs

  * jobs specify their stage


image : specify a Docker image that the job runs in 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://docs.gitlab.com/ci/yaml/#image

::

    epsilon:junosw blyth$ grep image .gitlab-ci.yml
      image: junosw/base:el7
      image: junosw/base:el9
      image: junosw/base:el7


Those are referring to images from https://hub.docker.com/r/junosw/base/tags
   

* https://docs.gitlab.com/ci/docker/using_docker_images/


gitlab docker
~~~~~~~~~~~~~~

* https://docs.gitlab.com/ci/docker/using_docker_images/

When a CI job runs in a Docker container, the before_script, script, and
after_script commands run in the /builds/<project-path>/ directory. Your image
may have a different default WORKDIR defined. To move to your WORKDIR, save the
WORKDIR as an environment variable so you can reference it in the container
during the job’s runtime.

 
RockyLinux vs AlmaLinux
~~~~~~~~~~~~~~~~~~~~~~~~

* https://tuxcare.com/blog/almalinux-vs-rocky-linux-comparing-enterprise-linux-distributions/


nvida cuda docker image for almalinux 9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



junoenv Dockerfile
~~~~~~~~~~~~~~~~~~~~

* https://code.ihep.ac.cn/JUNO/offline/junoenv/-/blob/main/docker/Dockerfile-junosw-base-el9




gitlab pipeline web interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://code.ihep.ac.cn/JUNO/offline/junosw/-/pipelines/16868
* https://code.ihep.ac.cn/JUNO/offline/junosw/-/jobs/64524

Installs to eg::

   /builds/JUNO/offline/junosw/InstallArea/lib64/libPMTSimParamSvc.so

End of the log::

    /builds/JUNO/offline/junosw
    Wed Mar  5 09:31:14 UTC 2025
    Uploading artifacts for successful job 00:03
    Uploading artifacts...
    InstallArea: found 1082 matching files and directories 
    Uploading artifacts as "archive" to coordinator... 201 Created  id=64524 responseStatus=201 Created token=glcbt-64
    Cleaning up project directory and file based variables 00:01
    Job succeeded


junosw/build.sh::

    23 export LANG=C
    24 export LANGUAGE=C
    25 export LC_ALL=C
    26 export LC_CTYPE=C
    27 # source utilites
    28 export JUNO_OFFLINE_SOURCE_DIR=$(dirname $(readlink -e $0 2>/dev/null) 2>/dev/null) # Darwin readlink doesnt accept -e
    29 
    ...
    166 function build-dir() {
    167     local blddir=$JUNO_OFFLINE_SOURCE_DIR/build
    168 
    169     # allow users to override the directory name of blddir
    170     if [ -n "$JUNO_OFFLINE_BLDDIR" ]; then
    171         blddir=${JUNO_OFFLINE_BLDDIR}
    172     fi
    173 
    174     echo $blddir
    175 }
    ...
    177 function install-dir() {
    178     local installdir=${JUNO_OFFLINE_SOURCE_DIR}/InstallArea
    179 
    180     # allow users to override the directory name of blddir
    181     if [ -n "$JUNO_OFFLINE_INSTALLDIR" ]; then
    182         installdir=${JUNO_OFFLINE_INSTALLDIR}
    183     fi
    184 
    185     echo $installdir
    186 }
    ...

    206 function run-build() {
    207     local installdir=$(install-dir)
    208     local blddir=$(build-dir)
    209     check-build-dir
    210     check-install-dir
    211 
    212     pushd $blddir
    213 

    /// note the assumption that source dir is one level up from build dir

    214     cmake .. $(check-var-enabled graphviz) \
    215              $(check-var-enabled withoec) \
    216              $(check-var-enabled online) \
    217              $(check-var-enabled PerformanceCheck) \
    218              $(check-var-enabled dc1) \
    219              $(check-var-enabled exportCompileCommands) \
    220              -DCMAKE_CXX_STANDARD=17 \
    221              -DPython_EXECUTABLE=$(which python) \
    222              -DCMAKE_BUILD_TYPE=$(cmake-build-type) \
    223              -DCMAKE_INSTALL_PREFIX=$installdir \
    224                      || error: "ERROR Found during cmake stage. "
    225 
    226     local njobs=-j$(nproc)
    227     cmake --build . $njobs || error: "ERROR Found during make stage. "
    228     cmake --install . || error: "ERROR Found during make install stage. "
    229 
    230     popd
    231 }
    ...
    237 check-juno-envvar
    238 date
    239 run-build
    240 date
    241 
    242 
    "build.sh" 242L, 7650C



::

     30 .build_job_template:
     31   stage: build
     32   image: junosw/base:el9
     33   variables:
     34     JUNOTOP: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/Jlatest
     35     JUNO_CLANG_PREFIX:
     36     EXTRA_BUILD_FLAGS:
     37   script:
     38     - sudo mount -t cvmfs juno.ihep.ac.cn /cvmfs/juno.ihep.ac.cn
     39     - export JUNO_OFFLINE_OFF=1 # Disable the official JUNOSW when build JUNOSW
     40     - source $JUNOTOP/setup.sh
     41     - if [ -n "$JUNO_CLANG_PREFIX" ]; then source $JUNO_CLANG_PREFIX/bashrc; fi
     42     - env $EXTRA_BUILD_FLAGS ./build.sh
     43 
     44 ##############################################################################
     45 # Build Job (el9)
     46 ##############################################################################
     47 
     48 build-job:gcc11:el9:       # This job runs in the build stage, which runs first.
     49   extends: .build_job_template
     50   artifacts:
     51     paths:
     52       - InstallArea
     53 


gitlab script
~~~~~~~~~~~~~~

* https://docs.gitlab.com/ci/yaml/script/


gitlab ci/cd settings
~~~~~~~~~~~~~~~~~~~~~~~

* https://code.ihep.ac.cn/JUNO/offline/junosw/-/settings/ci_cd


how is the way gitlab uses docker configured ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://code.ihep.ac.cn/JUNO/offline/cluster-management/-/blob/master/helmfile.yaml?ref_type=heads
* https://code.ihep.ac.cn/JUNO/offline/cluster-management/-/blob/master/applications/gitlab-runner/helmfile.yaml?ref_type=heads
* https://code.ihep.ac.cn/JUNO/offline/cluster-management/-/blob/master/applications/gitlab-runner/values.yaml.gotmpl?ref_type=heads






example "docker run" commandline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

junotop/junoenv/docker/README::

    docker run \
       -e JUNO_BITTEN_USERNAME=juno \
       -e JUNO_BITTEN_PASSWORD=xxxxxxxx \
       -e JUNO_BITTEN_CONFIG=/home/juno/config.ini \
       -v $(pwd)/config.ini:/home/juno/config.ini \
       -it mirguest/juno-bitten

"docker run"
~~~~~~~~~~~~~~

* https://docs.docker.com/engine/containers/run/

"docker run -it" gives interactive tty into the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/48368411/what-is-docker-run-it-flag


-it 
   is short for --interactive + --tty. When you docker run with this command
   it takes you straight inside the container.

-d 
   is short for --detach, which means you just run the container and then
   detach from it. Essentially, you run container in the background.


docker run -it ubuntu:xenial /bin/bash starts the container in the interactive
mode (hence -it flag) that allows you to interact with /bin/bash of the
container. That means now you will have bash session inside the container, so
you can ls, mkdir, or do any bash command inside the container.

The key here is the word "interactive". If you omit the flag, the container
still executes /bin/bash but exits immediately. With the flag, the container
executes /bin/bash then patiently waits for your input.


"docker run -v" option
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://docs.docker.com/get-started/docker-concepts/running-containers/sharing-local-files/
* https://docs.docker.com/get-started/docker-concepts/running-containers/sharing-local-files/#sharing-files-between-a-host-and-container


"docker run -it" option
~~~~~~~~~~~~~~~~~~~~~~~~~

artifact : declare job outputs 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://docs.gitlab.com/ci/jobs/job_artifacts/


gitlab with gpu
~~~~~~~~~~~~~~~~~

* https://docs.gitlab.com/runner/configuration/gpus/


junosw/cmake/legacy/JUNODependencies.cmake 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    163 ## Opticks
    164 if(DEFINED ENV{OPTICKS_PREFIX})
    165    set(Opticks_VERBOSE YES)
    166    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{OPTICKS_PREFIX}/cmake/Modules")
    167    find_package(Opticks MODULE)
    168    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Opticks_FOUND:${Opticks_FOUND}" )
    169 endif()




junotop/junoenv/docker/README
-------------------------------



using cvmfs with docker
------------------------

* https://cvmfs.readthedocs.io/en/latest/cpt-containers.html


Accessing CVMFS from Docker locally
-------------------------------------

* https://awesome-workshop.github.io/docker-cms/04-docker-cvmfs/index.html


* https://cvmfs-contrib.github.io/cvmfs-tutorial-2021/02_stratum0_client/
* https://cvmfs-contrib.github.io/cvmfs-tutorial-2021/02_stratum0_client/




