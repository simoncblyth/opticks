gitlab_ci
===========

Which approach for Opticks CI ?
---------------------------------

* Opticks and OptiX from /cvmfs not Docker image because that fits the pattern of other junosw externals


Current Dockerfile:

* https://code.ihep.ac.cn/JUNO/offline/junoenv/-/blob/main/docker/Dockerfile-junosw-base-el9
* junotop/junoenv/docker/Dockerfile-junosw-base-el9  

New one to create:

* junotop/junoenv/docker/Dockerfile-junosw-cuda-rl9   
* No Opticks in name, just adding CUDA.


nvidia/cuda:12.4.1-devel-rockylinux9

First approach to try for Dockerfile-junosw-cuda-rl9:

* change the "FROM almalinux:9" from Dockerfile-junosw-base-el9  into eg::

    FROM nvidia/cuda:12.4.1-runtime-rockylinux9
    FROM nvidia/cuda:12.4.1-devel-rockylinux9

The nvidia Dockerfile are not so simple, so doing the converse and appending
cuda setup to the junosw-base-el9  would be time consuming now and 
difficult to maintain for future CUDA version bumps::

    epsilon:cuda blyth$ git remote -v
    origin	https://gitlab.com/nvidia/container-images/cuda.git (fetch)
    origin	https://gitlab.com/nvidia/container-images/cuda.git (push)

    ./dist/12.4.1/rockylinux9/base/Dockerfile
    ./dist/12.4.1/rockylinux9/runtime/Dockerfile
    ./dist/12.4.1/rockylinux9/devel/Dockerfile

    ./dist/12.4.1/rockylinux9/runtime/cudnn/Dockerfile
    ./dist/12.4.1/rockylinux9/devel/cudnn/Dockerfile



2. build Dockerfile image in GHA, check size

   * ~/sandbox/.github/workflows/junosw-build-docker-image-and-scp.yml
   * ~/sandbox/junosw/Dockerfile

3. draw on nvidia/cuda Dockerfile to add whats needed for CUDA

   * try with just runtime
   * building opticks external is done separately, and installed to /cvmfs  
   * the goal of the image is to allow running the junosw+opticks build NOT to allow building opticks

4. add the /cvmfs config



How are nightly nightlies invoked ?
--------------------------------------

* https://code.ihep.ac.cn/JUNO/offline/junosw/-/pipeline_schedules


::

    /cvmfs/juno_nightlies.ihep.ac.cn/el9_amd64_gcc11/b/build-tools/build.sh 

    A[blyth@localhost Thu]$ cat /cvmfs/juno_nightlies.ihep.ac.cn/el9_amd64_gcc11/b/latest/setup.sh
    export JUNOTOP=/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.2.0
    export WORKTOP=/cvmfs/juno_nightlies.ihep.ac.cn/el9_amd64_gcc11/b/Thu
    source $JUNOTOP/setup.sh
    source $WORKTOP/junosw/InstallArea/setup.sh





Gitlab CI/CD
------------

* https://docs.gitlab.com/ci/
* https://code.ihep.ac.cn/JUNO/offline/junosw/-/jobs



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

  * other than simple things like trying just the runtime libs
  * *not such a big deal for JUNO CI anyhow, aim for yearly image updates*


cvmfs
------

* http://cvmfs-stratum-one.ihep.ac.cn/cvmfs/software/client_configure/ihep.ac.cn/ihep.ac.cn.pub
* https://cvmfs-stratum-one.ihep.ac.cn/cvmfs/software/client_configure/ihep.ac.cn/opticks.ihep.ac.cn.pub



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



junotop/junosw/.gitlab-ci.yml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



sudo mount -t cvmfs juno.ihep.ac.cn /cvmfs/juno.ihep.ac.cn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   sudo mount -t cvmfs juno.ihep.ac.cn /cvmfs/juno.ihep.ac.cn

   mount -t [type] [device] [dir]



* https://docs.docker.com/engine/storage/bind-mounts/

* https://stackoverflow.com/questions/64021556/how-to-execute-a-shell-script-that-has-mount-command-inside-dockerfile
* https://stackoverflow.com/questions/63516389/using-mount-command-while-docker-build

Looks like cannot "mount" within the Dockerfile building

* https://cernvm-forum.cern.ch/t/mount-cvmfs-in-container-without-access-to-docker-options/392


* https://awesome-workshop.github.io/docker-cms/04-docker-cvmfs/index.html



test gitlab ci locally ?
~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/32933174/use-gitlab-ci-to-run-tests-locally

NOT ANY MORE : FEATURE REMOVED

gitlab-runner 
~~~~~~~~~~~~~~~

* https://docs.gitlab.com/runner/install/
* https://docs.gitlab.com/runner/commands/#limitations-of-gitlab-runner-exec

* https://docs.gitlab.com/runner/install/linux-repository/?tab=RHEL%2FCentOS%2FFedora%2FAmazon+Linux

curl blocked, need to start proxy, plus el9 needs "socks5h" not "socks5"::

    A[blyth@localhost ~]$ curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh" 
    curl: (7) Failed to connect to 127.0.0.1 port 8080: Connection refused
    A[blyth@localhost ~]$ curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh" 
    curl: (97) connection to proxy closed

    A[blyth@localhost ~]$ vi ~/.curlrc
    A[blyth@localhost ~]$ cat ~/.curlrc   ## on AlmaLinux9 need "socks5h" not "socks5"
    proxy=socks5h://127.0.0.1:8080

    A[blyth@localhost ~]$ curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh" 
    #!/bin/bash

    unknown_os ()
    {
    ...


   curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh" | sudo bash

 

::

    A[blyth@localhost ~]$ curl -o script.rpm.sh -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh" 
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  7983  100  7983    0     0   7230      0  0:00:01  0:00:01 --:--:--  7230
    A[blyth@localhost ~]$ vi script.rpm.sh
    A[blyth@localhost ~]$ cat script.rpm.sh | sudo bash 
    Detected operating system as almalinux/9.
    Checking for curl...
    Detected curl...
    Downloading repository file: https://packages.gitlab.com/install/repositories/runner/gitlab-runner/config_file.repo?os=almalinux&dist=9&source=script
    done.
    Installing yum-utils...
    ...
    The repository is setup! You can now install packages.
    A[blyth@localhost ~]$ 


    A[blyth@localhost ~]$ sudo dnf install gitlab-runner


    A[blyth@localhost ~]$ which gitlab-runner
    /usr/bin/gitlab-runner
    A[blyth@localhost ~]$ gitlab-runner --help
    NAME:
       gitlab-runner - a GitLab Runner

    USAGE:
       gitlab-runner [global options] command [command options] [arguments...]

    VERSION:
       17.9.1 (bbf75488)




Argh "gitlab-runner exec" has been removed from gitlab-runner 16.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

::

    gitlab-runner exec docker test --docker-volumes "/home/elboletaire/.ssh/id_rsa:/root/.ssh/id_rsa:ro"


* https://gitlab.com/gitlab-org/gitlab/-/issues/385235

::

    deprecation notice in the 15.8 release post and fully remove gitlab-runner exec from the runner code base in the 16.0 release



Alt to "gitlab-runner exec" 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* https://stackoverflow.com/questions/78661760/any-altenatives-of-gitlab-runner-exec-docker-job-name-to-test-ci-cd-locally

Manual approach::

    lint-before-merge:
      stage: linting
      image: python:3.12
      rules:
        - if: ($CI_PIPELINE_SOURCE == "merge_request_event" && 
                  ($CI_MERGE_REQUEST_TARGET_BRANCH_NAME == "develop"|| $CI_MERGE_REQUEST_TARGET_BRANCH_NAME == "main"))
      script:
        - pip install flake8
        - flake8 . 


::

    sudo docker run -it --rm --name my-running-script \
          -w "/app" -v "$PWD":"/app" python:3.12 /bin/bash -c "pip install flake8 ; flake8 --exclude venv  ; echo "executed""


    #Where $PWD is my project with its ".gitlab-ci.yml"


* https://github.com/firecow/gitlab-ci-local



try to manually do what gitlab does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    A[blyth@localhost ~]$ scp L004:g/junosw_base_el9.tar . 

    A[blyth@localhost ~]$ docker load -i junosw_base_el9.tar
    6dca6b3e8763: Loading layer [==================================================>]  189.8MB/189.8MB
    2a11bd70fe4d: Loading layer [==================================================>]  20.99kB/20.99kB
    9451ba00a6af: Loading layer [==================================================>]  8.704kB/8.704kB
    6de449af58fe: Loading layer [==================================================>]  3.072kB/3.072kB
    1a8e11921bf7: Loading layer [==================================================>]  35.48MB/35.48MB
    2c03d98f88c8: Loading layer [==================================================>]  56.32kB/56.32kB
    1b1a1c0628ff: Loading layer [==================================================>]  31.78MB/31.78MB
    e515567f7c0b: Loading layer [==================================================>]  87.04MB/87.04MB
    0e4c7cd2124c: Loading layer [==================================================>]  1.786GB/1.786GB
    4be8f469385d: Loading layer [==================================================>]  6.656kB/6.656kB
    e869c153961b: Loading layer [==================================================>]  222.3MB/222.3MB
    81d50fdb49ef: Loading layer [==================================================>]   78.4MB/78.4MB
    ec4928d864b7: Loading layer [==================================================>]  80.85MB/80.85MB
    5773258293ac: Loading layer [==================================================>]  78.24MB/78.24MB
    39b75e8fb774: Loading layer [==================================================>]  78.64MB/78.64MB
    96544d0002e4: Loading layer [==================================================>]  79.01MB/79.01MB
    Loaded image: junosw/base:el9
    A[blyth@localhost ~]$ 

    A[blyth@localhost ~]$ docker images
    REPOSITORY                                     TAG                        IMAGE ID       CREATED         SIZE
    al9-cvmfs                                      latest                     ebccb0ed032b   18 hours ago    451MB
    nvidia_cuda_12_4_1_runtime_rockylinux9_amd64   latest                     72c9d5a2da10   19 hours ago    2.47GB
    bb42                                           latest                     c9d2aec48d25   5 months ago    4.27MB
    nvidia/cuda                                    12.4.1-devel-rockylinux9   ab9135746936   11 months ago   7.11GB
    <none>                                         <none>                     9cc24f05f309   15 months ago   176MB
    junosw/base                                    el9                        0fed15e4f2a2   15 months ago   2.69GB
       
    A[blyth@localhost ~]$ docker run -it junosw/base:el9 
    [juno@b64fc653a9d9 ~]$ ls -alst
    total 12
    0 drwx------. 2 juno juno  62 Nov 21  2023 .
    0 drwxr-xr-x. 1 root root  18 Nov 21  2023 ..
    4 -rw-r--r--. 1 juno juno  18 Jan 23  2023 .bash_logout
    4 -rw-r--r--. 1 juno juno 141 Jan 23  2023 .bash_profile
    4 -rw-r--r--. 1 juno juno 492 Jan 23  2023 .bashrc
    [juno@b64fc653a9d9 ~]$ pwd
    /home/juno
    [juno@b64fc653a9d9 ~]$ 
     


docker run script within container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

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
 
::

    You can also run a local script from the host directly::

        docker exec -i mycontainer bash < mylocal.sh 

    This reads the local host script and runs it
    inside the container. You can do this with other things (like .tgz files piped
    into tar) - its just using the '-i' to pipe into the container process std
    input. – Marvin Commented Dec 8, 2017 at 15:32

::

    A[blyth@localhost ~]$ docker run -it --name jel9 junosw/base:el9 
    [juno@798abcf0117e ~]$ 
        
    A[blyth@localhost ~]$ docker ps
    CONTAINER ID   IMAGE             COMMAND       CREATED          STATUS          PORTS     NAMES
    798abcf0117e   junosw/base:el9   "/bin/bash"   14 seconds ago   Up 14 seconds             jel9
    A[blyth@localhost ~]$ docker exec jel9 pwd
    /home/juno

    A[blyth@localhost ~]$ docker exec -i jel9 bash < docker-mock-gitlab-ci.sh 
    bash
    /home/juno
    A[blyth@localhost ~]$



    A[blyth@localhost ~]$ docker exec -i jel9 bash < docker-mock-gitlab-ci.sh 
    bash
    /home/juno
    Fuse not loaded
    total 0
    0 drwxr-xr-x. 2 root root  6 Nov 21  2023 .
    0 drwxr-xr-x. 5 root root 76 Nov 21  2023 ..
    A[blyth@localhost ~]$ 



Run it with /cvmfs mounted::

    A[blyth@localhost ~]$ docker run -it -v /cvmfs:/cvmfs:ro --name jel9 junosw/base:el9 
    docker: Error response from daemon: Conflict. The container name "/jel9" is already in use by container "798abcf0117e334ae41d6d4a40f2fc08a040e0dc0e14c39286f0da2121b206bf". You have to remove (or rename) that container to be able to reuse that name.

    Run 'docker run --help' for more information

    A[blyth@localhost ~]$ docker run -it -v /cvmfs:/cvmfs:ro --name jel9x junosw/base:el9 
    [juno@8380bd2324ae ~]$ 


Still says "Fuse not loaded" but seems to work:: 

    A[blyth@localhost ~]$ docker exec -i jel9x bash < docker-mock-gitlab-ci.sh 
    bash
    /home/juno
    Fuse not loaded
    total 14
    1 drwxrwxr-x.  3 975 975   26 Feb  3 15:50 dbdata
    1 drwxr-xr-x.  9 975 975   93 Dec 11 14:33 docutil
    1 drwxrwxr-x.  4 975 975   29 Sep 11 08:23 singularity
    1 drwxrwxr-x.  5 975 975   33 Jun 27  2024 el9_amd64_gcc11
    1 drwxrwxr-x.  5 975 975   29 Jun 13  2024 centos7_amd64_gcc1120
    1 drwxrwxr-x.  7 975 975   30 Jan  5  2024 sw
    1 drwxrwxr-x.  3 975 975   33 Dec 18  2023 centos7_amd64_gcc1120_opticks
    1 drwxrwxr-x.  4 975 975   29 Dec  1  2021 centos7_amd64_gcc830
    1 -rw-rw-r--.  1 975 975   32 Mar 27  2021 .cvmfsdirtab
    1 -rw-rw-r--.  1 975 975   28 Mar 27  2021 .cvmfsdirtab~
    1 drwxrwxr-x.  3 975 975   33 Jun  4  2020 sl7_amd64_gcc485
    1 drwxrwxr-x.  4 975 975   28 Jun  2  2020 ci
    1 drwxrwxr-x.  4 975 975   52 May 13  2020 sl6_amd64_gcc447
    1 drwxrwxr-x.  4 975 975   25 Apr 28  2020 sl6_amd64_gcc830
    1 drwxrwxr-x.  4 975 975   52 Nov 27  2019 sl6_amd64_gcc494
    1 drwxrwxr-x.  9 975 975  162 Jun 28  2019 sl6_amd64_gcc44
    1 drwxrwxr-x.  3 975 975   29 Jun 25  2019 sl7_amd64_gcc48
    1 drwxrwxr-x.  4 975 975   58 Mar 22  2017 sl5_amd64_gcc41
    1 -rw-r--r--.  1 975 975   45 Mar 27  2015 new_repository
    5 drwxr-xr-x. 18 975 975 4096 Mar 27  2015 .
    A[blyth@localhost ~]$ 



need to get the mounting sorted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hmm the build.sh giving lots of errors from ro filesystem.
Want to read from local directory and write into the container. 
 
* https://docs.docker.com/engine/storage/bind-mounts/
* https://ritviknag.com/tech-tips/how-to-mount-current-working-directory-to-your-docker-container/

::

    docker run \
      -it \
      --platform linux/amd64 \
      --mount type=bind,src=.,dst=/usr/app \
      --mount type=volume,dst=/usr/app/node_modules \
      alpine:latest


Above expts encapsulated into https://github.com/simoncblyth/sandbox/blob/master/docker-mock-gitlab-ci.sh
-----------------------------------------------------------------------------------------------------------

Usage::

     ~/sandbox/docker-mock-gitlab-ci.sh run   # start container
     ~/sandbox/docker-mock-gitlab-ci.sh exec  # invoke build script in above container



RockyLinux and AlmaLinux are close relatives : so below try junosw build with the rockylinux9 that comes with nvidia/cuda image
---------------------------------------------------------------------------------------------------------------------------------

* https://tuxcare.com/blog/almalinux-vs-rocky-linux-comparing-enterprise-linux-distributions/



Check junosw build with junosw/cuda:2.4.1-runtime-rockylinux9 : IT WORKS
------------------------------------------------------------------------------------------

::

    A[blyth@localhost ~]$ scp L004:g/junosw_cuda_12_4_1_runtime_rockylinux9.tar .



    A[blyth@localhost ~]$ docker load -i junosw_cuda_12_4_1_runtime_rockylinux9.tar
    5f70bf18a086: Loading layer [==================================================>]  1.024kB/1.024kB
    cfbded2b796b: Loading layer [==================================================>]  19.97kB/19.97kB
    ...
    80b1c74719ee: Loading layer [==================================================>]  40.14MB/40.14MB
    Loaded image: junosw/cuda:12.4.1-runtime-rockylinux9


    A[blyth@localhost ~]$ docker images
    REPOSITORY                                     TAG                          IMAGE ID       CREATED          SIZE
    junosw/cuda                                    12.4.1-runtime-rockylinux9   3b3a3332ae87   31 minutes ago   5.81GB
    junosw/base                                    el9                          987e8bddae3e   20 hours ago     2.51GB
    al9-cvmfs                                      latest                       ebccb0ed032b   44 hours ago     451MB
    nvidia_cuda_12_4_1_runtime_rockylinux9_amd64   latest                       72c9d5a2da10   45 hours ago     2.47GB
    bb42                                           latest                       c9d2aec48d25   5 months ago     4.27MB
    nvidia/cuda                                    12.4.1-devel-rockylinux9     ab9135746936   11 months ago    7.11GB
    <none>                                         <none>                       9cc24f05f309   15 months ago    176MB
    <none>                                         <none>                       0fed15e4f2a2   15 months ago    2.69GB
    A[blyth@localhost ~]$ 



    A[blyth@localhost ~]$ docker run --runtime=nvidia --gpus=all --rm -it junosw/cuda:12.4.1-runtime-rockylinux9 

    ==========
    == CUDA ==
    ==========

    CUDA Version 12.4.1

    Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    This container image and its contents are governed by the NVIDIA Deep Learning Container License.
    By pulling and using the container, you accept the terms and conditions of this license:
    https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

    A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

    [juno@ba1bcc1640be ~]$ nvidia-smi
    Wed Mar 12 09:17:38 2025       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.76                 Driver Version: 550.76         CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+






docker load of same tagged different tar junosw_base_el9_built.tar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




::

    A[blyth@localhost ~]$ scp L004:g/junosw_base_el9.tar junosw_base_el9_built.tar


    A[blyth@localhost ~]$ docker images
    REPOSITORY                                     TAG                        IMAGE ID       CREATED         SIZE
    al9-cvmfs                                      latest                     ebccb0ed032b   41 hours ago    451MB
    nvidia_cuda_12_4_1_runtime_rockylinux9_amd64   latest                     72c9d5a2da10   42 hours ago    2.47GB
    bb42                                           latest                     c9d2aec48d25   5 months ago    4.27MB
    nvidia/cuda                                    12.4.1-devel-rockylinux9   ab9135746936   11 months ago   7.11GB
    <none>                                         <none>                     9cc24f05f309   15 months ago   176MB
    junosw/base                                    el9                        0fed15e4f2a2   15 months ago   2.69GB


    A[blyth@localhost ~]$ docker load --platform linux/amd64 --input junosw_base_el9_built.tar
    7828e2f9e2fe: Loading layer [==================================================>]  19.97kB/19.97kB
    bdd4ecfb4213: Loading layer [==================================================>]  5.632kB/5.632kB
    4cfe1abca629: Loading layer [==================================================>]  3.072kB/3.072kB
    5b95069dfed0: Loading layer [==================================================>]  61.77MB/61.77MB
    6737bd33acb4: Loading layer [==================================================>]   55.3kB/55.3kB
    ca14a9c9abef: Loading layer [==================================================>]  26.48MB/26.48MB
    8e74b0612cf4: Loading layer [==================================================>]     89MB/89MB
    b7705250c6f9: Loading layer [==================================================>]  1.798GB/1.798GB
    38281dd9cc74: Loading layer [==================================================>]  6.656kB/6.656kB
    8f3bf5a55921: Loading layer [==================================================>]  173.9MB/173.9MB
    96845dfb595b: Loading layer [==================================================>]   39.6MB/39.6MB
    79d450f6d554: Loading layer [==================================================>]  42.05MB/42.05MB
    392742ae750a: Loading layer [==================================================>]  39.45MB/39.45MB
    04feea1ed969: Loading layer [==================================================>]   39.9MB/39.9MB
    7061644242bd: Loading layer [==================================================>]  40.28MB/40.28MB
    5f70bf18a086: Loading layer [==================================================>]  1.024kB/1.024kB
    The image junosw/base:el9 already exists, renaming the old one with ID sha256:0fed15e4f2a2d99ad86ac76e42ac10393ae339f6ce9d81f0288a280611838b38 to empty string
    Loaded image: junosw/base:el9

    A[blyth@localhost ~]$ docker images
    REPOSITORY                                     TAG                        IMAGE ID       CREATED         SIZE
    junosw/base                                    el9                        987e8bddae3e   17 hours ago    2.51GB
    al9-cvmfs                                      latest                     ebccb0ed032b   41 hours ago    451MB
    nvidia_cuda_12_4_1_runtime_rockylinux9_amd64   latest                     72c9d5a2da10   42 hours ago    2.47GB
    bb42                                           latest                     c9d2aec48d25   5 months ago    4.27MB
    nvidia/cuda                                    12.4.1-devel-rockylinux9   ab9135746936   11 months ago   7.11GB
    <none>                                         <none>                     9cc24f05f309   15 months ago   176MB
    <none>                                         <none>                     0fed15e4f2a2   15 months ago   2.69GB
    A[blyth@localhost ~]$ 



Start container and exec the build in two sessions::

    ~/sandbox/docker-mock-gitlab-ci.sh run
    ~/sandbox/docker-mock-gitlab-ci.sh exec


Doing the build very quick, and not a good test of the GHA built image, because of prior artifacts, so clean first::

    A[blyth@localhost junosw]$ sudo rm -rf build InstallArea   ## need sudo as belong to juno user

Then exec::

    ~/sandbox/docker-mock-gitlab-ci.sh exec


Try cuda_runtime recipe
~~~~~~~~~~~~~~~~~~~~~~~~~~

~/sandbox/.github/workflows/junosw-build-docker-image-and-scp.yml::

     37            #recipe=default
     38            recipe=cuda_runtime
     39            #recipe=cuda_devel
     40 
     41            if [ "$recipe" == "default" ]; then
     42 
     43              ref=almalinux:9
     44              tag=junosw/base:el9
     45              nam=junosw_base_el9
     46 
     47            elif [ "$recipe" == "cuda_runtime" ]; then
     48 
     49              ref=nvidia/cuda:12.4.1-runtime-rockylinux9
     50              tag=junosw/cuda:12.4.1-runtime-rockylinux9
     51              nam=junosw_cuda_12_4_1_runtime_rockylinux9
     52 
     53            elif [ "$recipe" == "cuda_devel" ]; then
     54 
     55              ref=nvidia/cuda:12.4.1-devel-rockylinux9
     56              tag=junosw/cuda:12.4.1-devel-rockylinux9
     57              nam=junosw_cuda_12_4_1_devel_rockylinux9
     58 
     59            fi
     60            out=/tmp/$nam.tar
       


* issue 1 : missing "almalinux-release-devel", switch to "rocky-repos" seems to work 

::

    ERROR: failed to solve: process "/bin/sh -c dnf install -y almalinux-release-devel" did not complete successfully: exit code: 1
    Error: Process completed with exit code 1.

    A[blyth@localhost junosw]$ rpm -ql almalinux-release-devel
    /etc/yum.repos.d/almalinux-devel.repo

    A[blyth@localhost junosw]$ cat /etc/yum.repos.d/almalinux-devel.repo
    # Devel repo for AlmaLinux
    # Not for production. For buildroot use only

    [devel]
    name=AlmaLinux $releasever - Devel
    mirrorlist=https://mirrors.almalinux.org/mirrorlist/$releasever/devel
    ...

Take a look within rockylinux9::

    docker run -it --runtime=nvidia --gpus all nvidia/cuda:12.4.1-devel-rockylinux9


    [root@69f2729917f3 yum.repos.d]# dnf  whatprovides /etc/yum.repos.d/rocky-devel.repo
    cuda                                                                                                                                                             86 kB/s | 2.6 MB     00:30    
    Rocky Linux 9 - BaseOS                                                                                                                                          851 kB/s | 2.3 MB     00:02    
    Rocky Linux 9 - AppStream                                                                                                                                       2.2 MB/s | 8.6 MB     00:03    
    Rocky Linux 9 - Extras                                                                                                                                           15 kB/s |  16 kB     00:01    
    rocky-repos-9.3-1.3.el9.noarch : Rocky Linux Package Repositories
    Repo        : @System
    Matched from:
    Filename    : /etc/yum.repos.d/rocky-devel.repo

    rocky-repos-9.5-1.2.el9.noarch : Rocky Linux Package Repositories
    Repo        : baseos
    Matched from:
    Filename    : /etc/yum.repos.d/rocky-devel.repo

    [root@69f2729917f3 yum.repos.d]# 


* https://wiki.rockylinux.org/rocky/repo/#notes-on-devel


issue 2 : missing redhat-lsb-core on rockylinux9, commenting it seems to work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* https://www.reddit.com/r/RockyLinux/comments/wjlh0s/need_to_install_redhatlsbcore_on_rocky_linux_9/?rdt=62275


Duiesel 2y ago : Now redhat-lsb-core package available in devel repo.
So just do following::

    sudo dnf install -y yum-utils
    sudo dnf config-manager --set-enabled devel
    sudo dnf update -y
    sudo dnf install redhat-lsb-core


* https://bodhi.fedoraproject.org/updates/FEDORA-EPEL-2023-336dbb57e0

* https://access.redhat.com/solutions/6973382

* https://en.wikipedia.org/wiki/Linux_Standard_Base

LSB is an abandoned Linux standardization attempt


* building image succeeds without redhat-lsb-core and with 


issue 3 : little hope for junosw+opticks build with runtime due to lack of cuda headers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trifurcate

1. create full fat recipe=cuda_devel image  junosw/cuda:12.4.1-devel-rockylinux9  
2. proceed with j+o build with runtime to see where the fails are
3. find where the fat comes from the below and try to slim 
 
   * ~/cuda/dist/12.4.1/rockylinux9/devel/Dockerfile


::

    [juno@558e34cf2c21 include]$ ls -alst
    total 100
     0 drwxr-xr-x. 3 root root   139 Apr  8  2024 .
     0 drwxr-xr-x. 1 root root    32 Apr  8  2024 ..
     0 drwxr-xr-x. 3 root root   144 Apr  8  2024 nvtx3
    56 -rw-r--r--. 1 root root 53680 Mar 15  2024 nvToolsExt.h
     8 -rw-r--r--. 1 root root  6009 Mar 15  2024 nvToolsExtCuda.h
     8 -rw-r--r--. 1 root root  5192 Mar 15  2024 nvToolsExtCudaRt.h
    12 -rw-r--r--. 1 root root  8360 Mar 15  2024 nvToolsExtOpenCL.h
    16 -rw-r--r--. 1 root root 14562 Mar 15  2024 nvToolsExtSync.h
    [juno@558e34cf2c21 include]$ pwd
    /usr/local/cuda/include




try to build runtimeplus image : exceeds GHA VM space
---------------------------------------------------------

::

    [ save 
    Wed Mar 12 13:18:01 UTC 2025
    write /dev/stdout: no space left on device
    Error: Process completed with exit code 1.


* https://github.com/marketplace/actions/maximize-build-disk-space

At the time of writing, public Github-hosted runners are using Azure DS2_v2
virtual machines, featuring a 84GB OS disk on / and a 14GB temp disk mounted on
/mnt.

* https://github.com/actions/runner-images/issues/2840


After rejig ~/sandbox/junosw/Dockerfile-junosw-cuda-runtimeplus-rl9 to be more like base uses less disk space
----------------------------------------------------------------------------------------------------------------     


Using ~/sandbox/.github/workflows/junosw-build-docker-image-and-scp.yml::

     34            echo "[ Build docker image and scp "
     35            pwd
     36 
     37            #recipe=base
     38            #recipe=runtime
     39            recipe=runtimeplus
     40            #recipe=devel
     41 
     42            tag=junosw/cuda:12.4.1-${recipe}-rockylinux9
     43            nam=junosw_cuda_12_4_1_${recipe}_rockylinux9
     44            #out=/tmp/$nam.tar   ## suspect less quota on /tmp 
     45            out=$PWD/$nam.tar

GHA::

    REPOSITORY    TAG                              IMAGE ID       CREATED          SIZE
    junosw/cuda   12.4.1-runtimeplus-rockylinux9   3d505c100ea8   17 seconds ago   7.89GB
    Wed Mar 12 14:41:49 UTC 2025

    ...

    7.5G	/home/runner/work/sandbox/sandbox/junosw_cuda_12_4_1_runtimeplus_rockylinux9.tar


    [scp.0
    Wed Mar 12 14:43:15 UTC 2025
    Wed Mar 12 15:41:28 UTC 2025
    ]scp.0


* scp took ~1hr for 7.5G


Test junosw build with junosw/cuda:12.4.1-runtimeplus-rockylinux9
--------------------------------------------------------------------

::

    scp L004:g/junosw_cuda_12_4_1_runtimeplus_rockylinux9.tar .    
         ## grab tar created by GHA

    docker load -i junosw_cuda_12_4_1_runtimeplus_rockylinux9.tar
         ## create the image 

    docker images
         ## list images

    docker ps -a
         ## list containers

    docker run -it --rm junosw/cuda:12.4.1-runtimeplus-rockylinux9
         ## without GPU access, gives warning re no GPU detected

    docker run -it --rm --runtime=nvidia --gpus=all junosw/cuda:12.4.1-runtimeplus-rockylinux9
         ## with GPU access, nvidia-smi works


::

    A[blyth@localhost ~]$ docker load -i junosw_cuda_12_4_1_runtimeplus_rockylinux9.tar
    f99b0574066c: Loading layer [==================================================>]   3.23GB/3.23GB      ##
    1a71b3728186: Loading layer [==================================================>]  19.97kB/19.97kB
    8557adab9336: Loading layer [==================================================>]  5.632kB/5.632kB
    d152ff33c263: Loading layer [==================================================>]  3.072kB/3.072kB
    39646110c209: Loading layer [==================================================>]  48.64MB/48.64MB
    9aa2fcce755d: Loading layer [==================================================>]  166.4kB/166.4kB
    8d2db3762123: Loading layer [==================================================>]  31.35MB/31.35MB
    ca1d7ab5c65c: Loading layer [==================================================>]  92.63MB/92.63MB
    3177780ecd95: Loading layer [==================================================>]  1.683GB/1.683GB     ##
    d4cc24c6c263: Loading layer [==================================================>]  6.656kB/6.656kB
    dc3bc5123512: Loading layer [==================================================>]  176.5MB/176.5MB
    ca85e6ef08f9: Loading layer [==================================================>]  41.85MB/41.85MB
    d12dab9dada4: Loading layer [==================================================>]  44.04MB/44.04MB
    428fa992aee9: Loading layer [==================================================>]  41.43MB/41.43MB
    bd9b2afee25f: Loading layer [==================================================>]  41.84MB/41.84MB
    c6d44b6e02d6: Loading layer [==================================================>]   42.2MB/42.2MB
    0c91a270d8d1: Loading layer [==================================================>]  664.6kB/664.6kB
    5f70bf18a086: Loading layer [==================================================>]  1.024kB/1.024kB
    Loaded image: junosw/cuda:12.4.1-runtimeplus-rockylinux9
    A[blyth@localhost ~]$ 

    ## 18 layers, only two are GB 3.23+1.68 = 4.91 G //// where is the other ~2.5 GB ? 
    ## how do the layers correspond to the Dockerfile lines ? 


junosw+opticks build within container
---------------------------------------

* :doc:`docker_junosw_opticks_container_build_shakedown`


Older notes
-------------


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



config.toml
------------

::

    A[blyth@localhost sandbox]$ sudo cat  /etc/gitlab-runner/config.toml
    concurrent = 1
    check_interval = 0
    shutdown_timeout = 0

    [session_server]
      session_timeout = 1800
    A[blyth@localhost sandbox]$ 



gitlab docker executor
------------------------

* https://docs.gitlab.com/runner/executors/docker/


gitlab ci yml extends
-----------------------

* https://docs.gitlab.com/ci/yaml/yaml_optimization/

gitlab yml extends override script
-----------------------------------

* https://code.ihep.ac.cn/JUNO/offline/junosw/-/merge_requests/822


using docker save .tar with OCI 
----------------------------------

::
 
    ctr -n k8s.io images import --digests simoncblyth_cuda_12_4_1_runtimeplus_rockylinux9.tar



register gitlab-runner
-----------------------

* https://medium.com/geekculture/5-ways-that-can-help-you-to-debug-your-gitlab-pipeline-b871fd626652


j+o build succeeded : how to test/use it ? 
--------------------------------------------

* https://code.ihep.ac.cn/JUNO/offline/junosw/-/jobs/65514
* /builds/JUNO/offline/junosw/InstallArea



gitlab-ci artifacts:
----------------------

* https://docs.gitlab.com/ci/jobs/job_artifacts/

deployment examples
---------------------

* https://github.com/key4hep/EDM4hep/blob/main/.gitlab-ci.yml


gitlab variables
-----------------

* https://about.gitlab.com/blog/2021/02/05/ci-deployment-and-environments/
* https://docs.gitlab.com/ci/variables/predefined_variables/
* https://docs.gitlab.com/user/project/deploy_tokens/#gitlab-deploy-token


* https://code.ihep.ac.cn/JUNO/offline/junosw/-/settings/ci_cd


Masked and hidden::

    Masked in job logs, and can never be revealed in the CI/CD settings after the variable is saved. 


gitlab secure files : Project-level Secure Files API
------------------------------------------------------

* https://docs.gitlab.com/ci/secure_files/
* https://code.ihep.ac.cn/help/api/secure_files.md

::

   curl --request GET --header "PRIVATE-TOKEN: <your_access_token>" \
              https://gitlab.example.com/api/v4/projects/1/secure_files/1/download --output myfile.jks


gitlab personal access tokens
-------------------------------

* https://docs.gitlab.com/user/profile/personal_access_tokens/

* https://docs.gitlab.com/api/rest/authentication/#personalprojectgroup-access-tokens





download-secure-files approach 
-------------------------------

* https://gitlab.com/gitlab-org/incubation-engineering/mobile-devops/download-secure-files

* downloads binary and runs it 
* downloads all the secure files
* above REST API looks better than this

::

    test:
      variables:
        SECURE_FILES_DOWNLOAD_PATH: './where/files/should/go/'
      script:
        - curl --silent "https://gitlab.com/gitlab-org/incubation-engineering/mobile-devops/download-secure-files/-/raw/main/installer" | bash


::

    epsilon:~ blyth$ curl "https://gitlab.com/gitlab-org/incubation-engineering/mobile-devops/download-secure-files/-/raw/main/installer"
    #!/usr/bin/env bash

    # This installer will:
    # 1. Detect the target platform, and download the appropriate distribution
    # 2. Copy the distribution to the bin directory as `download-secure-files`
    # 3. Make `download-secure-files` executable
    # 4. Run `download-secure-files`
    # Please note:
    # * This will only work on Linux and macOS systems
    # * curl and bash are required
    ...
    download_url="https://gitlab.com/gitlab-org/incubation-engineering/mobile-devops/download-secure-files/-/releases/permalink/latest/downloads/${bin_filename}"
    ...



REST API approach to get secure files
-----------------------------------------

Explored in ~/.ssh/gitlab_com_sandlab_api.sh : works, but awkward and brittle.


File type CI/CD variable looks better
---------------------------------------

* https://docs.gitlab.com/ci/jobs/ssh_keys/#verifying-the-ssh-host-keys
* https://docs.gitlab.com/ci/variables/#use-file-type-cicd-variables

* https://about.gitlab.com/blog/2018/08/02/using-the-gitlab-ci-slash-cd-for-smart-home-configuration-management/#preparing-the-server-and-gitlab-for-ssh-access
* https://docs.gitlab.com/ci/jobs/ssh_keys/
* https://gitlab.com/gitlab-examples/ssh-private-key/

* https://stackoverflow.com/questions/64699458/storing-ssh-private-key-in-gitlab-repository-variables 



gitlab environments
--------------------

* https://docs.gitlab.com/ci/environments/

A GitLab environment represents a specific deployment target for your
application, like development, staging, or production. Use it to manage
different configurations and deploy code during various stages of your software
lifecycle.


gitlab ssh keys
-----------------

* https://docs.gitlab.com/ci/jobs/ssh_keys/
* https://gitlab.com/gitlab-examples/ssh-private-key/

::

        ## Create a shell script that will echo the environment variable SSH_PASSPHRASE
      - echo 'echo $SSH_PASSPHRASE' > ~/.ssh/tmp && chmod 700 ~/.ssh/tmp

      ## Add the SSH key stored in SSH_PRIVATE_KEY variable to the agent store
      ## We're using tr to fix line endings which makes ed25519 keys work
      ## without extra base64 encoding.
      ## https://gitlab.com/gitlab-examples/ssh-private-key/issues/1#note_48526556
      ##
      ## If ssh-add needs a passphrase, it will read the passphrase from the current
      ## terminal if it was run from a terminal.  If ssh-add does not have a terminal
      ## associated with it but DISPLAY and SSH_ASKPASS are set, it will execute the
      ## program specified by SSH_ASKPASS and open an X11 window to read the
      ## passphrase.  This is particularly useful when calling ssh-add from a
      ## .xsession or related script. Setting DISPLAY=None drops the use of X11.
      - echo "$SSH_PRIVATE_KEY" | tr -d '\r' | DISPLAY=None SSH_ASKPASS=~/.ssh/tmp ssh-add -

      ##
      ## Use ssh-keyscan to scan the keys of your private server. Replace gitlab.com
      ## with your own domain name. You can copy and repeat that command if you have
      ## more than one server to connect to.
      ##
      - ssh-keyscan gitlab.com >> ~/.ssh/known_hosts
      - chmod 644 ~/.ssh/known_hosts



TODO : expt with this in new repo https://gitlab.com/simoncblyth/sandlab
--------------------------------------------------------------------------


gitlab access artifacts from previous stage
--------------------------------------------

* https://stackoverflow.com/questions/38140996/how-can-i-pass-gitlab-artifacts-to-another-stage


gitlab secrets
----------------

* https://docs.gitlab.com/ci/secrets/


gitlab file type variable flags
---------------------------------

Protect variable
   Export variable to pipelines running on protected branches and tags only.

Expand variable reference
   $ will be treated as the start of a reference to another variable.

   * https://gitlab.com/gitlab-org/gitlab/-/merge_requests/102212


::

Unable to create masked variable because:

    The value cannot contain the following characters: whitespace characters.


* https://forum.gitlab.com/t/mask-openssh-key/102405

it turns out you can add base64 variables and mask them! but you need to remove the linebreaks.
Example::

   openssl base64 -in <input_file> | tr -d ‘\n’


Use base64 to remove the newlines
-----------------------------------

* ~/env/tools/base64.bash

See ~/.ssh/SANDLAB_DEPLOY_KEY.rst



gitlab what are protected branches
-------------------------------------

* https://docs.gitlab.com/user/project/repository/branches/protected/

Protected branches enforce specific permissions on branches in GitLab to ensure
code stability and quality. Protected branches: Control which users can merge
and push code changes. Prevent accidental deletion of critical branches.

* The default branch for your repository is protected by default.

gitlab sign up without the trial
---------------------------------

* https://forum.gitlab.com/t/why-force-users-to-start-free-trial-and-why-sign-up-multiple-times/103525




bitbucket pipelines
---------------------

* https://support.atlassian.com/bitbucket-cloud/docs/get-started-with-bitbucket-pipelines/




deploy j+o tarball
--------------------

::

    A[blyth@localhost junosw]$ tar --transform "s/^InstallArea/jwhatever\/jversion/" -cf jwhatever.tar InstallArea
    A[blyth@localhost junosw]$ l
    total 40808
    40700 -rw-r--r--.  1 blyth blyth 41676800 Mar 19 17:16 jwhatever.tar
        4 drwxr-xr-x. 29 blyth blyth     4096 Mar 19 17:16 .
        4 drwx------. 32 blyth blyth     4096 Mar 19 17:14 ..
        4 drwxr-xr-x.  8 blyth blyth     4096 Mar 18 14:53 .git
        8 -rw-r--r--.  1 blyth blyth     8144 Mar 18 14:52 .gitlab-ci.yml
        4 drwxr-xr-x. 27 blyth blyth     4096 Mar 17 11:01 build
        0 drwxr-xr-x.  6 blyth blyth       92 Mar 17 11:01 InstallArea
        0 drwxr-xr-x.  3 blyth blyth      104 Mar 11 16:43 cmake
        8 -rwxr-xr-x.  1 blyth blyth     4174 Mar 11 16:43 junorun
        4 -rw-r--r--.  1 blyth blyth     1749 Mar 11 16:43 setup.sh

    A[blyth@localhost junosw]$ tar tvf jwhatever.tar
    drwxr-xr-x blyth/blyth       0 2025-03-17 11:01 jwhatever/jversion/
    drwxr-xr-x blyth/blyth       0 2025-03-17 11:01 jwhatever/jversion/bin/
    ...
    -rwxr-xr-x blyth/blyth    1722 2025-03-11 16:43 jwhatever/jversion/bin/tut_detsim.py
    -rwxr-xr-x blyth/blyth    3708 2025-03-11 16:43 jwhatever/jversion/bin/tut_elec2rec.py
    -rwxr-xr-x blyth/blyth    3986 2025-03-11 16:43 jwhatever/jversion/bin/tut_rtraw2rec.py


check gitlab-ci built j+o tarball
----------------------------------

::

    A[blyth@localhost ~]$ mv ~/J25.2.3_Opticks-v0.3.3.tar /data1/blyth/local/
    A[blyth@localhost local]$ tar xvf J25.2.3_Opticks-v0.3.3.tar


    A[blyth@localhost el9_amd64_gcc11]$ pwd
    /data1/blyth/local/J25.2.3_Opticks-v0.3.3/el9_amd64_gcc11
    A[blyth@localhost el9_amd64_gcc11]$ ll
    total 76
    drwxr-xr-x.   2 blyth blyth  4096 Mar 20 15:53 bin
    -rw-r--r--.   1 blyth blyth   301 Mar 20 15:53 ENV.bash
    drwxr-xr-x.  87 blyth blyth  4096 Mar 20 15:53 include
    drwxr-xr-x.   3 blyth blyth 20480 Mar 20 15:53 lib64
    drwxr-xr-x. 112 blyth blyth  4096 Mar 20 15:53 python
    -rw-r--r--.   1 blyth blyth 17243 Mar 20 15:23 setup.csh
    -rw-r--r--.   1 blyth blyth 17243 Mar 20 15:23 setup.sh
    A[blyth@localhost el9_amd64_gcc11]$ 






