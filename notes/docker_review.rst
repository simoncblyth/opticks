docker_review
======================


Tree
-----

.. toctree::

    docker
    docker-cuda
    docker-cvmfs
    docker_junosw_opticks_container_build_shakedown
    docker_push


Other docker notes
---------------------


docker.rst
    examining CUDA Dockerfile from NVIDIA, installing docker on A,  working out how to use docker, 
    docker via socks proxy issue, workaround using GitHub Actions "GHA"

docker-cuda.rst
    closer look at the official nvidia/cuda images for rockylinux9 as no almalinux9 released

docker-cvmfs.rst
    working out how to get CVMFS access from inside the container

docker_junosw_opticks_container_build_shakedown.rst
    shakedown the docker container required to get Opticks+JUNOSW to build

docker_push.rst
    notes on pushing the image to docker hub


hub.docker.com image used for Opticks + JUNOSW gitlab CI/CD 
--------------------------------------------------------------

* https://hub.docker.com/r/simoncblyth/cuda/tags
* https://hub.docker.com/layers/simoncblyth/cuda/12.4.1-runtimeplus-rockylinux9/images/sha256-f3209ee05a2e128302f039bcbb4189a1fcc8bfc94d24763b40d39260756fa59a


Preparation of CI/CD docker image for JUNOSW+Opticks done in github "sandbox" and gitlab "sandlab" repos 
-------------------------------------------------------------------------------------------------------------

* https://github.com/simoncblyth/sandbox/
* https://github.com/simoncblyth/sandbox/blob/master/docker-mock-gitlab-ci.sh
* https://github.com/simoncblyth/sandbox/blob/master/junosw/Dockerfile-junosw-cuda-runtimeplus-el9

  * developing Dockerfile used for Opticks+JUNOSW CI/CD with JUNO private gitlab 


* https://gitlab.com/simoncblyth/sandlab
* https://gitlab.com/simoncblyth/sandlab/-/blob/main/.gitlab-ci.yml


A : test docker images
-------------------------

::

    A[blyth@localhost junosw]$ which docker
    /usr/bin/docker

    A[blyth@localhost junosw]$ docker image ls
    REPOSITORY                                     TAG                              IMAGE ID       CREATED         SIZE
    junosw/cuda                                    12.4.1-el9                       8c10f253f281   5 months ago    7.89GB
    simoncblyth/cuda                               12.4.1-runtimeplus-rockylinux9   8c10f253f281   5 months ago    7.89GB
    junosw/cuda                                    12.4.1-runtimeplus-rockylinux9   3d505c100ea8   6 months ago    7.89GB
    junosw/cuda                                    12.4.1-runtime-rockylinux9       3b3a3332ae87   6 months ago    5.81GB
    junosw/base                                    el9                              987e8bddae3e   6 months ago    2.51GB
    al9-cvmfs                                      latest                           ebccb0ed032b   6 months ago    451MB
    nvidia_cuda_12_4_1_runtime_rockylinux9_amd64   latest                           72c9d5a2da10   6 months ago    2.47GB
    bb42                                           latest                           c9d2aec48d25   11 months ago   4.27MB
    nvidia/cuda                                    12.4.1-devel-rockylinux9         ab9135746936   17 months ago   7.11GB
    <none>                                         <none>                           9cc24f05f309   21 months ago   176MB
    <none>                                         <none>                           0fed15e4f2a2   21 months ago   2.69GB
    A[blyth@localhost junosw]$ 

::

    A[blyth@localhost junosw]$ docker ps 
    CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES

    A[blyth@localhost junosw]$ docker ps -a
    CONTAINER ID   IMAGE                                             COMMAND                  CREATED        STATUS                    PORTS     NAMES
    70fc5a25f906   simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9   "/opt/nvidia/nvidia_…"   2 months ago   Exited (0) 2 months ago             angry_sammet


Running an image with use of NVIDIA GPU
-----------------------------------------

::

    A[blyth@localhost ~]$ docker run --rm -it --runtime=nvidia --gpus=all simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9

    ==========
    == CUDA ==
    ==========

    CUDA Version 12.4.1

    Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    This container image and its contents are governed by the NVIDIA Deep Learning Container License.
    By pulling and using the container, you accept the terms and conditions of this license:
    https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

    A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

    [juno@80c5588ba3b4 ~]$ which curl-config
    /usr/bin/curl-config
    [juno@80c5588ba3b4 ~]$ curl-config --version
    libcurl 7.76.1
    [juno@80c5588ba3b4 ~]$ which nvidia-smi
    /usr/bin/nvidia-smi
    [juno@80c5588ba3b4 ~]$ nvidia-smi
    Wed Sep 10 06:34:54 2025       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA RTX 5000 Ada Gene...    Off |   00000000:AC:00.0 Off |                  Off |
    | 30%   31C    P8              6W /  250W |     665MiB /  32760MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+
    [juno@80c5588ba3b4 ~]$ 



NVIDIA Container Toolkit is needed on the host to allow GPU usage from containers
---------------------------------------------------------------------------------------

::

    A[blyth@localhost ~]$ which nvidia-ctk
    /usr/bin/nvidia-ctk

    A[blyth@localhost ~]$ nvidia-ctk --help
    NAME:
       NVIDIA Container Toolkit CLI - Tools to configure the NVIDIA Container Toolkit

    USAGE:
       NVIDIA Container Toolkit CLI [global options] command [command options]

    VERSION:
       1.17.8
    commit: f202b80a9b9d0db00d9b1d73c0128c8962c55f4d

    COMMANDS:
       hook     A collection of hooks that may be injected into an OCI spec
       runtime  A collection of runtime-related utilities for the NVIDIA Container Toolkit
       info     Provide information about the system
       cdi      Provide tools for interacting with Container Device Interface specifications
       system   A collection of system-related utilities for the NVIDIA Container Toolkit
       config   Interact with the NVIDIA Container Toolkit configuration
       help, h  Shows a list of commands or help for one command

    GLOBAL OPTIONS:
       --debug, -d    Enable debug-level logging (default: false) [$NVIDIA_CTK_DEBUG]
       --quiet        Suppress all output except for errors; overrides --debug (default: false) [$NVIDIA_CTK_QUIET]
       --help, -h     show help
       --version, -v  print the version
    A[blyth@localhost ~]$ 




Opticks + JUNOSW gitlab CI/CD uses the below docker image 
-----------------------------------------------------------

* https://hub.docker.com/r/simoncblyth/cuda/tags
* https://hub.docker.com/layers/simoncblyth/cuda/12.4.1-runtimeplus-rockylinux9/images/sha256-f3209ee05a2e128302f039bcbb4189a1fcc8bfc94d24763b40d39260756fa59a

This docker image was based on a CUDA image released by NVIDIA.



~/junosw/.gitlab-ci.yml::

     51 ##############################################################################
     52 # Opticks Build Job Template
     53 ##############################################################################
     54 .opticks_build_job_template:
     55   stage: build
     56   image: simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9
     57   variables:
     58     JUNOTOP: /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/Jlatest
     59     JUNO_CLANG_PREFIX:
     60     JUNO_OPTICKS_PREFIX: /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-vLatest
     61     EXTRA_BUILD_FLAGS: JUNO_CMAKE_BUILD_TYPE=Debug
     62   script:
     63     - for repo in juno.ihep.ac.cn opticks.ihep.ac.cn; do if [ ! -d /cvmfs/$repo ]; then sudo mkdir /cvmfs/$repo; fi; sudo mount -t cvmfs $repo /cvmfs/$repo; done
     64     - source .gitlab-ci/oj_helper.sh EMIT_ENV_CHECK
     65     - mkdir InstallConfig
     66     - source .gitlab-ci/oj_helper.sh EMIT_ENV    > InstallConfig/ENV.bash
     67     - cat InstallConfig/ENV.bash
     68     - source .gitlab-ci/oj_helper.sh EMIT_ENVSET > InstallConfig/envset.sh
     69     - cat InstallConfig/envset.sh
     70     - export JUNO_OFFLINE_OFF=1 # Disable the official JUNOSW when build JUNOSW
     71     - export OPTICKS_SETUP_VERBOSE=1   # ~/j/gitlab-ci/mockbuild.sh shows this protects against "set -eo pipefail"
     72     - source $JUNOTOP/setup.sh
     73     - if [ -n "$JUNO_CLANG_PREFIX" ]; then source $JUNO_CLANG_PREFIX/bashrc; fi
     74     - if [ -n "$JUNO_OPTICKS_PREFIX" ]; then source $JUNO_OPTICKS_PREFIX/bashrc; fi
     75     - env $EXTRA_BUILD_FLAGS ./build.sh
     76     - cp InstallConfig/ENV.bash  InstallArea/ENV.bash
     77     - cp InstallConfig/envset.sh InstallArea/envset.sh
     78 
     79 

