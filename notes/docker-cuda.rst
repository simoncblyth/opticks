notes/docker-cuda
====================







examine nvidia/cuda images on hub.docker.com and correponding Dockerfile on gitlab.com 
---------------------------------------------------------------------------------------


* https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist/12.4.1/rockylinux9?ref_type=heads


::

    base
    runtime
    devel
    cudnn


* https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.4.1/rockylinux9/base/Dockerfile?ref_type=heads

::

    FROM rockylinux:9 as base


* https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.4.1/rockylinux9/runtime/Dockerfile?ref_type=heads

::

    FROM ${IMAGE_NAME}:12.4.1-base-rockylinux9 as base


* https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.4.1/rockylinux9/devel/Dockerfile?ref_type=heads

::

    FROM ${IMAGE_NAME}:12.4.1-runtime-rockylinux9 as base


* https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.4.1/rockylinux9/devel/cudnn/Dockerfile?ref_type=heads

::

    FROM ${IMAGE_NAME}:12.4.1-devel-rockylinux9 as base




* https://hub.docker.com/_/rockylinux/tags

::

    https://hub.docker.com/layers/nvidia/cuda/12.4.1-devel-rockylinux9/images/sha256-483ac85033dfcf05066887e530fbb18b9f0abac2a84545900ef53733559fb20e

    docker pull nvidia/cuda:12.4.1-devel-rockylinux9@sha256-483ac85033dfcf05066887e530fbb18b9f0abac2a84545900ef53733559fb20e
    ## that failed so used

    docker pull nvidia/cuda:12.4.1-devel-rockylinux9



TARGETARCH
-----------

* https://docs.docker.com/reference/dockerfile/#automatic-platform-args-in-the-global-scope
* https://github.com/BretFisher/multi-platform-docker-build/blob/main/README.md


Opticks Users Dockerfile
-------------------------

* https://github.com/seriksen/opticks_docker/blob/main/Dockerfile


driver check
-------------

::

    [root@107c835344f6 entrypoint.d]# cat 50-gpu-driver-check.sh
    #!/bin/bash
    # Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    # Check if libcuda.so.1 -- the CUDA driver -- is present in the ld.so cache or in LD_LIBRARY_PATH
    _LIBCUDA_FROM_LD_CACHE=$(ldconfig -p | grep libcuda.so.1)
    _LIBCUDA_FROM_LD_LIBRARY_PATH=$( ( IFS=: ; for i in ${LD_LIBRARY_PATH}; do ls $i/libcuda.so.1 2>/dev/null | grep -v compat; done) )
    _LIBCUDA_FOUND="${_LIBCUDA_FROM_LD_CACHE}${_LIBCUDA_FROM_LD_LIBRARY_PATH}"

    # Check if /dev/nvidiactl (like on Linux) or /dev/dxg (like on WSL2) or /dev/nvgpu (like on Tegra) is present
    _DRIVER_FOUND=$(ls /dev/nvidiactl /dev/dxg /dev/nvgpu 2>/dev/null)

    # If either is not true, then GPU functionality won't be usable.
    if [[ -z "${_LIBCUDA_FOUND}" || -z "${_DRIVER_FOUND}" ]]; then
      echo
      echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
      echo "   Use the NVIDIA Container Toolkit to start this container with GPU support; see"
      echo "   https://docs.nvidia.com/datacenter/cloud-native/ ."
      export NVIDIA_CPU_ONLY=1
    fi
    [root@107c835344f6 entrypoint.d]# 






why is nvidia/cuda:12.4.1-devel-rockylinux9 image 7.11GB ? must contain both arch
-------------------------------------------------------------------------------------

* https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/12.4.1/rockylinux9/devel/Dockerfile
* https://hub.docker.com/layers/nvidia/cuda/12.4.1-devel-rockylinux9/images/sha256-483ac85033dfcf05066887e530fbb18b9f0abac2a84545900ef53733559fb20e 

::

    TAG
    12.4.1-devel-rockylinux9
    Last pushed 11 months by svccomputepackagin363
    docker pull nvidia/cuda:12.4.1-devel-rockylinux9

    Digest	OS/ARCH	Compressed size
    483ac85033df  linux/amd64  3.7 GB
    dff322bce0f0  linux/arm64  3.25 GB


check manifest.json
~~~~~~~~~~~~~~~~~~~~~

::

    A[blyth@localhost ~]$ tar xvf cuda12-4-1-devel-rl9-amd.tar manifest.json
    A[blyth@localhost ~]$ cat  manifest.json |  ~/o/bin/jsonpp.py
    [
        {
            "Config": "blobs/sha256/ab91357469369ca14066aa6c17087870114bf1d186c578eb2baaab4f2e632f25",
            "RepoTags": [
                "nvidia/cuda:12.4.1-devel-rockylinux9"
            ],
            "Layers": [
                "blobs/sha256/c4bc4a1387e82c199a05c950a61d31aba8e1481a94c63196b82e25ac8367e5d1",
                "blobs/sha256/29cf88fb44d49471d46488dc9efdbfac918043dcaf57c3486c06d2452490d385",
           ...

The .tar manifest.json gives no hint of multi-arch



multi arch docker image usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.infracloud.io/blogs/multi-arch-containers-ci-cd-integration/



* https://www.docker.com/blog/multi-arch-build-and-images-the-simple-way/


how to make the image smaller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devopscube.com/reduce-docker-image-size/


Dockerfile FROM nvidia/cuda:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



missing samples
~~~~~~~~~~~~~~~~

::

    [root@107c835344f6 /]# cd /usr/local/cuda/samples/1_Utilities/deviceQuery
    bash: cd: /usr/local/cuda/samples/1_Utilities/deviceQuery: No such file or directory
    [root@107c835344f6 /]# cd /usr/local/cuda/
    [root@107c835344f6 cuda]# ls
    bin  compat  compute-sanitizer	extras	gds  include  lib64  man  nvml	nvvm  share  src  targets




try using nvidia/cuda:12.4.1-devel-rockylinux9 image  
------------------------------------------------------

As docker hub is blocked and attempts to use socks proxy kill the proxy (maybe quota kill on proxy node?)
resort to using github action to pull the image, save it to tar and scp that to L004 with::

    ~/sandbox/.github/workflows/pull-docker-image-and-scp-2.yml  

Then from A::


    A[blyth@localhost ~]$ scp L004:g/cuda12-4-1-devel-rl9-amd.tar .
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ docker load -i cuda12-4-1-devel-rl9-amd.tar
    c4bc4a1387e8: Loading layer [==================================================>]  181.3MB/181.3MB
    29cf88fb44d4: Loading layer [==================================================>]  3.072kB/3.072kB
    dd00f6980f23: Loading layer [==================================================>]   5.12kB/5.12kB
    8b5530c65e23: Loading layer [==================================================>]  234.9MB/234.9MB
    5152f26b2054: Loading layer [==================================================>]  3.072kB/3.072kB
    04d6e2e7cd5c: Loading layer [==================================================>]  18.94kB/18.94kB
    55c5c28332fe: Loading layer [==================================================>]  2.062GB/2.062GB
    8bf266c350f2: Loading layer [==================================================>]  10.75kB/10.75kB
    1911f832adb7: Loading layer [==================================================>]   5.12kB/5.12kB
    b958fa547160: Loading layer [==================================================>]  4.649GB/4.649GB
    Loaded image: nvidia/cuda:12.4.1-devel-rockylinux9
    A[blyth@localhost ~]$ docker images
    REPOSITORY    TAG                        IMAGE ID       CREATED         SIZE
    bb42          latest                     c9d2aec48d25   5 months ago    4.27MB
    nvidia/cuda   12.4.1-devel-rockylinux9   ab9135746936   11 months ago   7.11GB
    <none>        <none>                     9cc24f05f309   15 months ago   176MB
    A[blyth@localhost ~]$ 




    A[blyth@localhost ~]$ docker load -i cuda12-4-1-devel-rl9-amd.tar
    c4bc4a1387e8: Loading layer [==================================================>]  181.3MB/181.3MB
    29cf88fb44d4: Loading layer [==================================================>]  3.072kB/3.072kB
    dd00f6980f23: Loading layer [==================================================>]   5.12kB/5.12kB
    8b5530c65e23: Loading layer [==================================================>]  234.9MB/234.9MB
    5152f26b2054: Loading layer [==================================================>]  3.072kB/3.072kB
    04d6e2e7cd5c: Loading layer [==================================================>]  18.94kB/18.94kB
    55c5c28332fe: Loading layer [==================================================>]  2.062GB/2.062GB
    8bf266c350f2: Loading layer [==================================================>]  10.75kB/10.75kB
    1911f832adb7: Loading layer [==================================================>]   5.12kB/5.12kB
    b958fa547160: Loading layer [==================================================>]  4.649GB/4.649GB
    Loaded image: nvidia/cuda:12.4.1-devel-rockylinux9
    A[blyth@localhost ~]$ docker images
    REPOSITORY    TAG                        IMAGE ID       CREATED         SIZE
    bb42          latest                     c9d2aec48d25   5 months ago    4.27MB
    nvidia/cuda   12.4.1-devel-rockylinux9   ab9135746936   11 months ago   7.11GB
    <none>        <none>                     9cc24f05f309   15 months ago   176MB


    A[blyth@localhost ~]$ docker run -it nvidia/cuda:12.4.1-devel-rockylinux9

    ==========
    == CUDA ==
    ==========

    CUDA Version 12.4.1

    Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    This container image and its contents are governed by the NVIDIA Deep Learning Container License.
    By pulling and using the container, you accept the terms and conditions of this license:
    https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

    A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

    WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
       Use the NVIDIA Container Toolkit to start this container with GPU support; see
       https://docs.nvidia.com/datacenter/cloud-native/ .

    [root@4e022fb353d9 /]# 




Following instruction to install "NVIDIA Container Toolkit"

* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


::

   curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
   sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

   ## nope this yields empty .repo
   ## instead download onto laptop and scp to A

   sudo rm /etc/yum.repos.d/nvidia-container-toolkit.repo
   cat ~/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

   sudo dnf install -y nvidia-container-toolkit



::

    A[blyth@localhost ~]$ sudo dnf install -y nvidia-container-toolkit
    AlmaLinux 9 - AppStream                                                                                                                                                  5.0 kB/s | 4.2 kB     00:00    
    AlmaLinux 9 - BaseOS                                                                                                                                                     4.5 kB/s | 3.8 kB     00:00    
    AlmaLinux 9 - CRB                                                                                                                                                        5.0 kB/s | 4.2 kB     00:00    
    AlmaLinux 9 - Devel                                                                                                                                                      4.8 kB/s | 4.2 kB     00:00    
    AlmaLinux 9 - Extras                                                                                                                                                     3.9 kB/s | 3.3 kB     00:00    
    CernVM packages                                                                                                                                                          4.1 kB/s | 3.0 kB     00:00    
    cuda-rhel9-x86_64                                                                                                                                                         19 kB/s | 3.5 kB     00:00    
    cuda-rhel9-x86_64                                                                                                                                                        3.4 MB/s | 2.6 MB     00:00    
    Docker CE Stable - x86_64                                                                                                                                                5.8 kB/s | 3.5 kB     00:00    
    Extra Packages for Enterprise Linux 9 - x86_64                                                                                                                            15 kB/s |  15 kB     00:01    
    Extra Packages for Enterprise Linux 9 - x86_64                                                                                                                           9.0 MB/s |  23 MB     00:02    
    packages for the GitHub CLI                                                                                                                                              5.0 kB/s | 3.0 kB     00:00    
    google-chrome                                                                                                                                                            6.3 kB/s | 1.3 kB     00:00    
    google-chrome                                                                                                                                                             12 kB/s | 4.3 kB     00:00    
    nvidia-container-toolkit                                                                                                                                                 749  B/s | 833  B     00:01    
    nvidia-container-toolkit                                                                                                                                                 5.9 kB/s | 3.1 kB     00:00    
    Importing GPG key 0xF796ECB0:
     Userid     : "NVIDIA CORPORATION (Open Source Projects) <cudatools@nvidia.com>"
     Fingerprint: C95B 321B 61E8 8C18 09C4 F759 DDCA E044 F796 ECB0
     From       : https://nvidia.github.io/libnvidia-container/gpgkey
    nvidia-container-toolkit                                                                                                                                                  19 kB/s |  31 kB     00:01    
    Dependencies resolved.
    =========================================================================================================================================================================================================
     Package                                                       Architecture                           Version                                    Repository                                         Size
    =========================================================================================================================================================================================================
    Installing:
     nvidia-container-toolkit                                      x86_64                                 1.17.4-1                                   cuda-rhel9-x86_64                                 1.2 M
    Installing dependencies:
     libnvidia-container-tools                                     x86_64                                 1.17.4-1                                   cuda-rhel9-x86_64                                  40 k
     libnvidia-container1                                          x86_64                                 1.17.4-1                                   cuda-rhel9-x86_64                                 1.0 M
     nvidia-container-toolkit-base                                 x86_64                                 1.17.4-1                                   cuda-rhel9-x86_64                                 5.6 M

    Transaction Summary
    =========================================================================================================================================================================================================
    Install  4 Packages

    Total download size: 7.9 M
    Installed size: 26 M
    Downloading Packages:
    (1/4): libnvidia-container-tools-1.17.4-1.x86_64.rpm                                                                                                                     168 kB/s |  40 kB     00:00    
    (2/4): libnvidia-container1-1.17.4-1.x86_64.rpm                                                                                                                          2.0 MB/s | 1.0 MB     00:00    
    (3/4): nvidia-container-toolkit-1.17.4-1.x86_64.rpm                                                                                                                      2.5 MB/s | 1.2 MB     00:00    
    (4/4): nvidia-container-toolkit-base-1.17.4-1.x86_64.rpm                                                                                                                 7.5 MB/s | 5.6 MB     00:00    
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total                                                                                                                                                                    8.0 MB/s | 7.9 MB     00:00     
    Running transaction check
    Transaction check succeeded.
    Running transaction test
    Transaction test succeeded.
    Running transaction
      Preparing        :                                                                                                                                                                                 1/1 
      Installing       : nvidia-container-toolkit-base-1.17.4-1.x86_64                                                                                                                                   1/4 
      Installing       : libnvidia-container1-1.17.4-1.x86_64                                                                                                                                            2/4 
      Running scriptlet: libnvidia-container1-1.17.4-1.x86_64                                                                                                                                            2/4 
      Installing       : libnvidia-container-tools-1.17.4-1.x86_64                                                                                                                                       3/4 
      Installing       : nvidia-container-toolkit-1.17.4-1.x86_64                                                                                                                                        4/4 
      Running scriptlet: nvidia-container-toolkit-1.17.4-1.x86_64                                                                                                                                        4/4 
      Verifying        : libnvidia-container-tools-1.17.4-1.x86_64                                                                                                                                       1/4 
      Verifying        : libnvidia-container1-1.17.4-1.x86_64                                                                                                                                            2/4 
      Verifying        : nvidia-container-toolkit-1.17.4-1.x86_64                                                                                                                                        3/4 
      Verifying        : nvidia-container-toolkit-base-1.17.4-1.x86_64                                                                                                                                   4/4 

    Installed:
      libnvidia-container-tools-1.17.4-1.x86_64         libnvidia-container1-1.17.4-1.x86_64         nvidia-container-toolkit-1.17.4-1.x86_64         nvidia-container-toolkit-base-1.17.4-1.x86_64        

    Complete!
    A[blyth@localhost ~]$ 



Configure Docker to use "nvidia-container-toolkit"
----------------------------------------------------

::


    A[blyth@localhost ~]$ sudo nvidia-ctk runtime configure --runtime=docker
    INFO[0000] Config file does not exist; using empty config 
    INFO[0000] Wrote updated config to /etc/docker/daemon.json 
    INFO[0000] It is recommended that docker daemon be restarted. 
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ sudo systemctl restart docker
    A[blyth@localhost ~]$ 


    #sudo docker run --rm --runtime=nvidia --gpus all  nvidia-smi

    sudo docker run -it --runtime=nvidia --gpus all nvidia/cuda:12.4.1-devel-rockylinux9



::

    A[blyth@localhost ~]$ sudo docker run -it --runtime=nvidia --gpus all nvidia/cuda:12.4.1-devel-rockylinux9
    [sudo] password for blyth: 

    ==========
    == CUDA ==
    ==========

    CUDA Version 12.4.1

    Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    This container image and its contents are governed by the NVIDIA Deep Learning Container License.
    By pulling and using the container, you accept the terms and conditions of this license:
    https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

    A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

    [root@107c835344f6 /]# nvidia-smi
    Mon Mar 10 03:01:10 2025       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.76                 Driver Version: 550.76         CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA RTX 5000 Ada Gene...    Off |   00000000:AC:00.0 Off |                  Off |
    | 30%   36C    P8             15W /  250W |     138MiB /  32760MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    +-----------------------------------------------------------------------------------------+
    [root@107c835344f6 /]# 



Tao junosw Dockerfile
----------------------

In legacy Tao used 

* https://code.ihep.ac.cn/JUNO/offline/junoenv/-/blob/main/docker/legacy/Dockerfile-centos7?ref_type=heads

In current approach only base done in Dockerfile, externals from /cvmfs 

* https://code.ihep.ac.cn/JUNO/offline/junoenv/-/blob/main/docker/Dockerfile-junosw-base-el9?ref_type=heads

HMM: that pins the linux flavor of the container to the one of the builds that are installed onto /cvmfs


The official nvidia/cuda images are rockylinux not almalinux, which is annoying. 


nvidia cuda almalinux Dockerfile
----------------------------------


will a rockylinux executable run on almalinux ?
-----------------------------------------------


how to deploy cuda application with Docker
--------------------------------------------

HMM: now have the container with GPU access, how to proceed

* its very bare bones, need loads of pkg to be able to build
* but Tao did that already


::

    FROM almalinux:9

    ARG PASSWORD

    RUN useradd juno
    RUN usermod -G wheel -a juno
    RUN echo -n "assumeyes=1" >> /etc/yum.conf

    RUN dnf install 'dnf-command(config-manager)'
    RUN dnf config-manager --set-enabled crb
    ...



* https://www.jmoisio.eu/en/blog/2020/06/01/building-cpp-containers-using-docker-and-cmake/





CUDA Docker Container
-----------------------

* https://github.com/NVIDIA/nvidia-container-toolkit
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

NGC
----

* https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda


NVIDIA CUDA OptiX Docker container
------------------------------------

* https://forums.developer.nvidia.com/t/optix-7-5-8-0-fails-inside-docker-but-works-on-host/280500

dhart::

   .. get all CUDA samples to run in the container first ...



Old OptiX Docker
-----------------

https://github.com/ozen/optix-docker

https://github.com/ozen/optix-docker/blob/master/Dockerfile::

    ARG CUDA_IMAGE_TAG=10.0-devel-ubuntu18.04
    FROM nvidia/cuda:${CUDA_IMAGE_TAG}
    MAINTAINER Yigit Ozen
    ARG OPTIX_VERSION=5.1.0
    ADD NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64 /usr/local/optix
    ENV LD_LIBRARY_PATH /usr/local/optix/lib64:${LD_LIBRARY_PATH}


::

    cp -R /path/to/NVIDIA-OptiX-SDK-5.1.0-linux64 .
    docker build -t optix --build-arg CUDA_IMAGE_TAG=10.0-runtime-ubuntu18.04 --build-arg OPTIX_VERSION=5.1.0 .



nvidia-docker
---------------

* https://github.com/NVIDIA/nvidia-docker
* This project has been superseded by the NVIDIA Container Toolkit.



NVIDIA Container Toolkit 
-------------------------

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html#

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda


NICE INTRO DOC
---------------

* https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html






NVIDIA GPUs natively supported as devices in Docker from 19.03 (2021-02-01)
-----------------------------------------------------------------------------

https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html

As of Docker release 19.03, NVIDIA GPUs are natively supported as devices in
the Docker runtime. This means that the special runtime provided by
nvidia-docker2 is no longer necessary. 

https://docs.docker.com/engine/release-notes/19.03/

* Added DeviceRequests to HostConfig to support NVIDIA GPUs. moby/moby#38828
* https://github.com/moby/moby/pull/38828



Dockerfile
------------

* https://docs.docker.com/reference/dockerfile/




::

    ARG  CODE_VERSION=latest
    FROM base:${CODE_VERSION}
    CMD  /code/run-app

    FROM extras:${CODE_VERSION}
    CMD  /code/run-extras










