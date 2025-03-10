notes/docker
===============



docker hub
-----------

* https://hub.docker.com/u/junosw
* https://hub.docker.com/r/junosw/base

* https://hub.docker.com/r/junosw/base
* https://hub.docker.com/r/junosw/base/tags

* https://hub.docker.com/r/mirguest/juno-cvmfs/tags



junotop/junoenv/docker/Dockerfile-junosw-base-el9
--------------------------------------------------


::

     01 FROM almalinux:9
      2 
      3 ARG PASSWORD
      4 
      5 RUN useradd juno
     ..
     45 RUN sudo dnf install -y libuuid-devel
     46 RUN sudo dnf install -y libnsl2-devel
     47 RUN sudo dnf install -y rsync
     48 
     49 USER juno
     50 WORKDIR /home/juno




CUDA images
-------------

* https://gitlab.com/nvidia/container-images/cuda
* https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda


NVIDIA Container Toolkit
------------------------- 

The NVIDIA Container Toolkit for Docker is required to run CUDA images

* https://github.com/NVIDIA/nvidia-container-toolkit

Make sure you have installed the NVIDIA driver for your Linux Distribution Note
that you do not need to install the CUDA Toolkit on the host system, but the
NVIDIA driver needs to be installed


* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html

::

   sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi



https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.3.2/centos7/devel/Dockerfile
--------------------------------------------------------------------------------------------------------

::

    


Docker "FROM scratch"
----------------------

* https://hub.docker.com/_/scratch

Dockerfile with multiple FROM
-------------------------------

* https://docs.docker.com/build/building/multi-stage/

::

    By default, the stages aren't named, and you refer to them by their integer
    number, starting with 0 for the first FROM instruction. However, you can name
    your stages, by adding an AS <NAME> to the FROM instruction. By default, the
    stages aren't named, and you refer to them by their integer number, starting
    with 0 for the first FROM instruction. However, you can name your stages, by
    adding an AS <NAME> to the FROM instruction. 




* https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.8.0/rockylinux9/devel/Dockerfile?ref_type=heads

::

    ARG IMAGE_NAME
    FROM ${IMAGE_NAME}:12.8.0-runtime-rockylinux9 as base

    FROM base as base-amd64

    ENV NV_CUDA_LIB_VERSION 12.8.0-1
    ENV NV_NVPROF_VERSION 12.8.57-1
    ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-12-8-${NV_NVPROF_VERSION}
    ENV NV_CUDA_CUDART_DEV_VERSION 12.8.57-1
    ENV NV_NVML_DEV_VERSION 12.8.55-1
    ENV NV_LIBCUBLAS_DEV_VERSION 12.8.3.14-1
    ENV NV_LIBNPP_DEV_VERSION 12.3.3.65-1
    ENV NV_LIBNPP_DEV_PACKAGE libnpp-devel-12-8-${NV_LIBNPP_DEV_VERSION}
    ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-devel
    ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.25.1-1
    ENV NCCL_VERSION 2.25.1
    ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.8
    ENV NV_CUDA_NSIGHT_COMPUTE_VERSION 12.8.0-1
    ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE cuda-nsight-compute-12-8-${NV_CUDA_NSIGHT_COMPUTE_VERSION}

    FROM base as base-arm64

    ENV NV_CUDA_LIB_VERSION 12.8.0-1
    ENV NV_CUDA_CUDART_DEV_VERSION 12.8.57-1
    ENV NV_NVML_DEV_VERSION 12.8.55-1
    ENV NV_LIBCUBLAS_DEV_VERSION 12.8.3.14-1
    ENV NV_LIBNPP_DEV_VERSION 12.3.3.65-1
    ENV NV_LIBNPP_DEV_PACKAGE libnpp-devel-12-8-${NV_LIBNPP_DEV_VERSION}
    ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-devel
    ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.25.1-1
    ENV NCCL_VERSION 2.25.1
    ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.8
    ENV NV_CUDA_NSIGHT_COMPUTE_VERSION 12.8.0-1
    ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE cuda-nsight-compute-12-8-${NV_CUDA_NSIGHT_COMPUTE_VERSION}


    FROM base-${TARGETARCH}

    ARG TARGETARCH

    LABEL maintainer "NVIDIA CORPORATION <sw-cuda-installer@nvidia.com>"

    RUN yum install -y \
        make \
        findutils \
        cuda-command-line-tools-12-8-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-12-8-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-12-8-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-12-8-${NV_CUDA_CUDART_DEV_VERSION} \
        ${NV_NVPROF_DEV_PACKAGE} \
        cuda-nvml-devel-12-8-${NV_NVML_DEV_VERSION} \
        libcublas-devel-12-8-${NV_LIBCUBLAS_DEV_VERSION} \
        ${NV_LIBNPP_DEV_PACKAGE} \
        ${NV_LIBNCCL_DEV_PACKAGE} \
        ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
        && yum clean all \
        && rm -rf /var/cache/yum/*

    ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs



My guess at how to build from that::

    DOCKER_BUILDKIT=1 IMAGE_NAME=cuda TARGETARCH=amd64 docker build --no-cache -f Dockerfile --target base-amd64 .
    DOCKER_BUILDKIT=1 IMAGE_NAME=cuda TARGETARCH=arm64 docker build --no-cache -f Dockerfile --target base-arm64 .
 

Using the "--target" explains why what looks like duplication is actually 
customization. 



Docker ARG and ENV
-------------------

* https://www.docker.com/blog/docker-best-practices-using-arg-and-env-in-your-dockerfiles/

::

    FROM ubuntu:latest
    ARG THEARG="foo"
    RUN echo $THEARG
    CMD ["env"]

* RUN : when building the image
* CMD : when running the image (more correct to say run the container created from the image)  


docker RUN vs CMD
--------------------

RUN executes commands and creates new image layers. CMD sets the command and
its parameters to be executed by default after the container is started.
However CMD can be replaced by docker run command line parameters



Official docker images for el9 cuda ? NOPE but there are for rockylinux9
-------------------------------------------------------------------------

* https://hub.docker.com/search?q=cuda
* https://hub.docker.com/r/nvidia/cuda

base: 
    Includes the CUDA runtime (cudart)

runtime: 
    Builds on the base and includes the CUDA math libraries⁠
    , and NCCL⁠. A runtime image that also includes cuDNN⁠
    is available.
devel: 
    Builds on the runtime and includes headers, development tools for building
    CUDA images. These images are particularly useful for multi-stage builds.


* https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/12.3.2/rockylinux9/devel/Dockerfile


nvidia "cuda" docker image for almalinux 9
--------------------------------------------

* https://www.linkedin.com/pulse/nvidia-gpu-support-almalinux-owain-kenway-824ge

* https://hub.docker.com/layers/laynedsauer/fermi-el9-cuda/devel/images/sha256-e016dcb63d349f3e56d8e9441499a30b56bc011986794bc2d34a398cd3cd8287

* https://hub.docker.com/r/laynedsauer/fermi-el9-cuda


unofficial "opticks" docker images 
------------------------------------

* https://hub.docker.com/search?q=opticks&badges=none


Dockerfile Primer
------------------

* https://docs.docker.com/reference/dockerfile/

::
   

    ADD	    Add local or remote files and directories.
    ARG	    Use build-time variables.
    CMD	    Specify default commands.
    COPY	Copy files and directories.
    ENTRYPOINT	Specify default executable.
    ENV	    Set environment variables.
    EXPOSE	Describe which ports your application is listening on.
    FROM	Create a new build stage from a base image.
    HEALTHCHECK	Check a container's health on startup.
    LABEL	Add metadata to an image.
    MAINTAINER	Specify the author of an image.
    ONBUILD	Specify instructions for when the image is used in a build.
    RUN	    Execute build commands.
    SHELL	Set the default shell of an image.
    STOPSIGNAL	Specify the system call signal for exiting a container.
    USER	Set user and group ID.
    VOLUME	Create volume mounts.
    WORKDIR	Change working directory.


NVIDIA Container Toolkit for Docker
-------------------------------------

The NVIDIA Container Toolkit⁠ for Docker is required to run CUDA images.



* https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/12.3.2/rockylinux9/devel/Dockerfile

::

    ARG IMAGE_NAME
    FROM ${IMAGE_NAME}:12.3.2-runtime-rockylinux9 as base

    FROM base as base-amd64

    ENV NV_CUDA_LIB_VERSION 12.3.2-1
    ENV NV_NVPROF_VERSION 12.3.101-1
    ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-12-3-${NV_NVPROF_VERSION}
    ENV NV_CUDA_CUDART_DEV_VERSION 12.3.101-1
    ENV NV_NVML_DEV_VERSION 12.3.101-1
    ENV NV_LIBCUBLAS_DEV_VERSION 12.3.4.1-1
    ENV NV_LIBNPP_DEV_VERSION 12.2.3.2-1
    ENV NV_LIBNPP_DEV_PACKAGE libnpp-devel-12-3-${NV_LIBNPP_DEV_VERSION}
    ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-devel
    ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.20.3-1
    ENV NCCL_VERSION 2.20.3
    ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.3
    ENV NV_CUDA_NSIGHT_COMPUTE_VERSION 12.3.2-1
    ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE cuda-nsight-compute-12-3-${NV_CUDA_NSIGHT_COMPUTE_VERSION}

    FROM base as base-arm64

    ENV NV_CUDA_LIB_VERSION 12.3.2-1
    ENV NV_CUDA_CUDART_DEV_VERSION 12.3.101-1
    ENV NV_NVML_DEV_VERSION 12.3.101-1
    ENV NV_LIBCUBLAS_DEV_VERSION 12.3.4.1-1
    ENV NV_LIBNPP_DEV_VERSION 12.2.3.2-1
    ENV NV_LIBNPP_DEV_PACKAGE libnpp-devel-12-3-${NV_LIBNPP_DEV_VERSION}
    ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-devel
    ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.20.3-1
    ENV NCCL_VERSION 2.20.3
    ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.3
    ENV NV_CUDA_NSIGHT_COMPUTE_VERSION 12.3.2-1
    ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE cuda-nsight-compute-12-3-${NV_CUDA_NSIGHT_COMPUTE_VERSION}


    FROM base-${TARGETARCH}

    ARG TARGETARCH

    LABEL maintainer "NVIDIA CORPORATION <sw-cuda-installer@nvidia.com>"

    RUN yum install -y \
        make \
        findutils \
        cuda-command-line-tools-12-3-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-12-3-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-12-3-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-12-3-${NV_CUDA_CUDART_DEV_VERSION} \
        ${NV_NVPROF_DEV_PACKAGE} \
        cuda-nvml-devel-12-3-${NV_NVML_DEV_VERSION} \
        libcublas-devel-12-3-${NV_LIBCUBLAS_DEV_VERSION} \
        ${NV_LIBNPP_DEV_PACKAGE} \
        ${NV_LIBNCCL_DEV_PACKAGE} \
        ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
        && yum clean all \
        && rm -rf /var/cache/yum/*

    ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs




Docker Install onto AlmaLinux9
----------------------------------


No official instructions:

* https://docs.docker.com/engine/install/



www.liquidweb.com
~~~~~~~~~~~~~~~~~~~~

* https://www.liquidweb.com/blog/install-docker-on-linux-almalinux/

Install::

    sudo dnf --refresh update
    sudo dnf upgrade

    sudo dnf install yum-utils
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

    sudo dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin    
    sudo dnf install docker-compose-plugin    


Daemon::

    sudo systemctl start docker
    sudo systemctl enable docker
    sudo systemctl status docker    # check running 


Check::

    sudo docker version
    docker --version
    sudo docker run hello-world


Add USER to docker group, to avoid needing sudo::

    sudo usermod -aG docker $USER


To uninstall::

    sudo dnf remove docker-ce docker-ce-cli containerd.io docker-compose-plugin



reintech.io
~~~~~~~~~~~~~

* https://reintech.io/blog/installing-docker-on-almalinux-9

::

    sudo dnf update -y
    sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
    sudo dnf install docker-ce docker-ce-cli containerd.io


used on A
~~~~~~~~~~~


::

    sudo dnf --refresh update
    sudo dnf upgrade
    sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo


Issue 1, blocked ssh connection::

    A[blyth@localhost ~]$ sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
    [sudo] password for blyth: 
    Adding repo from: https://download.docker.com/linux/centos/docker-ce.repo
    Curl error (35): SSL connect error for https://download.docker.com/linux/centos/docker-ce.repo [OpenSSL SSL_connect: Connection reset by peer in connection to download.docker.com:443 ]
    Error: Configuration of repo failed
    A[blyth@localhost ~]$ 


Huh the below seemed to work without starting the proxy, presumably enough packets got thru before the block::

    A[blyth@localhost ~]$ echo "proxy=socks5://127.0.0.1:8080" > ~/.curlrc
    A[blyth@localhost ~]$ vi ~/.curlrc 
    A[blyth@localhost ~]$ sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
    [sudo] password for blyth: 
    Adding repo from: https://download.docker.com/linux/centos/docker-ce.repo
    A[blyth@localhost ~]$ 

::

    A[blyth@localhost ~]$ sudo dnf search sshpass
    A[blyth@localhost ~]$ sudo dnf install sshpass


    A[blyth@localhost ~]$ sudo dnf install docker-ce docker-ce-cli containerd.io 

Some block, but seems to complete::

    Downloading Packages:
    [MIRROR] containerd.io-1.7.25-3.1.el9.x86_64.rpm: Curl error (35): SSL connect error for https://download.docker.com/linux/centos/9/x86_64/stable/Packages/containerd.io-1.7.25-3.1.el9.x86_64.rpm [OpenSSL SSL_connect: Connection reset by peer in connection to download.docker.com:443 ]
    (1/6): docker-ce-28.0.1-1.el9.x86_64.rpm                                                                                                                            6.7 MB/s |  20 MB     00:03    
    (2/6): docker-ce-cli-28.0.1-1.el9.x86_64.rpm                       



::

    A[blyth@localhost ~]$ sudo systemctl start docker
    [sudo] password for blyth: 
    A[blyth@localhost ~]$ sudo systemctl status docker 
    ● docker.service - Docker Application Container Engine
         Loaded: loaded (/usr/lib/systemd/system/docker.service; disabled; preset: disabled)
         Active: active (running) since Thu 2025-03-06 19:54:03 CST; 29s ago
    TriggeredBy: ● docker.socket
           Docs: https://docs.docker.com
       Main PID: 385298 (dockerd)
          Tasks: 21
         Memory: 36.4M
            CPU: 320ms
         CGroup: /system.slice/docker.service
                 └─385298 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock

    Mar 06 19:54:02 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:02.527925314+08:00" level=info msg="OTEL tracing is not configured, using no-op tracer provider"
    Mar 06 19:54:02 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:02.582996651+08:00" level=info msg="Loading containers: start."
    Mar 06 19:54:02 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:02.599998888+08:00" level=info msg="Firewalld: created docker-forwarding policy"
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.757071176+08:00" level=info msg="Loading containers: done."
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.796968861+08:00" level=info msg="Docker daemon" commit=bbd0a17 containerd-snapshotter=false storage-driver=overlay>
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.797368581+08:00" level=info msg="Initializing buildkit"
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.837909753+08:00" level=info msg="Completed buildkit initialization"
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.844627277+08:00" level=info msg="Daemon has completed initialization"
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.844712033+08:00" level=info msg="API listen on /run/docker.sock"
    Mar 06 19:54:03 localhost.localdomain systemd[1]: Started Docker Application Container Engine.
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ sudo systemctl enable docker
    Created symlink /etc/systemd/system/multi-user.target.wants/docker.service → /usr/lib/systemd/system/docker.service.
    A[blyth@localhost ~]$  sudo systemctl status docker 
    ● docker.service - Docker Application Container Engine
         Loaded: loaded (/usr/lib/systemd/system/docker.service; enabled; preset: disabled)
         Active: active (running) since Thu 2025-03-06 19:54:03 CST; 1min 23s ago
    TriggeredBy: ● docker.socket
           Docs: https://docs.docker.com
       Main PID: 385298 (dockerd)
          Tasks: 21
         Memory: 34.4M
            CPU: 324ms
         CGroup: /system.slice/docker.service
                 └─385298 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock

    Mar 06 19:54:02 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:02.527925314+08:00" level=info msg="OTEL tracing is not configured, using no-op tracer provider"
    Mar 06 19:54:02 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:02.582996651+08:00" level=info msg="Loading containers: start."
    Mar 06 19:54:02 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:02.599998888+08:00" level=info msg="Firewalld: created docker-forwarding policy"
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.757071176+08:00" level=info msg="Loading containers: done."
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.796968861+08:00" level=info msg="Docker daemon" commit=bbd0a17 containerd-snapshotter=false storage-driver=overlay>
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.797368581+08:00" level=info msg="Initializing buildkit"
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.837909753+08:00" level=info msg="Completed buildkit initialization"
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.844627277+08:00" level=info msg="Daemon has completed initialization"
    Mar 06 19:54:03 localhost.localdomain dockerd[385298]: time="2025-03-06T19:54:03.844712033+08:00" level=info msg="API listen on /run/docker.sock"
    Mar 06 19:54:03 localhost.localdomain systemd[1]: Started Docker Application Container Engine.
    lines 1-22/22 (END)


    A[blyth@localhost ~]$ sudo docker version
    Client: Docker Engine - Community
     Version:           28.0.1
     API version:       1.48
     Go version:        go1.23.6
     Git commit:        068a01e
     Built:             Wed Feb 26 10:42:23 2025
     OS/Arch:           linux/amd64
     Context:           default

    Server: Docker Engine - Community
     Engine:
      Version:          28.0.1
      API version:      1.48 (minimum version 1.24)
      Go version:       go1.23.6
      Git commit:       bbd0a17
      Built:            Wed Feb 26 10:40:43 2025
      OS/Arch:          linux/amd64
      Experimental:     false
     containerd:
      Version:          1.7.25
      GitCommit:        bcc810d6b9066471b0b6fa75f557a15a1cbf31bb
     runc:
      Version:          1.2.4
      GitCommit:        v1.2.4-0-g6c52b3f
     docker-init:
      Version:          0.19.0
      GitCommit:        de40ad0
    A[blyth@localhost ~]$ 


    A[blyth@localhost ~]$ docker --version
    Docker version 28.0.1, build 068a01e


Issue 2, proxy for docker::

    A[blyth@localhost ~]$ sudo docker run hello-world
    Unable to find image 'hello-world:latest' locally
    docker: Error response from daemon: Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)

    Run 'docker run --help' for more information
    A[blyth@localhost ~]$ 


docker proxy config
-------------------------

* https://stackoverflow.com/questions/23111631/cannot-download-docker-images-behind-a-proxy
* https://docs.docker.com/engine/daemon/proxy/#httphttps-proxy

::

   A[blyth@localhost ~]$ sudo mkdir -p /etc/systemd/system/docker.service.d

Create or edit the /etc/systemd/system/docker.service.d/proxy.conf file and add::

    [Service]
    Environment="HTTP_PROXY=socks5://127.0.0.1:<PROXY_PORT>"
    Environment="HTTPS_PROXY=socks5://127.0.0.1:<PROXY_PORT>"

Replace <PROXY_PORT> with your proxy port. Then, reload systemd and restart Docker::

    sudo systemctl daemon-reload
    sudo systemctl restart docker docker.service



::

    A[blyth@localhost ~]$ sudo mkdir -p /etc/systemd/system/docker.service.d
    A[blyth@localhost ~]$ sudo vi /etc/systemd/system/docker.service.d/proxy.conf
    A[blyth@localhost ~]$ 
    A[blyth@localhost ~]$ sudo systemctl daemon-reload
    A[blyth@localhost ~]$ sudo systemctl restart docker docker.service
    A[blyth@localhost ~]$ 


    A[blyth@localhost ~]$ sudo docker run hello-world
    Unable to find image 'hello-world:latest' locally
    docker: Error response from daemon: Get "https://registry-1.docker.io/v2/library/hello-world/manifests/latest": socks connect tcp 127.0.0.1:8080->registry-1.docker.io:443: EOF

    Run 'docker run --help' for more information
    A[blyth@localhost ~]$ 


    A[blyth@localhost docker.service.d]$ sudo systemctl show --property=Environment docker
    Environment=HTTP_PROXY=socks5://127.0.0.1:8080 HTTPS_PROXY=socks5://127.0.0.1:8080


This error might be due to quota on the CNAF gateway node ? 


Tao: Use GitHub actions to pull from dockerhub and push to IHEP
-----------------------------------------------------------------

* https://github.com/cepc/cepcsw-externals-mirroring/blob/master/.github/workflows/main.yml

* https://code.ihep.ac.cn/cepc/externals/mirroring

docker export 
---------------

* https://docs.docker.com/reference/cli/docker/container/export/

::

    docker export red_panda > latest.tar


* https://docs.docker.com/reference/cli/docker/image/import/

docker save vs export
---------------------

* https://www.nutrient.io/blog/docker-import-export-vs-load-save/

Docker provides four commands for transferring images and containers: save and
load work with images, preserving layers and metadata, while export and import
work with containers, capturing filesystem snapshots.
Use save/load for transferring complete images with history, and export/import
for filesystem snapshots or creating new base images.Use save/load for
transferring complete images with history, and export/import for filesystem
snapshots or creating new base images.


Docker images vs. containers
------------------------------

* https://www.nutrient.io/blog/docker-import-export-vs-load-save/

Images are static, read-only templates stored in registries like Docker Hub.

In contrast, a Docker container is a live, running instance of an image. Think
of it as a house built from the image blueprint. Containers provide an isolated
environment for running your application, ensuring it doesn’t interfere with
other processes on the host system.


github action
----------------

::

    epsilon:~ blyth$ git clone git@github.com:simoncblyth/sandbox.git
    Cloning into 'sandbox'...
    Connection to github.com port 22 [tcp/ssh] succeeded!
    warning: You appear to have cloned an empty repository.
    epsilon:~ blyth$ cd sandbox
    epsilon:sandbox blyth$ 



docker inside github actions
------------------------------

* https://aschmelyun.com/blog/using-docker-run-inside-of-github-actions/
* https://github.com/addnab/docker-run-action
* https://docs.docker.com/guides/gha/



gha
----

* https://docs.github.com/en/actions/writing-workflows


gha : running script
----------------------

* https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/adding-scripts-to-your-workflow

official docker actions on github
-----------------------------------

* https://github.com/orgs/docker/repositories
* https://github.com/docker/setup-docker-action

If you're using GitHub-hosted runners on Linux or Windows, Docker is already up
and running, so it might not be necessary to use this action.


gha : supported runners
------------------------

* https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources
* https://github.com/actions/runner-images/blob/ubuntu24/20250302.1/images/ubuntu/Ubuntu2404-Readme.md

Lots availble including::

   cmake, git. docker 

gha : VM and optionally containers within that
------------------------------------------------

GitHub actions provision a virtual machine - as you noted, either Ubuntu,
Windows or macOS - and run your workflow inside of that. You can then use that
virtual machine to run a workflow inside a container.





gha : container
----------------

* https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions#jobsjob_idcontainer
* https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions#example-running-a-job-within-a-container


hello-world
--------------

* https://hub.docker.com/_/hello-world/tags
* https://hub.docker.com/_/busybox


running-jobs-in-a-container
-----------------------------

* https://docs.github.com/en/actions/writing-workflows/choosing-where-your-workflow-runs/running-jobs-in-a-container



gha : publishing-docker-images
--------------------------------

* https://docs.github.com/en/actions/use-cases-and-examples/publishing-packages/publishing-docker-images




docker save/load and export/import
-----------------------------------

* https://www.nutrient.io/blog/docker-import-export-vs-load-save/

To summarize what you’ve learned:

docker save 
    works with Docker images. It saves everything needed to build a container
    from scratch. Use this command if you want to share an image with others.

docker load
    works with Docker images. Use this command if you want to run an image exported
    with save. Unlike pull, which requires connecting to a Docker registry, load
    can import from anywhere (e.g. a file system, URLs).

docker export
    works with Docker containers, and it exports a snapshot of the container’s file
    system. Use this command if you want to share or back up the result of building
    an image.

docker import
    works with the file system of an exported container, and it imports it as a
    Docker image. Use this command if you have an exported file system you want to
    explore or use as a layer for a new image.




gha : scp an archive somewhere
-------------------------------


* https://stackoverflow.com/questions/60253093/how-do-i-scp-repo-files-using-github-actions

* https://github.com/appleboy/scp-action

* https://github.com/marketplace/actions/transfer-files-in-repository-to-remote-server-via-scp

* https://zellwk.com/blog/github-actions-deploy/


gh
---

* https://github.com/cli/cli#installation

* https://cli.github.com/manual/
* https://cli.github.com/manual/gh_secret_set


* https://dev.to/raulpenate/begginers-guide-installing-and-using-github-cli-30ka

::
 
    sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
    sudo dnf install gh --repo gh-cli   



gh auth
--------

::

    A[blyth@localhost sandbox]$ gh secret set MYSECRET
    To get started with GitHub CLI, please run:  gh auth login
    Alternatively, populate the GH_TOKEN environment variable with a GitHub API authentication token.

* https://cli.github.com/manual/gh_auth_login

A::

    gh auth login --with-token < ~/.ssh/gh.txt


    A[blyth@localhost sandbox]$ gh secret set MYSECRET
    ? Paste your secret: ********

    ✓ Set Actions secret MYSECRET for simoncblyth/sandbox
    A[blyth@localhost sandbox]$ 



github action secrets
-----------------------

* https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions


gitlab ci/cd vs github actions
---------------------------------

* https://graphite.dev/guides/gitlab-cicd--vs-github-actions


gha "uses"  : specifies path to dir with an action.yml
------------------------------------------------------

* https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions#jobsjob_idstepsuses


So the below means there must be an action.yml at top level of the repo::

    uses : ./


use ssh keys in github actions
--------------------------------

* https://github.com/webfactory/ssh-agent
* https://github.com/webfactory/ssh-agent/blob/master/dist/index.js

  action.yml refs 3k+3k lines of node.js to do it 

* https://maxschmitt.me/posts/github-actions-ssh-key
 
  low level approach


* https://zellwk.com/blog/github-actions-deploy/
* https://github.com/marketplace/actions/install-ssh-key
* https://github.com/shimataro/ssh-key-action


ssh key setup : github_action_runner_vm.rst
----------------------------------------------

As key setup include sensitive info, document the setup separately::

   ~/home/admin/ssh/github_action_runner_vm.rst


Check busybox docker image created in GHA_VM in USA and scp to here on A
--------------------------------------------------------------------------

::

    A[blyth@localhost ~]$ docker load -i bb42.tar
    59654b79daad: Loading layer [==================================================>]  4.506MB/4.506MB
    Loaded image: bb42:latest
    A[blyth@localhost ~]$ docker image ls
    REPOSITORY   TAG       IMAGE ID       CREATED        SIZE
    bb42         latest    c9d2aec48d25   5 months ago   4.27MB
    A[blyth@localhost ~]$ 


    A[blyth@localhost ~]$ docker run bb42
    42
    A[blyth@localhost ~]$ docker run -it bb42
    42




try bigger image
------------------

::


    A[blyth@localhost ~]$ docker load -i rl9.tar
    44343de3ea1d: Loading layer [==================================================>]  181.3MB/181.3MB
    Loaded image ID: sha256:9cc24f05f309508aa852967ab1e3b582b302afc92605c24ce27715c683acd805
    A[blyth@localhost ~]$ docker images
    REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
    bb42         latest    c9d2aec48d25   5 months ago    4.27MB
    <none>       <none>    9cc24f05f309   15 months ago   176MB
    A[blyth@localhost ~]$ 


    A[blyth@localhost ~]$ docker run 9cc24f05f309508aa852967ab1e3b582b302afc92605c24ce27715c683acd805
    A[blyth@localhost ~]$ docker run -it 9cc24f05f309508aa852967ab1e3b582b302afc92605c24ce27715c683acd805
    [root@5a3d456ce126 /]# uname -a
    Linux 5a3d456ce126 5.14.0-427.16.1.el9_4.x86_64 #1 SMP PREEMPT_DYNAMIC Thu May 9 18:15:59 EDT 2024 x86_64 x86_64 x86_64 GNU/Linux
    [root@5a3d456ce126 /]# cat /etc/os-release
    NAME="Rocky Linux"
    VERSION="9.3 (Blue Onyx)"
    ID="rocky"
    ID_LIKE="rhel centos fedora"
    VERSION_ID="9.3"
    PLATFORM_ID="platform:el9"
    PRETTY_NAME="Rocky Linux 9.3 (Blue Onyx)"
    ANSI_COLOR="0;32"
    LOGO="fedora-logo-icon"
    CPE_NAME="cpe:/o:rocky:rocky:9::baseos"
    HOME_URL="https://rockylinux.org/"
    BUG_REPORT_URL="https://bugs.rockylinux.org/"
    SUPPORT_END="2032-05-31"
    ROCKY_SUPPORT_PRODUCT="Rocky-Linux-9"
    ROCKY_SUPPORT_PRODUCT_VERSION="9.3"
    REDHAT_SUPPORT_PRODUCT="Rocky Linux"
    REDHAT_SUPPORT_PRODUCT_VERSION="9.3"
    [root@5a3d456ce126 /]# 

    [root@5a3d456ce126 /]# arch
    x86_64
    [root@5a3d456ce126 /]# exit
    exit
    A[blyth@localhost ~]$ cat /etc/os-release
    NAME="AlmaLinux"
    VERSION="9.5 (Teal Serval)"
    ID="almalinux"
    ID_LIKE="rhel centos fedora"
    VERSION_ID="9.5"
    PLATFORM_ID="platform:el9"
    PRETTY_NAME="AlmaLinux 9.5 (Teal Serval)"
    ANSI_COLOR="0;34"
    LOGO="fedora-logo-icon"
    CPE_NAME="cpe:/o:almalinux:almalinux:9::baseos"
    HOME_URL="https://almalinux.org/"
    DOCUMENTATION_URL="https://wiki.almalinux.org/"
    BUG_REPORT_URL="https://bugs.almalinux.org/"

    ALMALINUX_MANTISBT_PROJECT="AlmaLinux-9"
    ALMALINUX_MANTISBT_PROJECT_VERSION="9.5"
    REDHAT_SUPPORT_PRODUCT="AlmaLinux"
    REDHAT_SUPPORT_PRODUCT_VERSION="9.5"
    SUPPORT_END=2032-06-01
    A[blyth@localhost ~]$ 








cuda samples : docker pull kills proxy as expected
----------------------------------------------------

* https://hub.docker.com/r/nvidia/samples/tags


::

    A[blyth@localhost ~]$ docker run --rm --gpus all nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2
    Unable to find image 'nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2' locally
    docker: Error response from daemon: Head "https://nvcr.io/v2/nvidia/k8s/cuda-sample/manifests/vectoradd-cuda10.2": EOF

    Run 'docker run --help' for more information
    A[blyth@localhost ~]$ 



NVIDIA : Example Dockerfiles for the official NVIDIA images published on Docker Hub
-------------------------------------------------------------------------------------

* https://gitlab.com/nvidia/container-images/samples


* https://gitlab.com/nvidia/container-images/samples/-/blob/main/cuda/archive/centos7/cuda-samples/Dockerfile?ref_type=heads


::

    FROM nvidia/cuda:9.0-base-centos7

    RUN yum install -y \
            cuda-samples-$CUDA_PKG_VERSION && \
        rm -rf /var/cache/yum/*

    WORKDIR /usr/local/cuda/samples

    RUN make -j"$(nproc)" -k || true

    CMD ./5_Simulations/nbody/nbody -benchmark -i=10000











