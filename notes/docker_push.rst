notes/docker_push
===================



docker access tokens
---------------------

* https://docs.docker.com/security/for-developers/access-tokens/

You can create a personal access token (PAT) to use as an alternative to your
password for Docker CLI authentication.

::

    name: ci

    on:
      push:
        branches: main

    jobs:
      login:
        runs-on: ubuntu-latest
        steps:
          -
            name: Login to Docker Hub
            uses: docker/login-action@v3
            with:
              username: ${{ vars.DOCKERHUB_USERNAME }}
              password: ${{ secrets.DOCKERHUB_TOKEN }}



login
-------



* https://github.com/docker/login-action
* https://github.com/docker/login-action#docker-hub



docker image history --no-trunc
----------------------------------

For checking the sizes and commands of each layer::

    docker image history junosw/cuda:12.4.1-runtimeplus-rockylinux9 --no-trunc  
    docker image history junosw/cuda:12.4.1-runtimeplus-rockylinux9

::

    A[blyth@localhost ~]$ docker image history junosw/cuda:12.4.1-runtimeplus-rockylinux9 
    IMAGE          CREATED         CREATED BY                                      SIZE      COMMENT
    3d505c100ea8   2 days ago      WORKDIR /home/juno                              0B        buildkit.dockerfile.v0
    <missing>      2 days ago      USER juno                                       0B        buildkit.dockerfile.v0
    <missing>      2 days ago      RUN |1 PASSWORD= /bin/sh -c dnf clean all &&…   616kB     buildkit.dockerfile.v0
    <missing>      2 days ago      RUN |1 PASSWORD= /bin/sh -c sudo dnf install…   42.2MB    buildkit.dockerfile.v0
    <missing>      2 days ago      RUN |1 PASSWORD= /bin/sh -c sudo dnf install…   41.7MB    buildkit.dockerfile.v0
    <missing>      2 days ago      RUN |1 PASSWORD= /bin/sh -c sudo dnf install…   41.4MB    buildkit.dockerfile.v0
    <missing>      2 days ago      RUN |1 PASSWORD= /bin/sh -c sudo dnf install…   44MB      buildkit.dockerfile.v0


naming convention
-------------------

* https://docs.docker.com/docker-hub/repos/create/

build and push
---------------

* https://docs.docker.com/get-started/docker-concepts/building-images/build-tag-and-publish-an-image/
* https://github.com/simoncblyth/sandbox/blob/master/.github/workflows/simoncblyth-build-docker-image-and-push.yml
* https://hub.docker.com/repository/docker/simoncblyth/cuda/tags/12.4.1-runtimeplus-rockylinux9/sha256-407212eab24e6c57be154779d2e1f175d95f8d2bcc0c61c4b980390eb8bb3a42?tab=layers


pull and scp
-------------

* https://github.com/simoncblyth/sandbox/blob/master/.github/workflows/simoncblyth-pull-docker-image-and-scp.yml


::

   docker pull simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9




load and inepect
-----------------

::

    A[blyth@localhost ~]$ scp L:g/simoncblyth_cuda_12_4_1_runtimeplus_rockylinux9.tar .


    A[blyth@localhost ~]$ docker load -i simoncblyth_cuda_12_4_1_runtimeplus_rockylinux9.tar
    286c48effa59: Loading layer [==================================================>]   3.23GB/3.23GB
    e421aba41390: Loading layer [==================================================>]  19.97kB/19.97kB
    1bfb048369bd: Loading layer [==================================================>]  5.632kB/5.632kB
    2ff4b71ed438: Loading layer [==================================================>]  3.072kB/3.072kB
    7634be47b369: Loading layer [==================================================>]  48.62MB/48.62MB
    91785ffe81b6: Loading layer [==================================================>]  166.4kB/166.4kB
    dae07dbdebb6: Loading layer [==================================================>]  31.33MB/31.33MB
    2bd5281209c1: Loading layer [==================================================>]  92.66MB/92.66MB
    d1d9965f5409: Loading layer [==================================================>]  1.683GB/1.683GB
    408d0eb08079: Loading layer [==================================================>]  6.656kB/6.656kB
    0138d012cebf: Loading layer [==================================================>]  176.4MB/176.4MB
    9c6222c29043: Loading layer [==================================================>]   41.8MB/41.8MB
    f6e87cb7782b: Loading layer [==================================================>]  43.99MB/43.99MB
    7c58d6fe155b: Loading layer [==================================================>]  41.38MB/41.38MB
    efb6484faf3d: Loading layer [==================================================>]  41.79MB/41.79MB
    9bc82fdfa6e0: Loading layer [==================================================>]  42.15MB/42.15MB
    bc3b8e8e02f1: Loading layer [==================================================>]  661.5kB/661.5kB
    5f70bf18a086: Loading layer [==================================================>]  1.024kB/1.024kB
    Loaded image: simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9
    A[blyth@localhost ~]$ 


        

    A[blyth@localhost ~]$ docker inspect simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9
    [
        {
            "Id": "sha256:8c10f253f281986904bef377325f49fe794d780a133cb6e784a289ba4ed114fd",
            "RepoTags": [
                "simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9"
            ],
            "RepoDigests": [],
            "Parent": "",
            "Comment": "buildkit.dockerfile.v0",
            "Created": "2025-03-15T10:55:18.238287846Z",
            "DockerVersion": "",
            "Author": "",
            "Config": {
                "Hostname": "",
                "Domainname": "",
                "User": "juno",
                "AttachStdin": false,
                "AttachStdout": false,
                "AttachStderr": false,
                "Tty": false,
                "OpenStdin": false,
                "StdinOnce": false,
                "Env": [
                    "PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "NVARCH=x86_64",
                    "NVIDIA_REQUIRE_CUDA=
                      cuda>=12.4 
                      brand=tesla,
                           driver>=470,driver<471 
                       brand=unknown,
                           driver>=470,driver<471 
                       brand=nvidia,
                           driver>=470,driver<471 
                       brand=nvidiartx,
                           driver>=470,driver<471 
                       brand=geforce,
                           driver>=470,driver<471 
                       brand=geforcertx,
                           driver>=470,driver<471 
                       brand=quadro,
                           driver>=470,driver<471 
                       brand=quadrortx,
                           driver>=470,driver<471 
                       brand=titan,
                           driver>=470,driver<471 
                       brand=titanrtx,
                           driver>=470,driver<471 

                       brand=tesla,
                           driver>=525,driver<526 
                       brand=unknown,
                           driver>=525,driver<526 
                       brand=nvidia,
                           driver>=525,driver<526 
                       brand=nvidiartx,
                           driver>=525,driver<526 
                       brand=geforce,
                           driver>=525,driver<526 
                       brand=geforcertx,
                           driver>=525,driver<526 
                       brand=quadro,
                           driver>=525,driver<526
                       brand=quadrortx,
                           driver>=525,driver<526
                       brand=titan,
                           driver>=525,driver<526
                       brand=titanrtx,
                           driver>=525,driver<526 

                       brand=tesla,
                           driver>=535,driver<536 
                       brand=unknown,
                           driver>=535,driver<536 
                       brand=nvidia,
                           driver>=535,driver<536 
                       brand=nvidiartx,
                           driver>=535,driver<536
                       brand=geforce,
                           driver>=535,driver<536 
                       brand=geforcertx,
                           driver>=535,driver<536 
                       brand=quadro,
                           driver>=535,driver<536 
                       brand=quadrortx,
                           driver>=535,driver<536
                       brand=titan,
                           driver>=535,driver<536 
                       brand=titanrtx,
                           driver>=535,driver<536",
                    "NV_CUDA_CUDART_VERSION=12.4.127-1",
                    "CUDA_VERSION=12.4.1",
                    "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
                    "NVIDIA_VISIBLE_DEVICES=all",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                    "NV_CUDA_LIB_VERSION=12.4.1-1",
                    "NV_NVTX_VERSION=12.4.127-1",
                    "NV_LIBNPP_VERSION=12.2.5.30-1",
                    "NV_LIBNPP_PACKAGE=libnpp-12-4-12.2.5.30-1",
                    "NV_LIBCUBLAS_VERSION=12.4.5.8-1",
                    "NV_LIBNCCL_PACKAGE_NAME=libnccl",
                    "NV_LIBNCCL_PACKAGE_VERSION=2.21.5-1",
                    "NV_LIBNCCL_VERSION=2.21.5",
                    "NCCL_VERSION=2.21.5",
                    "NV_LIBNCCL_PACKAGE=libnccl-2.21.5-1+cuda12.4",
                    "NVIDIA_PRODUCT_NAME=CUDA",
                    "NV_CUDA_CUDART_DEV_VERSION=12.4.127-1",
                    "LIBRARY_PATH=/usr/local/cuda/lib64/stubs"
                ],
                "Cmd": null,
                "Image": "",
                "Volumes": null,
                "WorkingDir": "/home/juno",
                "Entrypoint": [
                    "/opt/nvidia/nvidia_entrypoint.sh"
                ],
                "OnBuild": null,
                "Labels": {
                    "maintainer": "NVIDIA CORPORATION <sw-cuda-installer@nvidia.com>",
                    "src": "https://github.com/simoncblyth/sandbox/blob/master/junosw/Dockerfile-junosw-cuda-runtimeplus-el9"
                }
            },
            "Architecture": "amd64",
            "Os": "linux",
            "Size": 7888506169,
            "GraphDriver": {
                "Data": {
                    "LowerDir": "/var/lib/docker/overlay2/65fae8d95961b915e5b09b506eaf9eab1784c7f07dce285f1374460893e7c8d0/diff:/var/lib/docker/overlay2/cdcd90c41793b4bc36ddb342965573afd58bea6f7619966dcb5afa0971ce62e5/diff:/var/lib/docker/overlay2/6b0c874d30fa9809d6679964a8165a50101b38724362e8c6ec4028bd0cb4ad5c/diff:/var/lib/docker/overlay2/fc81beb20b77254a5609f2649c45edac7ee75242897b9a7f4717c46d249991f4/diff:/var/lib/docker/overlay2/4d198f70df1b54eb32c7f75eac8b6ccee059512d2043f5f801aa88bb99af3b5c/diff:/var/lib/docker/overlay2/211ef79d54a39ee63fecd4f749f0606f1ad46a4af9caa6ff9e28c4634c93671c/diff:/var/lib/docker/overlay2/8c31738ab60f9354ea8ea3bc65c12c2d763c4ceb6faa115b9fde97b2e588ea54/diff:/var/lib/docker/overlay2/9ac4dd02fbff385a0330f9d1d42ef276fb11c85abf644d3208cfd9431ca1b90d/diff:/var/lib/docker/overlay2/1b7230a8e86b6954604644adf64fc816a9d10cd4768b0f74ec1549f646fdb90e/diff:/var/lib/docker/overlay2/c8ad953663a3c4ada8afbc36c84909744dac5f1b304077a62dc4613514f1ffe1/diff:/var/lib/docker/overlay2/b7874a138e00f02249952b214a777e4fc3ac7e99c5571daa743948ed51aa5619/diff:/var/lib/docker/overlay2/6f48540089e3b88418f7a6f818d22e91ad793347503dea2cc705eee4b44e5486/diff:/var/lib/docker/overlay2/a284b27edc5f26fd88e5c68c9e78f2b581ca19d19c41d6daaf9a968b5c403cdf/diff:/var/lib/docker/overlay2/d3f7d1dccda9103065473fedea33a0a84d6fc9dc2ef3487828879a5944c1f0d4/diff:/var/lib/docker/overlay2/a91790d8a0fdb950786beceec6b289c0eaf57acb5e7ac3f76de11e3a250896a8/diff:/var/lib/docker/overlay2/5a7282a0086a57a1b6192d10d043edf71f515425e8621491899d82bea67b0b21/diff:/var/lib/docker/overlay2/9026eed0b8239f498ed5ba290a4e7053c4f88a77b83fae2d09f38b65f62f14aa/diff:/var/lib/docker/overlay2/8edda37cc43d331b6c1c947f7ca8add3d816c0eb1c7e625daf8bf058fc487105/diff:/var/lib/docker/overlay2/9a7cb01a39511aff3bf9da0ea88397e08517fbb0211b919e9f5f453cb00ca429/diff:/var/lib/docker/overlay2/70e45378687ae51feac9b6bbc93a5222232020d72cfeeaae1a889294ad8c4491/diff:/var/lib/docker/overlay2/ada26bc69a44df725a1fcc627c540351cb92332e741822d1a59bae5c0e0d19da/diff:/var/lib/docker/overlay2/35ac7fa27049068cf8c43bce0db9bdef7bf59ada5fc5619fb567e53efe59ce73/diff:/var/lib/docker/overlay2/c3cdf3b13aade8a7f3c11c777c180a165cabf548c665d17fab2bb2eb29f1616a/diff:/var/lib/docker/overlay2/87ea9709c4da7d006461de71d7408f3df08ad0228d12145a56c5571b66558667/diff:/var/lib/docker/overlay2/ddb12d7146d7681bbb25bd4e32ea218dc2312ae71503d6194db5ebf5fe78c33d/diff:/var/lib/docker/overlay2/31d50874f7afce4f1b59f448a31b134aca6b63582d7779d12df90148168b5815/diff",
                    "MergedDir": "/var/lib/docker/overlay2/0110e2191bb46b5f4a8044fe87d6d587d9a688646a9a43702ad378dd806b77f1/merged",
                    "UpperDir": "/var/lib/docker/overlay2/0110e2191bb46b5f4a8044fe87d6d587d9a688646a9a43702ad378dd806b77f1/diff",
                    "WorkDir": "/var/lib/docker/overlay2/0110e2191bb46b5f4a8044fe87d6d587d9a688646a9a43702ad378dd806b77f1/work"
                },
                "Name": "overlay2"
            },
            "RootFS": {
                "Type": "layers",
                "Layers": [
                    "sha256:c4bc4a1387e82c199a05c950a61d31aba8e1481a94c63196b82e25ac8367e5d1",
                    "sha256:29cf88fb44d49471d46488dc9efdbfac918043dcaf57c3486c06d2452490d385",
                    "sha256:dd00f6980f231b5e661fb1d93c48b68cbcdc9d690510dac8b6b2fba47cb5a073",
                    "sha256:8b5530c65e239967c28eacaeccd889aea1b885126b6a4d32f690b440fa164cab",
                    "sha256:5152f26b2054950f6a1e06c47f80f310ef896835df7613076c9e664691b29916",
                    "sha256:04d6e2e7cd5cd9256dfe7bf878f139b0f670f08ef06326dcf6546667ee613f3b",
                    "sha256:55c5c28332fe6981eafc318a64693acbc904d512a4576fa98037ddfac96935d8",
                    "sha256:8bf266c350f2a8b0209162e1bc95fbc1f55c7aa90051f5b3d56644f0c08ea33a",
                    "sha256:1911f832adb7f0960125b9daedec7cd92dd2219df22f4e7ca6f0e892d6d837ec",
                    "sha256:286c48effa593038d2306a96bf2628b6328dd4aa45930c792a39e8e88b94840e",
                    "sha256:e421aba413903017261f94968907adfebd9fccc331f881baeb5159b564f205a8",
                    "sha256:1bfb048369bd8bd5d0ebeaa3ea73db01e2bd3dddebca127d40f8be99d3df9a49",
                    "sha256:2ff4b71ed438712ca5ff6c1ff2a8e71a3731fe87add22c90c7675e703ab3bc60",
                    "sha256:7634be47b3693313831a13e2305163fb3ff43a6bce81b3e2d7d6f2e2b8e04ac4",
                    "sha256:91785ffe81b6ca6dc64630c1e9e3cd1ffdc49ef9426ddc7607e8274cb40a8172",
                    "sha256:dae07dbdebb6a9c2ffe83ef176b9769233baa8492213f291904be2d3d6096d21",
                    "sha256:2bd5281209c152c4692b1ed7c59825eab6c9fea2216f048452c6864bad7c8640",
                    "sha256:d1d9965f5409ed274779556dd9ab62cb515165156a62b5da14c1a14a2875c6ed",
                    "sha256:408d0eb08079250fe67dcfab208219409a12d27303f3630d000678313239de2e",
                    "sha256:0138d012cebf8c590e16c9312d9a2269bc0112298db72bded0a61d7976116ad8",
                    "sha256:9c6222c2904321b3e573d5bc7a7d38bfd38a070296fe3f8b3f43795d22d35b1b",
                    "sha256:f6e87cb7782b72e8c9ff1d75921f53dbd36db63005a7852e84e2b1144bb2eeee",
                    "sha256:7c58d6fe155b0da8d06433effa79544db7431a7a37693c7535ce52bd41e28cbe",
                    "sha256:efb6484faf3d6123f96ff9e9fcf2e7203a0926369c27af6fe7a8b5130b4ab0a7",
                    "sha256:9bc82fdfa6e074d00762382b547a452b39239ad541fffa17ac0d961ba701af67",
                    "sha256:bc3b8e8e02f1dccfe3b8a50ea4f409e33386ae334ba424148c6b62070d510aa6",
                    "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef"
                ]
            },
            "Metadata": {
                "LastTagTime": "0001-01-01T00:00:00Z"
            }
        }
    ]



 

NVIDIA_REQUIRE_CUDA
--------------------

* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
* https://stackoverflow.com/questions/75029780/older-driver-newer-cuda-toolkit-leads-to-container-startup-failure-any-config


::

                "Env": [
                    "PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "NVARCH=x86_64",
                    "NVIDIA_REQUIRE_CUDA=
                      cuda>=12.4 
                      brand=tesla,
                           driver>=470,driver<471 
                       brand=unknown,
                           driver>=470,driver<471 
                       brand=nvidia,
                           driver>=470,driver<471 
                       brand=nvidiartx,
                           driver>=470,driver<471 
                       brand=geforce,
                           driver>=470,driver<471 
                       brand=geforcertx,
                           driver>=470,driver<471 
                       brand=quadro,
                           driver>=470,driver<471 
                       brand=quadrortx,
                           driver>=470,driver<471 
                       brand=titan,
                           driver>=470,driver<471 
                       brand=titanrtx,
                           driver>=470,driver<471 

                       brand=tesla,
                           driver>=525,driver<526 
                       brand=unknown,
                           driver>=525,driver<526 
                       brand=nvidia,
                           driver>=525,driver<526 
                       brand=nvidiartx,
                           driver>=525,driver<526 
                       brand=geforce,
                           driver>=525,driver<526 
                       brand=geforcertx,
                           driver>=525,driver<526 
                       brand=quadro,
                           driver>=525,driver<526
                       brand=quadrortx,
                           driver>=525,driver<526
                       brand=titan,
                           driver>=525,driver<526
                       brand=titanrtx,
                           driver>=525,driver<526 

                       brand=tesla,
                           driver>=535,driver<536 
                       brand=unknown,
                           driver>=535,driver<536 
                       brand=nvidia,
                           driver>=535,driver<536 
                       brand=nvidiartx,
                           driver>=535,driver<536
                       brand=geforce,
                           driver>=535,driver<536 
                       brand=geforcertx,
                           driver>=535,driver<536 
                       brand=quadro,
                           driver>=535,driver<536 
                       brand=quadrortx,
                           driver>=535,driver<536
                       brand=titan,
                           driver>=535,driver<536 
                       brand=titanrtx,
                           driver>=535,driver<536",
                    "NV_CUDA_CUDART_VERSION=12.4.127-1",
                    "CUDA_VERSION=12.4.1",
                    "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
                    "NVIDIA_VISIBLE_DEVICES=all",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                    "NV_CUDA_LIB_VERSION=12.4.1-1",
                    "NV_NVTX_VERSION=12.4.127-1",




build warnings
---------------

::
        
    [ 28%] Linking CXX shared library ../../../lib/libTopTracker.so
    Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with __GLIBCXX__Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with __GLIBCXX__Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with __GLIBCXX____GLIBCXX____GLIBCXX____GLIBCXX__ 'Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with __GLIBCXX____GLIBCXX____GLIBCXX__Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with __GLIBCXX__ ' 'Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with 20231218__GLIBCXX__ '__GLIBCXX____GLIBCXX__ '__GLIBCXX__ ' 'Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with __GLIBCXX__20231218__GLIBCXX__ 'Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with  ' '__GLIBCXX__2023121820231218'
      Extraction of runtime standard library version was: ''
      Extraction of runtime standard library version was: ' ' ' 'Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with 202312182023121820231218 '20231218__GLIBCXX__Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with  ' '__GLIBCXX__'
      Extraction of runtime standard library version was: ' '2023121820231218Warning in cling::IncrementalParser::CheckABICompatibility():
      Possible C++ standard library mismatch, compiled with __GLIBCXX__20231218 ''
      Extraction of runtime standard library version was: ''
      Extraction of runtime standard library version was: '20240719202407192024071920231218'
    '



adding tag 
------------

::

    A[blyth@localhost junosw]$ docker tag simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9 junosw/cuda:el9
    A[blyth@localhost junosw]$ docker images
    REPOSITORY                                     TAG                              IMAGE ID       CREATED         SIZE
    junosw/cuda                                    el9                              8c10f253f281   43 hours ago    7.89GB
    simoncblyth/cuda                               12.4.1-runtimeplus-rockylinux9   8c10f253f281   43 hours ago    7.89GB
    junosw/cuda                                    12.4.1-runtimeplus-rockylinux9   3d505c100ea8   4 days ago      7.89GB
    junosw/cuda                                    12.4.1-runtime-rockylinux9       3b3a3332ae87   4 days ago      5.81GB
    junosw/base                                    el9                              987e8bddae3e   5 days ago      2.51GB
    al9-cvmfs                                      latest                           ebccb0ed032b   6 days ago      451MB
    nvidia_cuda_12_4_1_runtime_rockylinux9_amd64   latest                           72c9d5a2da10   6 days ago      2.47GB
    bb42                                           latest                           c9d2aec48d25   5 months ago    4.27MB
    nvidia/cuda                                    12.4.1-devel-rockylinux9         ab9135746936   11 months ago   7.11GB
    <none>                                         <none>                           9cc24f05f309   15 months ago   176MB
    <none>                                         <none>                           0fed15e4f2a2   16 months ago   2.69GB
    A[blyth@localhost junosw]$ 

    A[blyth@localhost junosw]$ docker rm junosw/cuda:el9
    Error response from daemon: No such container: junosw/cuda:el9
    A[blyth@localhost junosw]$ docker rmi junosw/cuda:el9
    Untagged: junosw/cuda:el9

   
    A[blyth@localhost junosw]$ docker tag simoncblyth/cuda:12.4.1-runtimeplus-rockylinux9 junosw/cuda:12.4.1-el9
    A[blyth@localhost junosw]$ docker images
    REPOSITORY                                     TAG                              IMAGE ID       CREATED         SIZE
    junosw/cuda                                    12.4.1-el9                       8c10f253f281   44 hours ago    7.89GB
    simoncblyth/cuda                               12.4.1-runtimeplus-rockylinux9   8c10f253f281   44 hours ago    7.89GB
    junosw/cuda                                    12.4.1-runtimeplus-rockylinux9   3d505c100ea8   4 days ago      7.89GB
    junosw/cuda                                    12.4.1-runtime-rockylinux9       3b3a3332ae87   4 days ago      5.81GB
    junosw/base                                    el9                              987e8bddae3e   5 days ago      2.51GB
    al9-cvmfs                                      latest                           ebccb0ed032b   6 days ago      451MB
    nvidia_cuda_12_4_1_runtime_rockylinux9_amd64   latest                           72c9d5a2da10   6 days ago      2.47GB
    bb42                                           latest                           c9d2aec48d25   5 months ago    4.27MB
    nvidia/cuda                                    12.4.1-devel-rockylinux9         ab9135746936   11 months ago   7.11GB
    <none>                                         <none>                           9cc24f05f309   15 months ago   176MB
    <none>                                         <none>                           0fed15e4f2a2   16 months ago   2.69GB
    A[blyth@localhost junosw]$ 


  



 
