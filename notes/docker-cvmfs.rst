notes/docker-cvmfs
====================


Dockerfile cvmfs setup runs but no access::

     25 
     26 RUN dnf install -y https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest.noarch.rpm \
     27   && dnf install -y cvmfs \
     28   && mkdir /etc/cvmfs/keys/ihep.ac.cn \
     29   && curl -o /etc/cvmfs/keys/ihep.ac.cn/ihep.ac.cn.pub http://cvmfs-stratum-one.ihep.ac.cn/cvmfs/software/client_configure/ihep.ac.cn/ihep.ac.cn.pub \
     30   && curl -o /etc/cvmfs/domain.d/ihep.ac.cn.conf http://cvmfs-stratum-one.ihep.ac.cn/cvmfs/software/client_configure/ihep.ac.cn.conf \
     31   && echo "CVMFS_REPOSITORIES='sft.cern.ch,juno.ihep.ac.cn,container.ihep.ac.cn'" | tee    /etc/cvmfs/default.local \
     32   && echo "CVMFS_HTTP_PROXY=DIRECT"                                               | tee -a /etc/cvmfs/default.local \
     33   && cat /etc/cvmfs/default.local \
     34   && mkdir -p /cvmfs/sft.cern.ch \
     35   && mkdir -p /cvmfs/juno.ihep.ac.cn \
     36   && mkdir -p /cvmfs/container.ihep.ac.cn;
     37 
     38 
     39 
     40 RUN ls -alst /cvmfs/juno.ihep.ac.cn/
     41 RUN cd  /cvmfs/juno.ihep.ac.cn/ && ls -alst
     42 

What is missing ? 





::

    A[blyth@localhost ~]$ scp L004:g/al9-cvmfs.tar .

    A[blyth@localhost ~]$ docker load -i al9-cvmfs.tar
    9f12f5d8dbb0: Loading layer [==================================================>]  216.6MB/216.6MB
    5f70bf18a086: Loading layer [==================================================>]  1.024kB/1.024kB
    1809e08b82e2: Loading layer [==================================================>]  244.5MB/244.5MB
    Loaded image: al9-cvmfs:latest
    A[blyth@localhost ~]$ 

    A[blyth@localhost ~]$ docker run -it al9-cvmfs 
    42
    A[blyth@localhost ~]$ docker run -it al9-cvmfs bash 
    [root@601f97838478 /]# pwd
    /
    [root@601f97838478 /]# 
    [root@601f97838478 /]# cd /cvmfs/juno.ihep.ac.cn/
    [root@601f97838478 juno.ihep.ac.cn]# ls -alst 
    total 0
    0 drwxr-xr-x. 2 root root  6 Mar 10 13:13 .
    0 drwxr-xr-x. 5 root root 76 Mar 10 13:13 ..
    [root@601f97838478 juno.ihep.ac.cn]# 



* https://cvmfs.readthedocs.io/en/latest/cpt-containers.html


Mounting works::    

    A[blyth@localhost ~]$ docker run -it -v /cvmfs:/cvmfs al9-cvmfs bash 
    [root@5f24d0579c8f /]# cd /cvmfs/juno.ihep.ac.cn/
    [root@5f24d0579c8f juno.ihep.ac.cn]# ls -alst 
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
    [root@5f24d0579c8f juno.ihep.ac.cn]# 




using cvmfs with docker
------------------------

* https://cvmfs.readthedocs.io/en/latest/cpt-containers.html


Accessing CVMFS from Docker locally
-------------------------------------

* https://awesome-workshop.github.io/docker-cms/04-docker-cvmfs/index.html


* https://cvmfs-contrib.github.io/cvmfs-tutorial-2021/02_stratum0_client/
* https://cvmfs-contrib.github.io/cvmfs-tutorial-2021/02_stratum0_client/




