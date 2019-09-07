##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

odocker-source(){ echo $BASH_SOURCE ; }
odocker-vi(){ vi $(odocker-source)  ; }
odocker-env(){  olocal- ; opticks- ; }
odocker-usage(){ cat << EOU

Docker 
==========

Overview
---------

Docker is single-application-centric so perhaps not such a good fit
for Opticks vs LXD/LXC.  As Opticks comprises many hundreds of executables and scripts. 

See Also
-----------

* olxd-  LXD/LXC
* vbx-  Virtualbox


Question : Which container/virtualization tech will allow me to fully test Opticks with different Linux distros? 
------------------------------------------------------------------------------------------------------------------

vbx- 
   virtualbox was real handly for reproducing an Ubuntu-16 issue 
   but virtualbox with access to host GPU for compute+graphics is too difficult, 
   maybe easier with docker or LXC ?



Install
---------

* https://docs.docker.com/install/
* https://hub.docker.com/_/ubuntu


Instructions to try to follow
---------------------------------

* https://devblogs.nvidia.com/gpu-containers-runtime/

* https://developer.ibm.com/linuxonpower/2018/09/19/using-nvidia-docker-2-0-rhel-7/

  Why dont need to install nvidia-docker-2 to get the OCI hook benefits with RHEL/CentOS.


NGC : NVIDIA GPU Cloud
------------------------

* :google:`docker container Titan V`

* https://docs.nvidia.com/ngc/index.html
* https://docs.nvidia.com/ngc/ngc-titan-setup-guide/index.html

  Instructions for Ubuntu only.

* https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)

  * instructions for CentOS


* https://docs.nvidia.com/ngc/ngc-user-guide/singularity.html


CentOS 7 too old ?
~~~~~~~~~~~~~~~~~~~~

nvidia-docker 2.0 requires GNU/Linux x86_64 with kernel version > 3.10



* https://en.wikipedia.org/wiki/CentOS

::

    cat /etc/redhat-release 
    CentOS Linux release 7.5.1804 (Core) 

    Linux localhost.localdomain 3.10.0-862.6.3.el7.x86_64 #1 SMP Tue Jun 26 16:32:21 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux


 


EOU
}
