nsight-source(){ echo $BASH_SOURCE ; }
nsight-vi(){     vi $BASH_SOURCE  ; }
nsight-env(){  nsight-systems-export ; }  # set PATH to find binaries
nsight-usage(){ cat << EOU

NVIDIA Nsight : Profiling/Debugging Tools
===========================================

nvprof and nvvp : old tools being replaced
----------------------------------------------

::

    which nvprof
    /usr/local/cuda-10.1/bin/nvprof

    which nvvp
    /usr/local/cuda-10.1/bin/nvvp


NVTX : NVIDIA Tools Extension : customize timeline
------------------------------------------------------------

* https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/

  * setup profiled ranges with colors to appear in timeline by using nvToolsExt.h and libnvToolsExt.so 

Overview
---------

* https://devblogs.nvidia.com/transitioning-nsight-systems-nvidia-visual-profiler-nvprof/

The Nsight tools are replacing NVIDIA Visual Profiler (nvvp), nvprof

Nsight Systems and Nsight Compute split system-level application analysis 
and individual CUDA kernel-level profiling into separate tools

Start with Nsight Systems to get a system-level overview of the workload and
eliminate any system level bottlenecks, such as unnecessary thread
synchronization or data movement, and improve the system level parallelism of
your algorithms. Once you have done that, then proceed to Nsight Compute or
Nsight Graphics to optimize the most significant CUDA kernels or graphics
workloads, respectively. 

Blog Articles
--------------

* https://devblogs.nvidia.com/migrating-nvidia-nsight-tools-nvvp-nvprof/


NVIDIA Nsight Systems : Tracing CUDA APIs and CPU Sampling : system level view : big picture
------------------------------------------------------------------------------------------------

* https://devblogs.nvidia.com/transitioning-nsight-systems-nvidia-visual-profiler-nvprof/

* https://developer.nvidia.com/nsight-systems


Binaries
---------

nsys
    comandline tool, eg:: 

       nsys profile -o noprefetch --stats=true ./add_cuda 
 
nsight-sys
    GUI, 

    * File > Open > and navigate to eg noprefetch.qdrep file 

    * hover cursor in timeline to see tooltips of callstack 


::

    [blyth@localhost nsight_systems]$ which nsys
    /opt/nvidia/nsight-systems/2019.5.1/bin/nsys

    [blyth@localhost nsight_systems]$ which nsight-sys
    /opt/nvidia/nsight-systems/2019.5.1/bin/nsight-sys



NVIDIA Nsight Compute : Metrics and Events of single kernel
------------------------------------------------------------------

* https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/


NVIDIA Nsight Graphics
----------------------


NVTX
-------

* https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
* https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx
* https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts/nvtx

Looks like two headers but only one lib::

    [blyth@localhost cuda-10.1]$ find . -name  nvToolsExt.h -exec ls -l {} \;
    -rw-r--r--. 1 root root 52247 Apr  8 14:43 ./targets/x86_64-linux/include/nvtx3/nvToolsExt.h
    -rw-r--r--. 1 root root 52247 Apr  8 14:43 ./NsightSystems-2018.3/Target-x86_64/nvtx/include/nvtx3/nvToolsExt.h

    -rw-r--r--. 1 root root 53680 Apr  8 14:43 ./targets/x86_64-linux/include/nvToolsExt.h

    [blyth@localhost cuda-10.1]$ find . -name 'libnvToolsExt.so'
    ./targets/x86_64-linux/lib/libnvToolsExt.so

    [blyth@localhost cuda-10.1]$ pwd
    /usr/local/cuda-10.1

    [blyth@localhost nsight_systems]$ nsight-
    [blyth@localhost nsight_systems]$ find /opt/nvidia/ -name nvToolsExt.h
    /opt/nvidia/nsight-systems/2019.5.1/target-linux-x64/nvtx/include/nvtx3/nvToolsExt.h



* https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm

More information is available as comments in nvToolsExt.h. NVTX v3 is a header-only C library. 
The headers ship with CUDA Toolkit and Nsight tools such as Nsight Graphics and Nsight Systems.



EOU
}


nsight-systems-dir(){ echo /opt/nvidia/nsight-systems/2019.5.1 ; } 
nsight-systems-export(){ PATH=$(nsight-systems-dir)/bin:$PATH ; }
nsight-systems-notes(){ cat << EON
$FUNCNAME
=======================
Installation
--------------

silver.precision::

    [root@gilda03 Downloads]# bash NVIDIA_Nsight_Systems_Linux_2019.5.1.58.run

    To uninstall the Nsight Systems 2019.5.1, please delete "/opt/nvidia/nsight-systems/2019.5.1"

gold.precision::

    [root@localhost Downloads]# bash NVIDIA_Nsight_Systems_Linux_2019.5.1.58.run
    Verifying archive integrity...  100%   All good.
    Uncompressing NVIDIA Nsight Systems  100%  
    Can't locate Env.pm in @INC (@INC contains: /usr/local/lib64/perl5 /usr/local/share/perl5 /usr/lib64/perl5/vendor_perl /usr/share/perl5/vendor_perl /usr/lib64/perl5 /usr/share/perl5 .) at ./install-linux.pl line 10.
    BEGIN failed--compilation aborted at ./install-linux.pl line 10.
    [root@localhost Downloads]# 

    [root@localhost Downloads]# yum install perl-Env

    After that the install works

    To uninstall the Nsight Systems 2019.5.1, please delete "/opt/nvidia/nsight-systems/2019.5.1"


EON
}


nsight-permfix-notes(){ cat << EON

https://developer.nvidia.com/ERR_NVGPUCTRPERM

::

    (base) [blyth@gilda03 nv]$ nvprof ./add_cuda
    ==11376== NVPROF is profiling process 11376, command: ./add_cuda
    Max error: 0
    ==11376== Warning: The user does not have permission to profile on the target device. 
              See the following link for instructions to enable permissions and get more information: 
             https://developer.nvidia.com/NVSOLN1000 



* some docs suggested "modprobe" rather than "options" in the conf file but that doesnt work
* have to reboot following changes to the conf

::

    (base) [blyth@gilda03 nv]$ modinfo nvidia
    filename:       /lib/modules/3.10.0-957.27.2.el7.x86_64/kernel/drivers/video/nvidia.ko
    alias:          char-major-195-*
    version:        435.21
    supported:      external
    license:        NVIDIA
    retpoline:      Y
    rhelversion:    7.6
    ...
    parm:           NVreg_RestrictProfilingToAdminUsers:int
    ...

::

    [blyth@localhost modprobe.d]$ su
    [root@localhost modprobe.d]# . ~blyth/opticks/bin/nsight.bash



EON
}

nsight-permfix(){  $FUNCNAME- > /etc/modprobe.d/$FUNCNAME.conf ; }
nsight-permfix-(){ cat << EOF
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
}



