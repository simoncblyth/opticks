6k_server_srun_shakedown
=========================

Getting into Server to test with srun
---------------------------------------

::

   oj6k () 
    { 
        : bash session on the server - eg to check nvidia-smi CUDA version etc;
        srun --partition=junogpu --qos=junoatmgpu --gres=gpu:pro6000:1 --cpus-per-task=1 --mem=4G --pty bash
    }



Issue 0 : Server environment setup
------------------------------------------

HMM the below misses externals setup::

    source /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-vLatest/bashrc

Generate convenience envset.sh for standalone Opticks OK running just like have for OJ running,
so can::

    source /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-vLatest/envset.sh
    cxs_min.sh


Issue 1 : TEST was defined from somewhere(?) to hitlitemerged which is not handled
-------------------------------------------------------------------------------------

Manual fix::

    TEST=medium_scan


Issue 2 : PATH order needs change for sreport
-------------------------------------------------

::

    L[blyth@junogpu001 ~]$ which sreport
    /usr/bin/sreport
    L[blyth@junogpu001 ~]$ echo $PATH | tr ":" "\n"
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/custom4/0.1.8/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/CLHEP/2.4.7.1/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Xercesc/3.2.4/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/custom4/0.1.8/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/CLHEP/2.4.7.1/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Xercesc/3.2.4/bin
    /usr/share/Modules/bin
    /usr/lib64/ccache
    /usr/local/bin
    /usr/bin
    /usr/local/sbin
    /usr/sbin
    /cvmfs/common.ihep.ac.cn/software/cctools
    /opt/puppetlabs/bin
    /usr/local/cuda-13.1/bin
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/bin
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/lib



Manual fix::

    L[blyth@junogpu001 oj]$ export PATH=/cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/bin:$PATH
    L[blyth@junogpu001 oj]$ export PATH=/cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/lib:$PATH
    L[blyth@junogpu001 oj]$ echo $PATH | tr ":" "\n"
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/lib
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/custom4/0.1.8/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/CLHEP/2.4.7.1/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Xercesc/3.2.4/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/custom4/0.1.8/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/CLHEP/2.4.7.1/bin
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Xercesc/3.2.4/bin
    /usr/share/Modules/bin
    /usr/lib64/ccache
    /usr/local/bin
    /usr/bin
    /usr/local/sbin
    /usr/sbin
    /cvmfs/common.ihep.ac.cn/software/cctools
    /opt/puppetlabs/bin
    /usr/local/cuda-13.1/bin
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/bin
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/lib

    L[blyth@junogpu001 oj]$ which sreport
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.5/lib/sreport
    L[blyth@junogpu001 oj]$ 


Issue 3 : Output /tmp dir not accessible from L
--------------------------------------------------

Need to prefix with /hpcfs/juno/junogpu/blyth/tmp for example

::

    L[blyth@junogpu001 oj]$ TMP=$HOME/tmp cxs_min.sh info  


Grab the sreport to workstation for examination
-------------------------------------------------

::

    [lo] A[blyth@localhost opticks]$ G=X9 sysrap/tests/sreport_ab.sh grep
    sysrap/tests/sreport_ab.sh : rsync G[X9] [/hpcfs/juno/junogpu/blyth/tmp/GEOM/J26_1_1_Opticks_v0_6_3/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan_sreport] from [LD] : enter YES to proceed : YES


G4 looks 50% slower than G3 ?  Must be some different config impact ?
-----------------------------------------------------------------------

::

    L[blyth@junogpu001 oj]$ export OPTICKS_MAX_PHOTON=M100                                                                      │
    L[blyth@junogpu001 oj]$ export OPTICKS_MAX_BOUNCE=31                                                                        │
    L[blyth@junogpu001 oj]$ cxs_min.sh


Using tmux to follow nvtop during the run shows it pulling 50G : which is too much.
Must be some different config in use, such as using default MaxSlot - which
will cause VRAM filling photon buffer to be allocated.

cxs_min.sh script was not sensitive to OPTICKS_MAX_SLOT


TODO : revive Opticks build on lxlogin - so can test without release
----------------------------------------------------------------------------


TODO : simple text table of scan timings for debugging
-----------------------------------------------------------



