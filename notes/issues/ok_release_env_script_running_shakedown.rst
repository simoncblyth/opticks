ok_release_env_script_running_shakedown
=========================================

Overview 
---------- 

Establish workflow to test running within release env without actually doing releases


TODO
-----

* get some more high profile scripts to work within release env
* run ctests on the release : did this before, try revival 


Make test release tarball and explode it
--------------------------------------------

1. oo # update build
2. okdist-;okdist--  ## make distrib tarball and explode it 

::

    === okdist-tarball-extract
    [2025-04-16 15:42:46,387] p195943 {/home/blyth/opticks/bin/oktar.py:251} INFO - extracting tarball with common prefix el7_amd64_gcc1120/Opticks-v0.3.5 into base /data/blyth/opticks_Debug 


Workflow after adding something to the release
-----------------------------------------------

In build env:

* build the package with the addition 
* rerun okdist--

In release env:

* try again the command that failed



Get into release env from the exploded tarball
------------------------------------------------

Make sure Opticks env not enabled, then  get into test Release env on P with::

    source /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bashrc 

    === opticks-setup-        add     append                 PATH /usr/local/cuda-11.7/bin
    === opticks-setup-        add     append                 PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin
    === opticks-setup-        add     append                 PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/lib
    === opticks-setup-        new     append    CMAKE_PREFIX_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5
    === opticks-setup-        add     append    CMAKE_PREFIX_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/externals
    === opticks-setup-        add     append    CMAKE_PREFIX_PATH /home/blyth/local/opticks/externals/OptiX_750
    === opticks-setup-      nodir     append      PKG_CONFIG_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/lib/pkgconfig
    === opticks-setup-        new     append      PKG_CONFIG_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/lib64/pkgconfig
    === opticks-setup-        add     append      PKG_CONFIG_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/externals/lib/pkgconfig
    === opticks-setup-        add     append      PKG_CONFIG_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/externals/lib64/pkgconfig
    === opticks-setup-        add     append      LD_LIBRARY_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/lib
    === opticks-setup-        add     append      LD_LIBRARY_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/lib64
    === opticks-setup-        add     append      LD_LIBRARY_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/externals/lib
    === opticks-setup-        add     append      LD_LIBRARY_PATH /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/externals/lib64
    === opticks-setup-      nodir     append      LD_LIBRARY_PATH /usr/local/cuda-11.7/lib
    === opticks-setup-        add     append      LD_LIBRARY_PATH /usr/local/cuda-11.7/lib64
    === opticks-setup-geant4- : ERROR no
    P[blyth@localhost junosw]$ echo $OPTICKS_PREFIX
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5


Note Geant4 not setup. Contrast with the jsw build env::

    === opticks-setup-geant4- : sourcing /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/bin/geant4.sh



cxr_min.sh viz : WORKS AFTER EXTERNAL GEOM SETUP + CVD
----------------------------------------------------------

::

    P[blyth@localhost junosw]$ which cxr_min.sh 
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin/cxr_min.sh

Hookup external geometry and pick the GPU::

    xgeom(){
      export GEOM=J25_3_0_Opticks_v0_3_5 
      export ${GEOM}_CFBaseFromGEOM=/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.3.0_Opticks-v0.3.5/el9_amd64_gcc11/2025_04_14/.opticks/GEOM/J25_3_0_Opticks_v0_3_5
    }

    xgeom   
    export CUDA_VISIBLE_DEVICES=1
    cxr_min.sh   





Where to set PYTHONPATH ? Needs to be in the opticks bashrc
--------------------------------------------------------------

::

    export PYTHONPATH=$OPTICKS_PREFIX/py:$PYTHONPATH



.gitlab-ci.yml::

    JUNO_OPTICKS_PREFIX: /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-vLatest


.gitlab-ci/oj_helper.sh::

    094 elif [ "$arg" == "EMIT_ENVSET" ]; then
     95 
     96     cat << EOS
     97 #!/bin/bash
     98 # generated from .gitlab-ci.yml of $CI_PROJECT_PATH
     99 # $BASH_SOURCE $arg
    100 
    101 HERE=\$(dirname \$(realpath \$BASH_SOURCE))
    102 DEFAULT_PREFIX=\$HERE
    103 PREFIX=\${OJ_PREFIX:-\$DEFAULT_PREFIX}
    104 
    105 source \$PREFIX/ENV.bash             ## define JUNOTOP and JUNO_OPTICKS_PREFIX 
    106 source \$JUNOTOP/setup.sh            ## hookup many JUNOSW externals excluding Opticks
    107 source \$JUNO_OPTICKS_PREFIX/bashrc  ## hookup for Opticks and its externals
    108 
    109 source \$PREFIX/bin/opticks_juno.sh  ## oj bash functions 
    110 oj_geomsetup
    111 
    112 export CMAKE_PREFIX_PATH=\$PREFIX:\${CMAKE_PREFIX_PATH}
    113 export PATH=\$PREFIX/bin:\${PATH}
    114 export LD_LIBRARY_PATH=\$PREFIX/lib64:\${LD_LIBRARY_PATH}
    115 export PYTHONPATH=\$PREFIX/lib64:\${PYTHONPATH}
    116 export PYTHONPATH=\$PREFIX/python:\${PYTHONPATH}
    117 
    118 EOS


::

    P[blyth@localhost ~]$ cat /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-vLatest/bashrc
    #!/bin/bash
    #  
    #    D O   N O T   E D I T 
    #
    # generated by opticks-bashrc-generate-
    #
    # opticks-setup-hdr- Mon Mar 17 09:51:03 PM CST 2025

    NAME=$(basename $BASH_SOURCE)
    MSG="=== $NAME :" 



Change the bashrc and recreate test release dir::

   opticks-vi  ## change opticks-setup-paths-
   opticks-
   opticks-setup-generate
   okdist-;okdist-- 





G4CXTest_raindrop.sh : WORKS AFTER :  G4 + C4 SETUP + CVD + PYTHONPATH + adding CSG omission 
---------------------------------------------------------------------------------------------------


::

    p_release_env()
    {
        type $FUNCNAME

        : have to leave GPU choice to user
        export CUDA_VISIBLE_DEVICES=1   

        : G4 + C4 setup is included with OJ release : so leave manual 
        source /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/bin/geant4.sh
        source /home/blyth/junotop/ExternalLibs/custom4/0.1.8/bashrc

        : okdist-- use check release
        source /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bashrc 

        : external GEOM setup : getting it from the real OJ release 
        export GEOM=J25_3_0_Opticks_v0_3_5 
        export ${GEOM}_CFBaseFromGEOM=/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.3.0_Opticks-v0.3.5/el9_amd64_gcc11/2025_04_14/.opticks/GEOM/J25_3_0_Opticks_v0_3_5
    }


::

    P[blyth@localhost junosw]$ which G4CXTest_raindrop.sh 
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin/G4CXTest_raindrop.sh

    P[blyth@localhost junosw]$ G4CXTest_raindrop.sh run
    G4CXTest: error while loading shared libraries: libG4Tree.so: cannot open shared object file: No such file or directory
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin/G4CXTest_raindrop.sh : run error
    P[blyth@localhost junosw]$ 

Manual G4 setup::

    P[blyth@localhost junosw]$ source /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/bin/geant4.sh


    P[blyth@localhost junosw]$ G4CXTest_raindrop.sh run
    G4CXTest: error while loading shared libraries: libCustom4.so: cannot open shared object file: No such file or directory
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin/G4CXTest_raindrop.sh : run error


From jsw env find Custom4::

    P[blyth@localhost opticks_Debug]$ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep -i Custom4
    /home/blyth/junotop/ExternalLibs/custom4/0.1.8
    /home/blyth/junotop/ExternalLibs/custom4/0.1.8
    /home/blyth/junotop/ExternalLibs/custom4/0.1.8
    /home/blyth/junotop/ExternalLibs/custom4/0.1.8

Back in release env hookup Custom4::

    P[blyth@localhost junosw]$ source /home/blyth/junotop/ExternalLibs/custom4/0.1.8/bashrc

Now it runs with CVD warning and cx assert::

    P[blyth@localhost junosw]$ G4CXTest_raindrop.sh run
    2025-04-16 16:27:09.492 INFO  [262843] [G4CXApp::Create@338] U4Recorder::Switches
    WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:PMTSIM_STANDALONE
    NOT:PRODUCTION
    NOT:WITH_INSTRUMENTED_DEBUG




    **************************************************************
     Geant4 version Name: geant4-10-04-patch-02 [MT]   (25-May-2018)
                           Copyright : Geant4 Collaboration
    ...

    scontext::initConfig : MORE THAN ONE VISIBLE DEVICES - CHECK CUDA_VISIBLE_DEVICES envvar 
    G4CXApp::InitSensDet sdn PMTSDMgr sd YES
    
    ... 

    2025-04-16 16:27:21.912 INFO  [262843] [U4Recorder::PreUserTrackingAction_Optical@450]  modulo 100000 : ulabel.id 100000
    2025-04-16 16:27:23.209 INFO  [262843] [U4Recorder::PreUserTrackingAction_Optical@450]  modulo 100000 : ulabel.id 0
    G4CXTest: /home/blyth/opticks/g4cx/G4CXOpticks.cc:457: void G4CXOpticks::simulate(int, bool): Assertion `cx' failed.
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin/G4CXTest_raindrop.sh: line 183: 262843 Aborted                 (core dumped) $bin
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin/G4CXTest_raindrop.sh : run error
    P[blyth@localhost junosw]$ 


After pick the device "run" and "dbg" both complete::

    P[blyth@localhost ~]$ export CUDA_VISIBLE_DEVICES=1
    P[blyth@localhost ~]$ G4CXTest_raindrop.sh run

"ana" fails for lack of PYTHONPATH::

    P[blyth@localhost ~]$ G4CXTest_raindrop.sh ana
    Traceback (most recent call last):
      File "G4CXTest_raindrop.py", line 9, in <module>
        from opticks.ana.fold import Fold
    ModuleNotFoundError: No module named 'opticks'
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin/G4CXTest_raindrop.sh : ana error with script G4CXTest_raindrop.py
    P[blyth@localhost ~]$ which python
    ~/local/env/tools/conda/miniconda3/bin/python
    P[blyth@localhost ~]$



After setting PYTHONPATH find missing module opticks.CSG::

    P[blyth@localhost ~]$ export PYTHONPATH=/data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/py

    P[blyth@localhost ~]$ G4CXTest_raindrop.sh ana
    [from opticks.ana.p import * 
    [ana/p.py:from opticks.CSG.CSGFoundry import CSGFoundry 
    Traceback (most recent call last):
      File "G4CXTest_raindrop.py", line 10, in <module>
        from opticks.sysrap.sevt import SEvt, SAB
      File "/data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/py/opticks/sysrap/sevt.py", line 14, in <module>
        from opticks.ana.p import * 
      File "/data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/py/opticks/ana/p.py", line 191, in <module>
        from opticks.CSG.CSGFoundry import CSGFoundry 
    ModuleNotFoundError: No module named 'opticks.CSG'
    /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5/bin/G4CXTest_raindrop.sh : ana error with script G4CXTest_raindrop.py
    P[blyth@localhost ~]$ 


Following CSGFoundry.py inclusion "ana" and "pdb" are working.


"cf2" fails, must change to using PATH resolution for release running::

    .if [ "${arg/cf2}" != "$arg" ]; then
    -    $DIR/../../sysrap/tests/sseq_index_test.sh info_run_ana
    +    sseq_index_test.sh info_run_ana
         [ $? -ne 0 ] && echo $BASH_SOURCE : cf2 error && exit 5
     fi


Update the release script::

    gx
    om
    okdist-;okdist--


After that "cf2" succeeds. 




Testing actual release with git@github.com:simoncblyth/ok.git find viz attempt over ssh crashes gnome-shell
-----------------------------------------------------------------------------------------------------------------

::

    ok env
    cxr_min.sh 

Need to detect remote ssh usage and give an error before attempting to popup a window::

    P[blyth@localhost ~]$ opticks-f SSH_
    ./ana/fold.py:    has_SSH_CLIENT = not os.environ.get("SSH_CLIENT", None) is None 
    ./ana/fold.py:    has_SSH_TTY = not os.environ.get("SSH_TTY", None) is None 
    ./ana/fold.py:    return has_SSH_CLIENT or has_SSH_TTY
    ./bin/OpticksCTestRunner.sh:   [ -n "$SSH_CLIENT" -o -n "$SSH_TTY" ] && echo YES || echo NO
    ./bin/OpticksCTestRunner.sh:   SSH_CLIENT  \"$SSH_CLIENT\" 
    ./bin/OpticksCTestRunner.sh:   SSH_TTY     \"$SSH_TTY\"
    ./examples/UseOptiXGeometry/go.sh:if [ -n "$SSH_TTY" ]; then 
    ./examples/UseOptiXGeometryInstanced/go.sh:if [ -n "$SSH_TTY" ]; then 
    ./examples/UseOptiXGeometryInstancedOCtx/go.sh:if [ -n "\$SSH_TTY" ]; then 
    ./examples/UseOptiXGeometryOCtx/go.sh:if [ -n "$SSH_TTY" ]; then 
    ./examples/UseOptiXTextureLayeredOKImgGeo/go.sh:if [ -n "$SSH_TTY" ]; then 
    ./sysrap/SSys.cc:    char* ssh_client = getenv("SSH_CLIENT");
    ./sysrap/SSys.cc:    char* ssh_tty = getenv("SSH_TTY");
    P[blyth@localhost opticks]$ cd ~/ok


::

    357 bool SSys::IsRemoteSession()
    358 {
    359     char* ssh_client = getenv("SSH_CLIENT");
    360     char* ssh_tty = getenv("SSH_TTY");
    361 
    362     bool is_remote = ssh_client != NULL || ssh_tty != NULL ;
    363 
    364     LOG(verbose) << "SSys::IsRemoteSession"
    365                << " ssh_client " << ssh_client
    366                << " ssh_tty " << ssh_tty
    367                << " is_remote " << is_remote
    368                ;
    369 
    370     return is_remote ;
    371 }




OPTICKS_MAX_SLOT=M1 cxt_min.sh   ## CUDA error when try to simtrace over the slots
--------------------------------------------------------------------------------------

HMM: would have expected to get a cleaner error 
when try to simtrace more rays than slots, OR simtrace would 
do multiple launches. 


::

    (base) A[blyth@localhost ~]$ ok test4
                     arg : test4
                   regex : test([[:digit:]]{1,2})
                      m0 : test4
                      m1 : 4
                 cmdline : OPTICKS_MAX_SLOT=M1 cxt_min.sh
    [ cmdline - OPTICKS_MAX_SLOT=M1 cxt_min.sh
    -bash OK_LOGDIR /tmp/blyth/opticks/GEOM/J25_3_0_Opticks_v0_3_5/ok_sh
    /tmp/blyth/opticks/GEOM/J25_3_0_Opticks_v0_3_5/ok_sh
    OPTICKS_MAX_SLOT=M1 cxt_min.sh
    /data1/blyth/local/opticks_Debug/bin/cxt_min.sh - External GEOM setup detected
                     External_CFBaseFromGEOM : J25_3_0_Opticks_v0_3_5_CFBaseFromGEOM 
       J25_3_0_Opticks_v0_3_5_CFBaseFromGEOM : /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.3.0_Opticks-v0.3.5/el9_amd64_gcc11/2025_04_14/.opticks/GEOM/J25_3_0_Opticks_v0_3_5 
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/cxt_min.sh 
                  script : /data1/blyth/local/opticks_Debug/bin/cxt_min.py 
                     bin : CSGOptiXTMTest 
               which_bin : /data1/blyth/local/opticks_Debug/lib/CSGOptiXTMTest 
                  allarg : info_fold_run_dbg_brab_grab_ana 
                  defarg : run_info 
                     arg : run_info 
                    GEOM : J25_3_0_Opticks_v0_3_5 
    J25_3_0_Opticks_v0_3_5_CFBaseFromGEOM : /cvmfs/opticks.ihep.ac.cn/oj/releases/J25.3.0_Opticks-v0.3.5/el9_amd64_gcc11/2025_04_14/.opticks/GEOM/J25_3_0_Opticks_v0_3_5 
                    FOLD : /data1/blyth/tmp/GEOM/J25_3_0_Opticks_v0_3_5/CSGOptiXTMTest/0/A000 
                     MOI :  
                     LOG :  
                  LOGDIR : /data1/blyth/tmp/GEOM/J25_3_0_Opticks_v0_3_5/CSGOptiXTMTest/ 
                    BASE : /data1/blyth/tmp/GEOM/J25_3_0_Opticks_v0_3_5 
    CUDA_VISIBLE_DEVICES : 0 
                    CEGS : 16:0:9:2000 
    /data1/blyth/local/opticks_Debug/bin/cxt_min.sh : run/dbg : delete prior LOGNAME CSGOptiXTMTest.log
    2025-04-18 16:14:18.084 INFO  [723229] [SEventConfig::SetDevice@1333] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33796980736
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 262530128
    HeuristicMaxSlot(VRAM)/M         : 262
    HeuristicMaxSlot_Rounded(VRAM)   : 262000000
    MaxSlot/M                        : 1

    2025-04-18 16:14:18.084 INFO  [723229] [SEventConfig::SetDevice@1345]  Configured_MaxSlot/M 1 Final_MaxSlot/M 1 HeuristicMaxSlot_Rounded/M 262 changed NO  DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2025-04-18 16:14:18.168 INFO  [723229] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000 SEventConfig::MaxCurand 1000000000
    //CSGOptiX7.cu : simtrace idx 0 photon_idx 0  genstep_idx 0 evt->num_simtrace 1254000 
    //CSGOptiX7.cu : simtrace idx 0 pos.xyz -96000.000,  0.000,-54000.000 mom.xyz  -0.805,  0.000,  0.593  
    terminate called after throwing an instance of 'CUDA_Exception'
      what():  CUDA error on synchronize with error 'an illegal memory access was encountered' (/home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1064)

    /data1/blyth/local/opticks_Debug/bin/cxt_min.sh: line 178: 723229 Aborted                 (core dumped) $bin
    /data1/blyth/local/opticks_Debug/bin/cxt_min.sh run/dbg error
    /tmp/blyth/opticks/GEOM/J25_3_0_Opticks_v0_3_5/ok_sh
    -bash OK_LOGDIR /tmp/blyth/opticks/GEOM/J25_3_0_Opticks_v0_3_5/ok_sh
    total 0
    0 drwxr-xr-x. 2 blyth blyth  6 Apr 17 14:36 .
    0 drwxr-xr-x. 6 blyth blyth 87 Apr 17 14:36 ..
    /home/blyth
    ] cmdline - OPTICKS_MAX_SLOT=M1 cxt_min.sh
    (base) A[blyth@localhost ~]$ 



::

    OPTICKS_MAX_SLOT=M2 cxt_min.sh

