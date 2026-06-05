6k_server_opticks_build
=========================

Objective : test new Opticks on server without making cvmfs release
---------------------------------------------------------------------

Problem lxlogin lacks CUDA 13.1 so have to build on compute node


Build srun bash setup
-----------------------

::

    L[blyth@lxlogin004 oj]$ t oj6b
    oj6b () 
    { 
        : ~/oj/oj.bash;
        : building;
        oj6_env;
        srun --partition=junogpu --qos=junoatmgpu --gres=gpu:pro6000:1 --cpus-per-task=8 --mem=16G --pty bash
    }


Issue 1 : wrong nvcc from CUDA 12.4, not 13.1
------------------------------------------------

Had to "om-cleaninstall" to get rid of the cmake cache



Server opticks-t
-------------------

::


    SLOW: tests taking longer that 15.0 seconds
      32 /43  Test #32 : CSGTest.CSGMakerTest                                    Passed                         22.13  

    FAILS:  0   / 221   :  Fri Jun  5 10:55:48 2026  :  GEOM J26_1_1_Opticks_v0_6_3  


* TODO: Why CSGMakerTest so slow ?



Prefer to run in runtime env not the build env : so tmux directly for better env control
-------------------------------------------------------------------------------------------

::

    [blyth@lxlogin004 ~]$ t lx6t
    lx6t () 
    { 
        : ~/j/lxlogin.sh;
        : tmux building and testing;
        lx6_env;
        srun --partition=junogpu --qos=junoatmgpu --gres=gpu:pro6000:1 --cpus-per-task=8 --mem=16G --pty tmux new-session -A -s opticks_work
    }



Remember : ctrl-b then % to split vertically

::

     source /hpcfs/juno/junogpu/blyth/local/opticks_Debug/envset.sh


HMM no envset.sh standardly - but there is bashrc
----------------------------------------------------

::

    L[blyth@junogpu001 ~]$ source /hpcfs/juno/junogpu/blyth/local/opticks_Debug/envset.sh
    -bash: /hpcfs/juno/junogpu/blyth/local/opticks_Debug/envset.sh: No such file or directory
    L[blyth@junogpu001 ~]$ ls /hpcfs/juno/junogpu/blyth/local/opticks_Debug/
    bashrc  bin  build  cmake  externals  gl  include  lib  lib64  optix  py  tests
    L[blyth@junogpu001 ~]$ 

The bashrc is old, its not being generated standardly. Has to do  opticks-setup-generate
in the build tmux panel.




Now can run with server built opticks
----------------------------------------

buildtime tmux panel
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    lo
    oo    ## now added "opticks-setup-generate" to oo


runtime tmux panel
~~~~~~~~~~~~~~~~~~~~

::

    source /hpcfs/juno/junogpu/blyth/local/opticks_Debug/envset.sh
    cxs_min.sh report

lxlogin tab : for tasks not using GPU : like sreport running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    source /hpcfs/juno/junogpu/blyth/local/opticks_Debug/envset.sh
    cxs_min.sh report



