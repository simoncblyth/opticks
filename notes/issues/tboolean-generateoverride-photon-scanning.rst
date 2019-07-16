tboolean-generateoverride-photon-scanning
================================================


Revisit with aligned running, and now with ceiling of 100M
------------------------------------------------------------

::


    TBOOLEAN_TAG=1   ts box --generateoverride -1   --rngmax 3 
    TBOOLEAN_TAG=10  ts box --generateoverride -10  --rngmax 10 

    OpticksProfile=ERROR TBOOLEAN_TAG=100 ts box --generateoverride -100 --rngmax 100 --nog4propagate 



Things to vary only change GPU side::

    --compute 
    --rtx 0/1 
    --cvd 0   1   0,1
    --stack 
 


Issue : 100M CUDA illegal address : fixed by revivng production running  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    OpticksProfile=ERROR TBOOLEAN_TAG=100 ts box --generateoverride -100 --rngmax 100 --nog4propagate 


    OpticksProfile=ERROR TBOOLEAN_TAG=10  ts box --generateoverride -10  --rngmax 100 --nog4propagate        ##  0.417472

    ta box --tag 10    # had to permit some cmdline and rngmax differences between A and B : due to prior Geant4 -10 with smaller rngmax   


    OpticksProfile=ERROR TBOOLEAN_TAG=20  ts box --generateoverride -20  --rngmax 100 --nog4propagate        ##  0.797826 

    ta box --tag 20    # non-existing tagdir for g4, made ana/ab.py changes to still operate to some extent with missing B 


    OpticksProfile=ERROR TBOOLEAN_TAG=30  ts box --generateoverride -30  --rngmax 100 --nog4propagate



* :doc:`30M-interop-launch-CUDA-invalid-address`




Revisit now with production mode
-----------------------------------

::

    [blyth@localhost opticks]$ scan-ph-cmds
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 1 --compute --production --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 10 --compute --production --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 100 --compute --production --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 1000 --compute --production --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 10000 --compute --production --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 100000 --compute --production --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 1000000 --compute --production --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 10000000 --compute --production --nog4propagate --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 100000000 --compute --production --nog4propagate --cvd 1 --rtx 0
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 1 --compute --production --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 10 --compute --production --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 100 --compute --production --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 1000 --compute --production --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 10000 --compute --production --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 100000 --compute --production --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 1000000 --compute --production --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 10000000 --compute --production --nog4propagate --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1 --generateoverride 100000000 --compute --production --nog4propagate --cvd 1 --rtx 1
    [blyth@localhost opticks]$ 



Revisit tboolean arguments pfx and cat for easier organization
------------------------------------------------------------------

* cut thru some tboolean bash thickets by making OpticksCfg sensitive to TESTNAME envvar as a default for cat and pfx 

  * allows to remove cat and pfx options from tboolean-- so can use from higher level scanning 
  * succeeds to write into tagdir /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0/torch/1

::

    blyth@localhost optickscore]$ echo $OPTICKS_ANA_DEFAULTS
    det=g4live,cat=cvd_1_rtx_0,src=torch,tag=1,pfx=scan-ph
    [blyth@localhost optickscore]$ ip profile.py 
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    defaults det g4live cat cvd_1_rtx_0 src torch tag 1 pfx scan-ph 
    [2019-07-16 13:34:27,792] p114672 {__init__            :profile.py:21} INFO     -  tagdir:/home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0/torch/1 name:ab.pro.ap tag:1 g4:False 
    [2019-07-16 13:34:27,796] p114672 {__init__            :profile.py:21} INFO     -  tagdir:/home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0/torch/-1 name:ab.pro.bp tag:-1 g4:True 
    ab.pro



Multievent
------------

::

    OpticksProfile=ERROR ts box --pfx scan-ph --cat cvd_1_rtx_0 --generateoverride 1 --compute --production --cvd 1 --rtx 0 --multievent 2 -D


* :doc:`revive_multievent_for_profiling_overheads`

Opticks photon scanning performance begs the question : what are the overheads ?



WOW : TITAN RTX with RTX mode ON (R1) : giving extra 7x  : exceeding 10,000x speedup for 3M photons : this is an exceedingly simple geometry though
--------------------------------------------------------------------------------------------------------------------------------------------------------


* this is not changing tag, just defaulting to 1, it just overwrites the arrays 

::

    [blyth@localhost issues]$ scan-cmds
    tboolean.sh box --generateoverride 1 --error --cvd 1 --rtx 1
    tboolean.sh box --generateoverride 1000 --error --cvd 1 --rtx 1
    tboolean.sh box --generateoverride 10000 --error --cvd 1 --rtx 1
    tboolean.sh box --generateoverride 100000 --error --cvd 1 --rtx 1
    tboolean.sh box --generateoverride 200000 --error --cvd 1 --rtx 1
    tboolean.sh box --generateoverride 500000 --error --cvd 1 --rtx 1
    tboolean.sh box --generateoverride 1000000 --error --cvd 1 --rtx 1
    tboolean.sh box --generateoverride 2000000 --error --cvd 1 --rtx 1
    tboolean.sh box --generateoverride 3000000 --error --cvd 1 --rtx 1


::

    [blyth@localhost issues]$ scan-post
    scan.py /tmp/tboolean-box
    dirs : 38  dtimes : 19 
      ok1 : OpticksEvent_launch.launch001 
      ok2 : DeltaTime.OPropagator::launch_0 
      ok3 : OpticksEvent_prelaunch.prelaunch000 
      ok4 : DeltaTime.OpSeeder::seedPhotonsFromGenstepsViaOptiX_0 
       g4 : DeltaTime.CG4::propagate_0 
     20190604_183232   tag0:-1  tag1:1    n:1            ok1:    0.0056  ok2:    0.0039  g4:    1.3398   g4/ok1:     240.0  g4/ok2:     343.0   ok3:    1.7323 ok4:    0.0156       
     20190604_183240   tag0:-1  tag1:1    n:1000         ok1:    0.0056  ok2:    0.0078  g4:    1.4219   g4/ok1:     254.8  g4/ok2:     182.0   ok3:    1.7358 ok4:    0.0156       
     20190604_183248   tag0:-1  tag1:1    n:10000        ok1:    0.0060  ok2:    0.0078  g4:    2.2852   g4/ok1:     377.8  g4/ok2:     292.5   ok3:    1.7219 ok4:    0.0156       
     20190604_183257   tag0:-1  tag1:1    n:100000       ok1:    0.0108  ok2:    0.0117  g4:   10.5547   g4/ok1:     975.7  g4/ok2:     900.7   ok3:    1.7546 ok4:    0.0156       
     20190604_183316   tag0:-1  tag1:1    n:200000       ok1:    0.0184  ok2:    0.0195  g4:   19.7266   g4/ok1:    1073.4  g4/ok2:    1010.0   ok3:    1.7761 ok4:    0.0156       
     20190604_200306   tag0:-1  tag1:1    n:500000       ok1:    0.0412  ok2:    0.0430  g4:   46.7695   g4/ok1:    1135.9  g4/ok2:    1088.5   ok3:    1.8456 ok4:    0.0430       
     20190604_200407   tag0:-1  tag1:1    n:1000000      ok1:    0.0795  ok2:    0.0781  g4:   92.6797   g4/ok1:    1165.4  g4/ok2:    1186.3   ok3:    1.9338 ok4:    0.0234       
     20190604_201355   tag0:-1  tag1:1    n:2000000      ok1:    0.1565  ok2:    0.1562  g4:  187.8633   g4/ok1:    1200.3  g4/ok2:    1202.3   ok3:    2.1452 ok4:    0.0117       
     20190604_201742   tag0:-1  tag1:1    n:3000000      ok1:    0.2307  ok2:    0.2305  g4:  280.1250   g4/ok1:    1214.1  g4/ok2:    1215.5   ok3:    2.4038 ok4:    0.0312       
     ------------- above with RTX off and both GPUS ---- below with RTX ON and just TITAN RTX -------------------------------------------------------------------------------
     20190604_213817   tag0:-1  tag1:1    n:1            ok1:    0.0027  ok2:    0.0000  g4:    1.3477   g4/ok1:     493.5  g4/ok2:       0.0   ok3:    1.1203 ok4:    0.0156       
     20190604_213824   tag0:-1  tag1:1    n:1000         ok1:    0.0028  ok2:    0.0039  g4:    1.4102   g4/ok1:     505.1  g4/ok2:     361.0   ok3:    1.1080 ok4:    0.0156       
     20190604_213831   tag0:-1  tag1:1    n:10000        ok1:    0.0028  ok2:    0.0039  g4:    2.2109   g4/ok1:     793.3  g4/ok2:     566.0   ok3:    1.2067 ok4:    0.0312       
     20190604_213839   tag0:-1  tag1:1    n:100000       ok1:    0.0027  ok2:    0.0039  g4:   10.4961   g4/ok1:    3957.8  g4/ok2:    2687.0   ok3:    1.1292 ok4:    0.0117       
     20190604_213856   tag0:-1  tag1:1    n:200000       ok1:    0.0035  ok2:    0.0039  g4:   19.4219   g4/ok1:    5542.8  g4/ok2:    4972.0   ok3:    1.2208 ok4:    0.0273       
     20190604_213923   tag0:-1  tag1:1    n:500000       ok1:    0.0064  ok2:    0.0039  g4:   46.8047   g4/ok1:    7344.2  g4/ok2:   11982.0   ok3:    1.0817 ok4:    0.0312       
     20190604_214022   tag0:-1  tag1:1    n:1000000      ok1:    0.0107  ok2:    0.0117  g4:   90.5586   g4/ok1:    8477.7  g4/ok2:    7727.7   ok3:    1.0916 ok4:    0.0117       
     20190604_214211   tag0:-1  tag1:1    n:2000000      ok1:    0.0231  ok2:    0.0234  g4:  181.6055   g4/ok1:    7851.9  g4/ok2:    7748.5   ok3:    1.1282 ok4:    0.0156       
     20190604_214545   tag0:-1  tag1:1    n:3000000      ok1:    0.0304  ok2:    0.0273  g4:  273.9727   g4/ok1:    9025.9  g4/ok2:   10019.6   ok3:    1.1570 ok4:    0.0117       
    [blyth@localhost issues]$                                           


* timings unchanged for < 0.5M photons



RTX OFF : TITAN V and TITAN RTX
---------------------------------------

* RTX mode was OFF, and CVD was unset : so both TITAN V and TITAN RTX in use

::

    [blyth@localhost ~]$ scan-;scan-cmds
    tboolean.sh box --generateoverride 1 --error
    tboolean.sh box --generateoverride 1000 --error
    tboolean.sh box --generateoverride 10000 --error
    tboolean.sh box --generateoverride 100000 --error
    tboolean.sh box --generateoverride 200000 --error
    tboolean.sh box --generateoverride 500000 --error
    tboolean.sh box --generateoverride 1000000 --error
    tboolean.sh box --generateoverride 2000000 --error
    tboolean.sh box --generateoverride 3000000 --error


::

    [blyth@localhost opticks]$ scan-post
    scan.py /tmp/tboolean-box
    dirs : 18  dtimes : 9 
      ok1 : OpticksEvent_launch.launch001 
      ok2 : DeltaTime.OPropagator::launch_0 
      ok3 : OpticksEvent_prelaunch.prelaunch000 
      ok4 : DeltaTime.OpSeeder::seedPhotonsFromGenstepsViaOptiX_0 
       g4 : DeltaTime.CG4::propagate_0 
     20190604_183232   tag0:-1  tag1:1    n:1            ok1:    0.0056  ok2:    0.0039  g4:    1.3398   g4/ok1:     240.0  g4/ok2:     343.0   ok3:    1.7323 ok4:    0.0156       
     20190604_183240   tag0:-1  tag1:1    n:1000         ok1:    0.0056  ok2:    0.0078  g4:    1.4219   g4/ok1:     254.8  g4/ok2:     182.0   ok3:    1.7358 ok4:    0.0156       
     20190604_183248   tag0:-1  tag1:1    n:10000        ok1:    0.0060  ok2:    0.0078  g4:    2.2852   g4/ok1:     377.8  g4/ok2:     292.5   ok3:    1.7219 ok4:    0.0156       
     20190604_183257   tag0:-1  tag1:1    n:100000       ok1:    0.0108  ok2:    0.0117  g4:   10.5547   g4/ok1:     975.7  g4/ok2:     900.7   ok3:    1.7546 ok4:    0.0156       
     20190604_183316   tag0:-1  tag1:1    n:200000       ok1:    0.0184  ok2:    0.0195  g4:   19.7266   g4/ok1:    1073.4  g4/ok2:    1010.0   ok3:    1.7761 ok4:    0.0156       
     20190604_200306   tag0:-1  tag1:1    n:500000       ok1:    0.0412  ok2:    0.0430  g4:   46.7695   g4/ok1:    1135.9  g4/ok2:    1088.5   ok3:    1.8456 ok4:    0.0430       
     20190604_200407   tag0:-1  tag1:1    n:1000000      ok1:    0.0795  ok2:    0.0781  g4:   92.6797   g4/ok1:    1165.4  g4/ok2:    1186.3   ok3:    1.9338 ok4:    0.0234       
     20190604_201355   tag0:-1  tag1:1    n:2000000      ok1:    0.1565  ok2:    0.1562  g4:  187.8633   g4/ok1:    1200.3  g4/ok2:    1202.3   ok3:    2.1452 ok4:    0.0117       
     20190604_201742   tag0:-1  tag1:1    n:3000000      ok1:    0.2307  ok2:    0.2305  g4:  280.1250   g4/ok1:    1214.1  g4/ok2:    1215.5   ok3:    2.4038 ok4:    0.0312       
    [blyth@localhost opticks]$ 



* almost to 1000x at around 100k photons without RTX (using both TITAN V and TITAN RTX)

::

    tboolean.sh box --generateoverride 100000 --error --cvd 1 --rtx 1 




During running, noted very different memory usage reported by nvidia-smi, almost twice used on TITAN V::

    [blyth@localhost opticks]$ nvidia-smi
    Tue Jun  4 20:20:22 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0  On |                  N/A |
    | 41%   33C    P8    20W / 280W |    661MiB / 24189MiB |      2%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:A6:00.0 Off |                  N/A |
    | 33%   47C    P8    28W / 250W |    317MiB / 12036MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0    129223      C   /home/blyth/local/opticks/lib/OKG4Test       161MiB |
    |    0    255296      G   /usr/bin/X                                   355MiB |
    |    0    256000      G   /usr/bin/gnome-shell                         132MiB |
    |    1    129223      C   /home/blyth/local/opticks/lib/OKG4Test       305MiB |
    +-----------------------------------------------------------------------------+



Hmm some deviations with scatters. But this is a non-aligned comparison, so scatters should be excluded ?::

    [blyth@localhost ana]$ tboolean.py
    args: /home/blyth/opticks/ana/tboolean.py
    [2019-06-04 21:02:54,687] p202496 {/home/blyth/opticks/ana/env.py:143} WARNING - legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init
    [2019-06-04 21:02:54,688] p202496 {/home/blyth/opticks/ana/tboolean.py:66} INFO - pfx tboolean-box tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    [2019-06-04 21:02:54,688] p202496 {/home/blyth/opticks/ana/ab.py:109} INFO - ab START
    [2019-06-04 21:02:54,689] p202496 {/home/blyth/opticks/ana/evt.py:317} INFO - loaded metadata from /tmp/tboolean-box/evt/tboolean-box/torch/1 
    [2019-06-04 21:02:54,689] p202496 {/home/blyth/opticks/ana/evt.py:318} INFO - metadata                   /tmp/tboolean-box/evt/tboolean-box/torch/1 7eacac80dd923603e57d550d0e482e00 2e8d01898525028639a5bd74dca33805 3000000     0.2307 COMPUTE_MODE  
    [2019-06-04 21:02:54,691] p202496 {/home/blyth/opticks/ana/evt.py:257} INFO - testcsgpath tboolean-box 
    [2019-06-04 21:02:54,692] p202496 {/home/blyth/opticks/ana/evt.py:267} INFO - reldir /tmp/tboolean-box/GItemList 
    [2019-06-04 21:02:54,692] p202496 {/home/blyth/opticks/ana/base.py:236} INFO - txt GMaterialLib reldir  /tmp/tboolean-box/GItemList 
    [2019-06-04 21:02:57,957] p202496 {/home/blyth/opticks/ana/evt.py:317} INFO - loaded metadata from /tmp/tboolean-box/evt/tboolean-box/torch/-1 
    [2019-06-04 21:02:57,959] p202496 {/home/blyth/opticks/ana/evt.py:318} INFO - metadata                  /tmp/tboolean-box/evt/tboolean-box/torch/-1 dfab648a405a7b4aa4205d321e855289 5bb3a14ad1f7060f0497d7dda57221ca 3000000    -1.0000 COMPUTE_MODE  
    [2019-06-04 21:02:57,962] p202496 {/home/blyth/opticks/ana/evt.py:257} INFO - testcsgpath tboolean-box 
    [2019-06-04 21:02:57,962] p202496 {/home/blyth/opticks/ana/evt.py:267} INFO - reldir /tmp/tboolean-box/GItemList 
    [2019-06-04 21:02:57,962] p202496 {/home/blyth/opticks/ana/base.py:236} INFO - txt GMaterialLib reldir  /tmp/tboolean-box/GItemList 
    [2019-06-04 21:03:01,441] p202496 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 32.878057233426475 ndf 27 c2p 1.2177058234602398 c2_pval 0.2011239991588083 
    [2019-06-04 21:03:01,445] p202496 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 28.515196736139934 ndf 20 c2p 1.4257598368069968 c2_pval 0.09775350119603299 
    ab.a.metadata:                  /tmp/tboolean-box/evt/tboolean-box/torch/1 7eacac80dd923603e57d550d0e482e00 2e8d01898525028639a5bd74dca33805 3000000     0.2307 COMPUTE_MODE 
    [2019-06-04 21:03:01,456] p202496 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 32.878057233426475 ndf 27 c2p 1.2177058234602398 c2_pval 0.2011239991588083 
    [2019-06-04 21:03:01,460] p202496 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 28.515196736139934 ndf 20 c2p 1.4257598368069968 c2_pval 0.09775350119603299 
    [2019-06-04 21:03:01,462] p202496 {/home/blyth/opticks/ana/seq.py:284} INFO -  c2sum 13.74372100648584 ndf 10 c2p 1.374372100648584 c2_pval 0.18500547799540035 
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/tboolean-box/torch/  1 :  20190604-2022 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-box/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/tboolean-box/torch/ -1 :  20190604-2022 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-box/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    tboolean-box
    .                seqhis_ana  1:tboolean-box:tboolean-box   -1:tboolean-box:tboolean-box        c2        ab        ba 
    .                            3000000   3000000        32.88/27 =  1.22  (pval:0.201 prob:0.799)  
    0000             8ccd   2638631   2638737             0.00        1.000 +- 0.001        1.000 +- 0.001  [4 ] TO BT BT SA
    0001              8bd    185063    184644             0.47        1.002 +- 0.002        0.998 +- 0.002  [3 ] TO BR SA
    0002            8cbcd    162067    162486             0.54        0.997 +- 0.002        1.003 +- 0.002  [5 ] TO BT BR BT SA
    0003           8cbbcd      9985     10096             0.61        0.989 +- 0.010        1.011 +- 0.010  [6 ] TO BT BR BR BT SA
    0004              86d       847       802             1.23        1.056 +- 0.036        0.947 +- 0.033  [3 ] TO SC SA
    0005            86ccd       736       740             0.01        0.995 +- 0.037        1.005 +- 0.037  [5 ] TO BT BT SC SA
    0006          8cbbbcd       625       585             1.32        1.068 +- 0.043        0.936 +- 0.039  [7 ] TO BT BR BR BR BT SA
    0007              4cd       563       540             0.48        1.043 +- 0.044        0.959 +- 0.041  [3 ] TO BT AB
    0008            8c6cd       269       261             0.12        1.031 +- 0.063        0.970 +- 0.060  [5 ] TO BT SC BT SA
    0009       bbbbbbb6cd       255       191             9.18        1.335 +- 0.084        0.749 +- 0.054  [10] TO BT SC BR BR BR BR BR BR BR
    0010            8cc6d       118       100             1.49        1.180 +- 0.109        0.847 +- 0.085  [5 ] TO SC BT BT SA
    0011          8cc6ccd       105        92             0.86        1.141 +- 0.111        0.876 +- 0.091  [7 ] TO BT BT SC BT BT SA
    0012               4d       100        93             0.25        1.075 +- 0.108        0.930 +- 0.096  [2 ] TO AB
    0013           8cbc6d        69        66             0.07        1.045 +- 0.126        0.957 +- 0.118  [6 ] TO SC BT BR BT SA
    0014             4ccd        65        93             4.96        0.699 +- 0.087        1.431 +- 0.148  [4 ] TO BT BT AB
    0015           8cb6cd        58        52             0.33        1.115 +- 0.146        0.897 +- 0.124  [6 ] TO BT SC BR BT SA
    0016             86bd        55        40             2.37        1.375 +- 0.185        0.727 +- 0.115  [4 ] TO BR SC SA
    0017         8cbc6ccd        53        70             2.35        0.757 +- 0.104        1.321 +- 0.158  [8 ] TO BT BT SC BT BR BT SA
    0018           86cbcd        50        50             0.00        1.000 +- 0.141        1.000 +- 0.141  [6 ] TO BT BR BT SC SA
    0019             4bcd        33        33             0.00        1.000 +- 0.174        1.000 +- 0.174  [4 ] TO BT BR AB
    .                            3000000   3000000        32.88/27 =  1.22  (pval:0.201 prob:0.799)  
    .                pflags_ana  1:tboolean-box:tboolean-box   -1:tboolean-box:tboolean-box        c2        ab        ba 
    .                            3000000   3000000        13.74/10 =  1.37  (pval:0.185 prob:0.815)  
    0000             1880   2638631   2638737             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] TO|BT|SA
    0001             1480    185063    184644             0.47        1.002 +- 0.002        0.998 +- 0.002  [3 ] TO|BR|SA
    0002             1c80    172706    173203             0.71        0.997 +- 0.002        1.003 +- 0.002  [4 ] TO|BT|BR|SA
    0003             18a0      1229      1193             0.54        1.030 +- 0.029        0.971 +- 0.028  [4 ] TO|BT|SA|SC
    0004             10a0       847       802             1.23        1.056 +- 0.036        0.947 +- 0.033  [3 ] TO|SA|SC
    0005             1808       628       633             0.02        0.992 +- 0.040        1.008 +- 0.040  [3 ] TO|BT|AB
    0006             1ca0       396       374             0.63        1.059 +- 0.053        0.944 +- 0.049  [5 ] TO|BT|BR|SA|SC
    0007             1c20       278       213             8.60        1.305 +- 0.078        0.766 +- 0.052  [4 ] TO|BT|BR|SC
    0008             1008       100        93             0.25        1.075 +- 0.108        0.930 +- 0.096  [2 ] TO|AB
    0009             14a0        75        62             1.23        1.210 +- 0.140        0.827 +- 0.105  [4 ] TO|BR|SA|SC
    0010             1c08        42        40             0.05        1.050 +- 0.162        0.952 +- 0.151  [4 ] TO|BT|BR|AB
    0011             1408         5         6             0.00        0.833 +- 0.373        1.200 +- 0.490  [3 ] TO|BR|AB
    .                            3000000   3000000        13.74/10 =  1.37  (pval:0.185 prob:0.815)  
    .                seqmat_ana  1:tboolean-box:tboolean-box   -1:tboolean-box:tboolean-box        c2        ab        ba 
    .                            3000000   3000000        28.52/20 =  1.43  (pval:0.098 prob:0.902)  
    0000             1232   2638631   2638737             0.00        1.000 +- 0.001        1.000 +- 0.001  [4 ] Vm G2 Vm Rk
    0001              122    185910    185446             0.58        1.003 +- 0.002        0.998 +- 0.002  [3 ] Vm Vm Rk
    0002            12332    162336    162747             0.52        0.997 +- 0.002        1.003 +- 0.002  [5 ] Vm G2 G2 Vm Rk
    0003           123332     10065     10164             0.48        0.990 +- 0.010        1.010 +- 0.010  [6 ] Vm G2 G2 G2 Vm Rk
    0004            12232       736       740             0.01        0.995 +- 0.037        1.005 +- 0.037  [5 ] Vm G2 Vm Vm Rk
    0005          1233332       646       600             1.70        1.077 +- 0.042        0.929 +- 0.038  [7 ] Vm G2 G2 G2 G2 Vm Rk
    0006              332       563       540             0.48        1.043 +- 0.044        0.959 +- 0.041  [3 ] Vm G2 G2
    0007       3333333332       273       209             8.50        1.306 +- 0.079        0.766 +- 0.053  [10] Vm G2 G2 G2 G2 G2 G2 G2 G2 G2
    0008            12322       118       100             1.49        1.180 +- 0.109        0.847 +- 0.085  [5 ] Vm Vm G2 Vm Rk
    0009          1232232       105        92             0.86        1.141 +- 0.111        0.876 +- 0.091  [7 ] Vm G2 Vm Vm G2 Vm Rk
    0010               22       100        93             0.25        1.075 +- 0.108        0.930 +- 0.096  [2 ] Vm Vm
    0011             1222        75        60             1.67        1.250 +- 0.144        0.800 +- 0.103  [4 ] Vm Vm Vm Rk
    0012           123322        69        66             0.07        1.045 +- 0.126        0.957 +- 0.118  [6 ] Vm Vm G2 G2 Vm Rk
    0013             2232        65        93             4.96        0.699 +- 0.087        1.431 +- 0.148  [4 ] Vm G2 Vm Vm
    0014         12332232        53        70             2.35        0.757 +- 0.104        1.321 +- 0.158  [8 ] Vm G2 Vm Vm G2 G2 Vm Rk
    0015           122332        50        50             0.00        1.000 +- 0.141        1.000 +- 0.141  [6 ] Vm G2 G2 Vm Vm Rk
    0016         12333332        34        41             0.65        0.829 +- 0.142        1.206 +- 0.188  [8 ] Vm G2 G2 G2 G2 G2 Vm Rk
    0017             3332        33        33             0.00        1.000 +- 0.174        1.000 +- 0.174  [4 ] Vm G2 G2 G2
    0018          1233322        23        16             1.26        1.438 +- 0.300        0.696 +- 0.174  [7 ] Vm Vm G2 G2 G2 Vm Rk
    0019        123332232        20        15             0.71        1.333 +- 0.298        0.750 +- 0.194  [9 ] Vm G2 Vm Vm G2 G2 G2 Vm Rk
    .                            3000000   3000000        28.52/20 =  1.43  (pval:0.098 prob:0.902)  
    ab.a.metadata:                  /tmp/tboolean-box/evt/tboolean-box/torch/1 7eacac80dd923603e57d550d0e482e00 2e8d01898525028639a5bd74dca33805 3000000     0.2307 COMPUTE_MODE 
    ab.a.metadata.csgmeta0:[]
    rpost_dv maxdvmax:558.13779107 maxdv:[0.013763847773677895, 0.013763847773674343, 0.0, 0.0, 558.137791070284, 20.09521774956511] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA : 2638631  2638737  :   2320538 37128608/    788: 0.000  mx/mn/av 0.01376/     0/2.921e-07  eps:0.0002    
     0001            :                       TO BR SA :  185063   184644  :     11234  134808/      6: 0.000  mx/mn/av 0.01376/     0/6.126e-07  eps:0.0002    
     0002            :                 TO BT BR BT SA :  162067   162486  :      8610  172200/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :    9985    10096  :        23     552/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0005            :                 TO BT BT SC SA :     736      740  :         1      20/      5: 0.250  mx/mn/av  558.1/     0/ 32.49  eps:0.0002    
     0007            :                       TO BT AB :     563      540  :         2      24/      4: 0.167  mx/mn/av   20.1/     0/ 1.399  eps:0.0002    
    rpol_dv maxdvmax:1.19685029984 maxdv:[0.0, 0.0, 0.0, 0.0, 1.196850299835205, 0.0] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA : 2638631  2638737  :   2320538 27846456/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                       TO BR SA :  185063   184644  :     11234  101106/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :  162067   162486  :      8610  129150/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :    9985    10096  :        23     414/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0005            :                 TO BT BT SC SA :     736      740  :         1      15/      6: 0.400  mx/mn/av  1.197/     0/0.2446  eps:0.0002    
     0007            :                       TO BT AB :     563      540  :         2      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    ox_dv maxdvmax:558.138122559 maxdv:[2.384185791015625e-07, 0.0, 4.76837158203125e-07, 4.76837158203125e-07, 558.1381225585938, 20.08863639831543] 
      idx        msg :                            sel :    lcu1     lcu2  :     nitem   nelem/  ndisc: fdisc  mx/mn/av     mx/    mn/   avg  eps:eps    
     0000            :                    TO BT BT SA : 2638631  2638737  :   2320538 27846456/      0: 0.000  mx/mn/av 2.384e-07/     0/2.484e-08  eps:0.0002    
     0001            :                       TO BR SA :  185063   184644  :     11234  134808/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :  162067   162486  :      8610  103320/      0: 0.000  mx/mn/av 4.768e-07/     0/4.47e-08  eps:0.0002    
     0003            :              TO BT BR BR BT SA :    9985    10096  :        23     276/      0: 0.000  mx/mn/av 4.768e-07/     0/4.47e-08  eps:0.0002    
     0005            :                 TO BT BT SC SA :     736      740  :         1      12/      9: 0.750  mx/mn/av  558.1/     0/ 52.33  eps:0.0002    
     0007            :                       TO BT AB :     563      540  :         2      24/      4: 0.167  mx/mn/av  20.09/     0/ 1.398  eps:0.0002    
    c2p : {'seqmat_ana': 1.4257598368069968, 'pflags_ana': 1.374372100648584, 'seqhis_ana': 1.2177058234602398} c2pmax: 1.4257598368069968  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 1.196850299835205, 'rpost_dv': 558.137791070284} rmxs_max_: 558.13779107  CUT ok.rdvmax 0.1  RC:88 
    pmxs_ : {'ox_dv': 558.1381225585938} pmxs_max_: 558.138122559  CUT ok.pdvmax 0.001  RC:99 
    [2019-06-04 21:03:19,899] p202496 {/home/blyth/opticks/ana/tboolean.py:74} INFO - early exit as non-interactive
    [blyth@localhost ana]$ 



The skips were not being applied, get rid of deviants after fix that::

    ab.a.metadata:                  /tmp/tboolean-box/evt/tboolean-box/torch/1 7eacac80dd923603e57d550d0e482e00 2e8d01898525028639a5bd74dca33805 3000000     0.2307 COMPUTE_MODE 
    ab.a.metadata.csgmeta0:[]
    rpost_dv maxdvmax: 0.01376 maxdv: 0.01376  0.01376        0        0  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA : 2638631  2638737  :     2320538  37128608/      788: 0.000  mx/mn/av   0.01376/        0/2.921e-07  eps:0.0002    
     0001            :                       TO BR SA :  185063   184644  :       11234    134808/        6: 0.000  mx/mn/av   0.01376/        0/6.126e-07  eps:0.0002    
     0002            :                 TO BT BR BT SA :  162067   162486  :        8610    172200/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :    9985    10096  :          23       552/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
    rpol_dv maxdvmax:       0 maxdv:       0        0        0        0  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA : 2638631  2638737  :     2320538  27846456/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0001            :                       TO BR SA :  185063   184644  :       11234    101106/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :  162067   162486  :        8610    129150/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :    9985    10096  :          23       414/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
    ox_dv maxdvmax:4.768e-07 maxdv:2.384e-07        0 4.768e-07 4.768e-07  skip:SC AB RE
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem/    ndisc: fdisc  mx/mn/av        mx/       mn/      avg  eps:eps    
     0000            :                    TO BT BT SA : 2638631  2638737  :     2320538  27846456/        0: 0.000  mx/mn/av 2.384e-07/        0/2.484e-08  eps:0.0002    
     0001            :                       TO BR SA :  185063   184644  :       11234    134808/        0: 0.000  mx/mn/av         0/        0/        0  eps:0.0002    
     0002            :                 TO BT BR BT SA :  162067   162486  :        8610    103320/        0: 0.000  mx/mn/av 4.768e-07/        0/ 4.47e-08  eps:0.0002    
     0003            :              TO BT BR BR BT SA :    9985    10096  :          23       276/        0: 0.000  mx/mn/av 4.768e-07/        0/ 4.47e-08  eps:0.0002    
    c2p : {'seqmat_ana': 1.4257598368069968, 'pflags_ana': 1.374372100648584, 'seqhis_ana': 1.2177058234602398} c2pmax: 1.4257598368069968  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 4.76837158203125e-07} pmxs_max_: 4.76837158203e-07  CUT ok.pdvmax 0.001  RC:0 
    [2019-06-04 21:26:38,869] p241135 {/home/blyth/opticks/ana/tboolean.py:71} INFO - early exit as non-interactive



