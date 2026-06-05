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

    lo    ## aka local_ok_build
    oo    ## now added "opticks-setup-generate" to oo


runtime tmux panel
~~~~~~~~~~~~~~~~~~~~

::

    source /hpcfs/juno/junogpu/blyth/local/opticks_Debug/envset.sh
    lu     ## alternative to above line, doing local_ok_usage

    cxs_min.sh report


lxlogin tab : for tasks not using GPU : like sreport running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    source /hpcfs/juno/junogpu/blyth/local/opticks_Debug/envset.sh
    lu     ## alternative to above line, doing local_ok_usage

    cxs_min.sh report







RTX 6000 PRO BLACKWELL, MAX_BOUNCE 63
------------------------------------------

::

    lu ; cxs_min.sh
    ...

    -[NPFold::compare_subarray.a
    [NP::descTable_ (12, 13, )
                        SbOE0    SbOE1    SeOE0     tBOE     tsG3     tsG4     tsG5     tsG6     tsG7     tsG8     tPrL     tPoL     tEOE
              //A000        0      166   293567       23    23821    24125    24147    24188    95090    99706    99751   253036   293585
              //A001        0       94   167533       20      181      181      202      219      622      629      650   151546   167549
              //A002        0       92  1584374       19      177      178      213      234      680      688      708  1431895  1584392
              //A003        0      102  3152329       20      193      194      245      275      768      776      798  2854659  3152347
              //A004        0      124  4610677       20      216      217      273      313      876      887      908  4157756  4610695
              //A005        0      105  6072248       20      197      197      251      291      911      921      942  5491226  6072265
              //A006        0      107  7576709       19      198      199      251      281      955      962      984  6849024  7576727
              //A007        0      109  9088696       20      199      200      251      282      986      994     1016  8213296  9088713
              //A008        0       98 10609345       19      189      190      241      271     1025     1033     1053  9583452 10609363
              //A009        0       98 12121786       20      188      189      239      269     1080     1087     1108 10952034 12121803
              //A010        0      103 13644084       20      198      198      262      307     1239     1248     1269 12314946 13644102
              //A011        0      110 15152814       20      200      201      262      301     1272     1280     1305 13686536 15152831
    num_timestamp 0 auto-offset from t0 0
              TOTAL:        0     1308 84074162      240    25957    26269    26837    27231   105504   110211   110492 75939406 84074372

       SbOE0 : SEvt__beginOfEvent_0
       SbOE1 : SEvt__beginOfEvent_1
       SeOE0 : SEvt__endOfEvent_0
        tBOE : t_BeginOfEvent
        tsG3 : t_setGenstep_3
        tsG4 : t_setGenstep_4
        tsG5 : t_setGenstep_5
        tsG6 : t_setGenstep_6
        tsG7 : t_setGenstep_7
        tsG8 : t_setGenstep_8
        tPrL : t_PreLaunch
        tPoL : t_PostLaunch
        tEOE : t_EndOfEvent
    ]NP::descTable_ (12, 13, )

    NPFold::save("$SREPORT_FOLD")
     resolved to  [/hpcfs/juno/junogpu/blyth/tmp/GEOM/J26_1_1_Opticks_v0_6_3/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan_sreport]
    ]sreport.main : CREATED REPORT 




RTX 5000 ADA, MAX_BOUNCE 63
-----------------------------

::

    lu ; cxs_min.sh
    ...

    [NP::descTable_ (12, 13, )
                        SbOE0    SbOE1    SeOE0     tBOE     tsG3     tsG4     tsG5     tsG6     tsG7     tsG8     tPrL     tPoL     tEOE
              //A000        0       99   269292       11      153      305      318      333     1439    13848    13875   254604   269298
              //A001        0       47   246443        6       81       82       98      108      902      908      915   228768   246449
              //A002        0       46  2361817        6       81       81       97      106     1039     1042     1050  2211942  2361822
              //A003        0      146  4751794       44      190      190      209      220     1375     1379     1388  4436080  4751800
              //A004        0       55  7137962        7       89       89      109      122     1442     1446     1454  6680821  7137969
              //A005        0       57  9567061        7       96       97      119      136     1962     1967     1977  8974559  9567068
              //A006        0       59 11982665        8      189      189      218      238     1904     1910     1923 11285279 11982672
              //A007        0       59 14520409        8      101      101      130      150     1977     1982     1992 13641009 14520415
              //A008        0       65 16995037        9      107      108      134      152     2189     2207     2244 16046454 16995044
              //A009        0      141 19637298       44      199      200      231      251     2428     2434     2447 18473146 19637304
              //A010        0       64 22147099        9      109      109      141      161     2497     2502     2513 20868157 22147106
              //A011        0       61 24555825        8      114      114      150      172     2725     2732     2749 23128661 24555833
    num_timestamp 0 auto-offset from t0 0
              TOTAL:        0      899 134172702      167     1509     1665     1954     2149    21879    34357    34527 126229480 134172780

       SbOE0 : SEvt__beginOfEvent_0
       SbOE1 : SEvt__beginOfEvent_1
       SeOE0 : SEvt__endOfEvent_0
        tBOE : t_BeginOfEvent
        tsG3 : t_setGenstep_3
        tsG4 : t_setGenstep_4
        tsG5 : t_setGenstep_5
        tsG6 : t_setGenstep_6
        tsG7 : t_setGenstep_7
        tsG8 : t_setGenstep_8
        tPrL : t_PreLaunch
        tPoL : t_PostLaunch
        tEOE : t_EndOfEvent
    ]NP::descTable_ (12, 13, )
    ...
    NPFold::save("$SREPORT_FOLD")
     resolved to  [/tmp/blyth/opticks/GEOM/J26_1_1_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan_sreport]
    ]sreport.main : CREATED REPORT 
    ]sreport.main






