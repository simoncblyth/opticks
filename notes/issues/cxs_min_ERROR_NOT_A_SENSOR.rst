cxs_min_ERROR_NOT_A_SENSOR
=============================


Looks like the formerly default resolved geometry folder did not have PMT info
resulting in lots of dumping::

    P[blyth@localhost CSGOptiX]$ TEST=debug ~/o/cxs_min.sh 
    /home/blyth/o/cxs_min.sh : FOUND A_CFBaseFromGEOM /data/blyth/opticks/G4CXOpticks_setGeometry_Test/J_2024aug27 containing CSGFoundry/prim.npy
    knobs is a function
    knobs () 
    { 
        type $FUNCNAME;
        local exceptionFlags;
        local debugLevel;
        local optLevel;
        exceptionFlags=NONE;
        debugLevel=NONE;
        optLevel=LEVEL_3;
        export PIP__CreatePipelineOptions_exceptionFlags=$exceptionFlags;
        export PIP__CreateModule_debugLevel=$debugLevel;
        export PIP__linkPipeline_debugLevel=$debugLevel;
        export PIP__CreateModule_optLevel=$optLevel
    }
                    GEOM : J_2024aug27 
                    BASE : /data/blyth/opticks/GEOM/J_2024aug27 
                    TEST : debug 
                  LOGDIR : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 
                 BINBASE : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest 
                     CVD :  
    CUDA_VISIBLE_DEVICES : 1 
                    SDIR : /data/blyth/junotop/opticks/CSGOptiX 
                    FOLD :  
                     LOG :  
                    NEVT :  
    2024-10-14 16:26:52.396  396417730 : [/home/blyth/o/cxs_min.sh 
    //qsim::propagate_at_surface_CustomART idx      16 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx      18 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx      22 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx      26 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    ...
    //qsim::propagate_at_surface_CustomART idx      69 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx      25 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx      28 lpmtid -1 : ERROR NOT-A-SENSOR : NAN_ABORT 
    2024-10-14 16:26:55.365  365873422 : ]/home/blyth/o/cxs_min.sh 



Switching to cvmfs geometry by swapping C and A below avoids that::

     68 Resolve_CFBaseFromGEOM()
     69 {
     70    : LOOK FOR CFBase directory containing CSGFoundry geometry
     71    : HMM COULD PUT INTO GEOM.sh TO AVOID DUPLICATION ? BUT TOO MUCH HIDDEN ?
     72    : G4CXOpticks_setGeometry_Test GEOM TAKES PRECEDENCE OVER .opticks/GEOM
     73    : HMM : FOR SOME TESTS WANT TO LOAD GDML BUT FOR OTHERS CSGFoundry
     74    : to handle that added gdml resolution to eg g4cx/tests/GXTestRunner.sh
     75 
     76    local C_CFBaseFromGEOM=$TMP/G4CXOpticks_setGeometry_Test/$GEOM
     77    local B_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
     78    local A_CFBaseFromGEOM=/cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/$GEOM
     79 



