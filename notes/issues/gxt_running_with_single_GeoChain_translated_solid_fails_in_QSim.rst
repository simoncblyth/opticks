gxt_running_with_single_GeoChain_translated_solid_fails_in_QSim
==================================================================


Setup
--------

Translate single PMTSim solid with GeoChain "nmskSolidMaskVirtual"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit GEOM.txt to identify the PMTSim solid::

    geom
         28 nmskSolidMaskVirtual_XZ

Run the geochain translation::

    gc
    ./translate.sh 


Introduce new "full" geometry to "geom_" using CFBaseFromGEOM approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     37 #geom=J001
     38 #geom=J003
     39 geom=nmskSolidMaskVirtual

     ..

     73 elif [ "$GEOM" == "J003" ]; then
     74 
     75     export J003_CFBaseFromGEOM=$HOME/.opticks/ntds3/G4CXOpticks
     76 
     77 else
     78     case $GEOM in
     79         nmsk*) export ${GEOM}_CFBaseFromGEOM=/tmp/$USER/opticks/GeoChain/$GEOM ;;
     80     esac
     81 fi



Try gxt run
~~~~~~~~~~~~~~~~

The GeoChain translated geometry is loaded, but get fail in QSim QCerenkov setup.
QCerenkov is not needed for simtrace, so can probably just do some skipping in simtrace running SEventConfig::IsRGModeSimtrace()::

    gx
    ./gxt.sh


    N[blyth@localhost g4cx]$ ./gxt.sh dbg
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                               gp_ : nmskSolidMaskVirtual_GDMLPath 
                                gp :  
                               cg_ : nmskSolidMaskVirtual_CFBaseFromGEOM 
                                cg : /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual 
                       TMP_GEOMDIR : /tmp/blyth/opticks/nmskSolidMaskVirtual 
                           GEOMDIR : /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual 

                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON_.sh
                              GEOM : nmskSolidMaskVirtual
              OPTICKS_INPUT_PHOTON : RandomSpherical10_f8.npy
      OPTICKS_INPUT_PHOTON_ABSPATH : /home/blyth/.opticks/InputPhotons/RandomSpherical10_f8.npy
        OPTICKS_INPUT_PHOTON_LABEL : RandomSpherical10
                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON.sh 
                         ScriptDir : ./../bin 
              OPTICKS_INPUT_PHOTON : RandomSpherical10_f8.npy 
        OPTICKS_INPUT_PHOTON_FRAME :  
      OPTICKS_INPUT_PHOTON_ABSPATH : /home/blyth/.opticks/InputPhotons/RandomSpherical10_f8.npy 

                       BASH_SOURCE : ./../bin/COMMON.sh
                              GEOM : nmskSolidMaskVirtual
              OPTICKS_INPUT_PHOTON : RandomSpherical10_f8.npy
        OPTICKS_INPUT_PHOTON_FRAME : 
                               MOI : 
    gdb -ex r --args G4CXSimtraceTest -ex r
    Fri Aug 26 18:24:24 CST 2022

    2022-08-26 18:24:28.425 INFO  [182404] [G4CXOpticks::setGeometry@137]  argumentless 
    2022-08-26 18:24:28.425 INFO  [182404] [G4CXOpticks::setGeometry@150] [ CFBASEFromGEOM 
    2022-08-26 18:24:28.425 INFO  [182404] [G4CXOpticks::setGeometry@151] [ CSGFoundry::Load 
    2022-08-26 18:24:28.426 INFO  [182404] [CSGFoundry::Load@2456] [ argumentless 
    2022-08-26 18:24:28.427 INFO  [182404] [CSGFoundry::ResolveCFBase@2519]  cfbase /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual readable 1
    2022-08-26 18:24:28.429 INFO  [182404] [CSGFoundry::setMeta@138]                      : -
    2022-08-26 18:24:28.429 INFO  [182404] [CSGFoundry::setMeta@138]                 HOME : /home/blyth
    2022-08-26 18:24:28.429 INFO  [182404] [CSGFoundry::setMeta@138]                 USER : blyth
    2022-08-26 18:24:28.429 INFO  [182404] [CSGFoundry::setMeta@138]               SCRIPT : -
    2022-08-26 18:24:28.429 INFO  [182404] [CSGFoundry::setMeta@138]                  PWD : /data/blyth/junotop/opticks/g4cx
    2022-08-26 18:24:28.429 INFO  [182404] [CSGFoundry::setMeta@138]              CMDLINE : -
    2022-08-26 18:24:28.429 INFO  [182404] [CSGFoundry::load@2264] [ loaddir /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual/CSGFoundry
    2022-08-26 18:24:28.430 INFO  [182404] [CSGFoundry::loadArray@2617]  ni     1 nj 3 nk 4 solid.npy
    2022-08-26 18:24:28.430 INFO  [182404] [CSGFoundry::loadArray@2617]  ni     1 nj 4 nk 4 prim.npy
    2022-08-26 18:24:28.430 INFO  [182404] [CSGFoundry::loadArray@2617]  ni     7 nj 4 nk 4 node.npy
    2022-08-26 18:24:28.430 INFO  [182404] [CSGFoundry::loadArray@2617]  ni     3 nj 4 nk 4 tran.npy
    2022-08-26 18:24:28.430 INFO  [182404] [CSGFoundry::loadArray@2617]  ni     3 nj 4 nk 4 itra.npy
    2022-08-26 18:24:28.430 INFO  [182404] [CSGFoundry::loadArray@2617]  ni     1 nj 4 nk 4 inst.npy
    2022-08-26 18:24:28.430 INFO  [182404] [CSGFoundry::load@2288] [ SSim::Load 
    2022-08-26 18:24:28.431 INFO  [182404] [CSGFoundry::load@2290] ] SSim::Load 
    2022-08-26 18:24:28.431 INFO  [182404] [CSGFoundry::load@2295] ] loaddir /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual/CSGFoundry
    ...
    2022-08-26 18:24:28.432 INFO  [182404] [G4CXOpticks::setGeometry@254]  ] new SEvt 
    2022-08-26 18:24:28.432 INFO  [182404] [G4CXOpticks::setGeometry@259] [ CSGOptiX::Create 
    2022-08-26 18:24:28.433 INFO  [182404] [CSGFoundry::getOriginCFBase@2197]  CAUTION HOW YOU USE THIS : MISUSE CAN EASILY LEAD TO INCONSISTENCY BETWEEN RESULTS AND GEOMETRY 
    2022-08-26 18:24:28.433 INFO  [182404] [CSGOptiX::Create@198] [ fd.descBase CSGFoundry.descBase 
     CFBase       /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual
     OriginCFBase /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual

    2022-08-26 18:24:28.433 INFO  [182404] [CSGOptiX::InitSim@177] [
    2022-08-26 18:24:28.433 INFO  [182404] [QSim::UploadComponents@102] [ ssim 0xd7e580
    2022-08-26 18:24:28.433 INFO  [182404] [QSim::UploadComponents@104] [ new QBase
    2022-08-26 18:24:28.433 INFO  [182404] [QBase::init@51] [ QU::UploadArray 
    2022-08-26 18:24:28.449 INFO  [182404] [QBase::init@53] ] QU::UploadArray : takes ~0.25-0.3s : appearing in analog timings as it is first GPU contact 
    2022-08-26 18:24:28.449 INFO  [182404] [QSim::UploadComponents@106] ] new QBase : latency here of about 0.3s from first device access, if latency of >1s need to start nvidia-persistenced 
    2022-08-26 18:24:28.449 INFO  [182404] [QSim::UploadComponents@107] QBase::desc base 0x1fd9570 d_base 0x7fffbfc00000 base.desc qbase::desc pidx 4294967295
    2022-08-26 18:24:28.449 INFO  [182404] [QSim::UploadComponents@109] [ new QRng 
    2022-08-26 18:24:28.626 INFO  [182404] [QSim::UploadComponents@111] ] new QRng 
    2022-08-26 18:24:28.626 INFO  [182404] [QSim::UploadComponents@113] QRng path /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin rngmax 1000000 qr 0x21bd2c0 d_qr 0x7fffbfc00200
    2022-08-26 18:24:28.626 ERROR [182404] [QSim::UploadComponents@120]  optical and bnd null  SSim::OPTICAL optical.npy SSim::BND bnd.npy
    2022-08-26 18:24:28.627 ERROR [182404] [QDebug::MakeInstance@58] AS NO QBnd at QDebug::MakeInstance the qdebug cerenkov genstep is using default matline of zero 
    THIS MEANS qdebug CERENKOV GENERATION WILL LIKELY INFINITE LOOP AND TIMEOUT 
     cerenkov_matline 0
     TO FIX THIS YOU PROBABLY NEED TO RERUN THE GEOMETRY CONVERSION TO UPDATE THE PERSISTED SSim IN CSGFoundry/SSim 
    2022-08-26 18:24:28.627 INFO  [182404] [QSim::UploadComponents@132] QDebug::desc  dbg 0x21c0ec0 d_dbg 0x7fffbfc00400
     QState::Desc QState::Desc
    material1 ( 1.000,1000.000,1000.000, 0.000) 
    material2 ( 1.500,1000.000,1000.000, 0.000) 
    m1group2  (300.000, 0.000, 0.000, 0.000) 
    surface   ( 0.000, 0.000, 0.000, 0.000) 
    optical   (     0,     0,     0,     0) 

     dbg.p.desc  pos ( 0.000, 0.000, 0.000)  t     0.000  mom ( 1.000, 0.000, 0.000)  iindex 1065353216  pol ( 0.000, 1.000, 0.000)  wl  500.000   bn 0 fl 0 id 0 or 1 ix 0 fm 0 ab    ii 1065353216
    2022-08-26 18:24:28.627 ERROR [182404] [QSim::UploadComponents@145]   propcom null, SSim::PROPCOM propcom.npy
    2022-08-26 18:24:28.627 ERROR [182404] [QSim::UploadComponents@152]  icdf null, SSim::ICDF icdf.npy
    G4CXSimtraceTest: /data/blyth/junotop/opticks/qudarap/QCerenkov.cc:81: static qcerenkov* QCerenkov::MakeInstance(): Assertion `bnd' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) 

    (gdb) bt
    #0  0x00007fffeb848387 in raise () from /lib64/libc.so.6
    #1  0x00007fffeb849a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffeb8411a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffeb841252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffed88ca4e in QCerenkov::MakeInstance () at /data/blyth/junotop/opticks/qudarap/QCerenkov.cc:81
    #5  0x00007fffed88cb9e in QCerenkov::QCerenkov (this=0x21f8a20) at /data/blyth/junotop/opticks/qudarap/QCerenkov.cc:129
    #6  0x00007fffed825ad5 in QSim::UploadComponents (ssim=0xd7e580) at /data/blyth/junotop/opticks/qudarap/QSim.cc:162
    #7  0x00007fffefa303bd in CSGOptiX::InitSim (ssim=0xd7e580) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:183
    #8  0x00007fffefa306a2 in CSGOptiX::Create (fd=0xd73940) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:202
    #9  0x00007ffff7b8fc80 in G4CXOpticks::setGeometry (this=0x7fffffff59a0, fd_=0xd73940) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:260
    #10 0x00007ffff7b8f133 in G4CXOpticks::setGeometry (this=0x7fffffff59a0) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:158
    #11 0x0000000000408154 in main (argc=3, argv=0x7fffffff5ef8) at /data/blyth/junotop/opticks/g4cx/tests/G4CXSimtraceTest.cc:24
    (gdb) 

::

    244 void G4CXOpticks::setGeometry(CSGFoundry* fd_)
    245 {
    246 #ifdef __APPLE__
    247     return ;
    248 #endif
    249     fd = fd_ ;
    250     LOG(LEVEL) << "[ fd " << fd ;
    251 
    252     LOG(LEVEL) << " [ new SEvt " ;
    253     SEvt* sev = new SEvt ;
    254     LOG(LEVEL) << " ] new SEvt " ;
    255 
    256     sev->setReldir("ALL");
    257     sev->setGeo((SGeo*)fd);
    258 
    259     LOG(LEVEL) << "[ CSGOptiX::Create " ;
    260     cx = CSGOptiX::Create(fd);   // uploads geometry to GPU 
    261     LOG(LEVEL) << "] CSGOptiX::Create " ;
    262     qs = cx->sim ;
    263     LOG(LEVEL)  << " cx " << cx << " qs " << qs << " QSim::Get " << QSim::Get() ;
    264 
    265 




Try to early exit QSim::UploadComponents for simtrace running 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



::

     101 void QSim::UploadComponents( const SSim* ssim  )
     102 {
     103     LOG(LEVEL) << "[ ssim " << ssim ;
     104 
     105     LOG(LEVEL) << "[ new QBase" ;
     106     QBase* base = new QBase ;
     107     LOG(LEVEL) << "] new QBase : latency here of about 0.3s from first device access, if latency of >1s need to start nvidia-persistenced " ;
     108     LOG(LEVEL) << base->desc();
     109 
     110 
     111     bool is_simtrace = SEventConfig::IsRGModeSimtrace() ;
     112     if(is_simtrace) LOG(LEVEL) << " early exit for simtrace running " ;
     113 
     114 


This fails at launch::

    terminate called after throwing an instance of 'sutil::CUDA_Exception'
      what():  CUDA error on synchronize with error 'an illegal memory access was encountered' (/data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:785)

    ./gxt.sh: line 155: 184118 Aborted                 (core dumped) $bin
    ./gxt.sh run G4CXSimtraceTest error
    N[blyth@localhost g4cx]$ 


Instead try skipping QCerenkov in QSim::UploadComponents::

     165 
     166     // TODO: make this more like the others : acting on the available inputs rather than the mode
     167     bool is_simtrace = SEventConfig::IsRGModeSimtrace() ;
     168     if(is_simtrace == false )
     169     {
     170         QCerenkov* cerenkov = new QCerenkov  ;
     171         LOG(LEVEL) << cerenkov->desc();
     172     }
     173     else
     174     {
     175         LOG(LEVEL) << " skip QCerenkov for simtrace running " ;
     176     }


Skipping QCerenkov completes but problem with ana finding CFBase::


    epsilon:g4cx blyth$ ./gxt.sh ana
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                               gp_ : nmskSolidMaskVirtual_GDMLPath 
                                gp :  
                               cg_ : nmskSolidMaskVirtual_CFBaseFromGEOM 
                                cg : /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual 
                       TMP_GEOMDIR : /tmp/blyth/opticks/nmskSolidMaskVirtual 
                           GEOMDIR : /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual 

                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON_.sh
                              GEOM : nmskSolidMaskVirtual
              OPTICKS_INPUT_PHOTON : RandomSpherical10_f8.npy
      OPTICKS_INPUT_PHOTON_ABSPATH : /Users/blyth/.opticks/InputPhotons/RandomSpherical10_f8.npy
        OPTICKS_INPUT_PHOTON_LABEL : RandomSpherical10
                       BASH_SOURCE : ./../bin/OPTICKS_INPUT_PHOTON.sh 
                         ScriptDir : ./../bin 
              OPTICKS_INPUT_PHOTON : RandomSpherical10_f8.npy 
        OPTICKS_INPUT_PHOTON_FRAME :  
      OPTICKS_INPUT_PHOTON_ABSPATH : /Users/blyth/.opticks/InputPhotons/RandomSpherical10_f8.npy 

                       BASH_SOURCE : ./../bin/COMMON.sh
                              GEOM : nmskSolidMaskVirtual
              OPTICKS_INPUT_PHOTON : RandomSpherical10_f8.npy
        OPTICKS_INPUT_PHOTON_FRAME : 
                               MOI : 
    CSGFoundry.CFBase returning [/], note:[via CFBASE] 
    ERROR CSGFoundry.CFBase returned None OR non-existing CSGFoundry dir so cannot CSGFoundry.Load
    Fold : symbol t base /tmp/blyth/opticks/GeoChain/nmskSolidMaskVirtual/G4CXSimtraceTest/ALL 
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    ~/opticks/g4cx/tests/G4CXSimtraceTest.py in <module>
         61     a = Fold.Load("$A_FOLD", symbol="a")
         62     b = Fold.Load("$B_FOLD", symbol="b")
    ---> 63     print("cf.cfbase : %s " % cf.cfbase)
         64 
         65     print("---------Fold.Load.done")

    AttributeError: 'NoneType' object has no attribute 'cfbase'
    > /Users/blyth/opticks/g4cx/tests/G4CXSimtraceTest.py(63)<module>()
         61     a = Fold.Load("$A_FOLD", symbol="a")
         62     b = Fold.Load("$B_FOLD", symbol="b")
    ---> 63     print("cf.cfbase : %s " % cf.cfbase)
         64 
         65     print("---------Fold.Load.done")

    ipdb>                                                                                                                                                                                                     


Fixed this by changing gxt.sh grab to grab one level up::

    185 if [ "grab" == "$arg" ]; then
    186     #source $gxtdir/../bin/rsync.sh $UBASE 
    187     source $gxtdir/../bin/rsync.sh $UGEOMDIR
    188 fi


ana/feature.py CSG/CSGFoundry.py needed changes to handle no SSim. 


