scan-revival-with-new-juno-geometry
=====================================


Checking *scan-vi*::

    520 scan-pf-cmd(){
    521    local num_photons=$1
    522    local cat=$2
    523    local num_abbrev=$(scan-num $num_photons)
    524    local cmd="OKTest --target 62590  --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --save --production --savehit --dbghitmask TO,BT,RE,SC,SA --multievent     10 --xanalytic "
    525    cmd="$cmd $(scan-rngmax-opt $num_photons) $(scan-cat $cat)"
    526    echo $cmd
    527 }


Observe that target is using a hardcoded volume index that is inevitably outdated.  



Informal scan : OKTest --targetpvn pPoolLining --generateoverride -3 --rngmax 3 --cvd 1 --rtx 1 
-----------------------------------------------------------------------------------------------------

* informal scan suggests about 1 second per 1 million photons with the killer geometry removed "-e ~8,"

::

    O[blyth@localhost ~]$ OKTest --targetpvn pPoolLining --generateoverride -3 --rngmax 3 --cvd 1 --rtx 1 
    2021-04-24 21:43:36.255 INFO  [160939] [OPropagator::launch@287] 0 : (0;3000000,1)  launch time 28.3312

    O[blyth@localhost ~]$ OKTest --targetpvn pPoolLining --generateoverride -3 --rngmax 3 --cvd 1 --rtx 1 -e ~8,  
    2021-04-24 21:47:59.108 INFO  [168270] [OPropagator::launch@287] 0 : (0;3000000,1)  launch time 3.09744

    O[blyth@localhost ~]$ OKTest --targetpvn pPoolLining --generateoverride -10 --rngmax 10 --cvd 1 --rtx 1 -e ~8, 
    2021-04-24 21:49:58.268 INFO  [171050] [OPropagator::launch@287] 0 : (0;10000000,1)  launch time 9.59775

    O[blyth@localhost ~]$ OKTest --targetpvn pPoolLining --generateoverride -100 --rngmax 100 --cvd 1 --rtx 1 -e ~8, 

    2021-04-24 21:54:52.943 FATAL [178337] [OCtx::create_buffer@300] skip upload_buffer as num_bytes zero key:OSensorLib_sensor_data
    2021-04-24 21:54:52.943 FATAL [178337] [OCtx::create_buffer@300] skip upload_buffer as num_bytes zero key:OSensorLib_texid
    terminate called after throwing an instance of 'optix::Exception'
      what():  Memory allocation failed (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Out of memory)


100M runs into OOM on TITAN RTX::

    O[blyth@localhost opticks]$ CDeviceTest 
    2021-04-24 22:28:45.543 INFO  [236793] [CDevice::Dump@262] visible devices[0:TITAN_V 1:TITAN_RTX]
    2021-04-24 22:28:45.543 INFO  [236793] [CDevice::Dump@266] idx/ord/mpc/cc:0/0/80/70  11.784 GB  TITAN V
    2021-04-24 22:28:45.543 INFO  [236793] [CDevice::Dump@266] idx/ord/mpc/cc:1/1/72/75  23.652 GB  TITAN RTX

Scan a little to find the photon limit with TITAN RTX:: 

    O[blyth@localhost opticks]$ OContext=INFO OKTest --targetpvn pPoolLining --generateoverride -50 --rngmax 100 --cvd 1 --rtx 1 -e ~8, 
    2021-04-24 22:33:08.138 INFO  [241410] [OPropagator::launch@287] 0 : (0;50000000,1)  launch time 49.2484

Timings in "--production" mode are very simular, howver maybe less memory use (indexing is skipped in production?)::

    O[blyth@localhost opticks]$ OContext=INFO OKTest --targetpvn pPoolLining --generateoverride -50 --rngmax 100 --cvd 1 --rtx 1 -e ~8, --production
    2021-04-24 22:42:18.897 INFO  [255345] [OPropagator::launch@287] 0 : (0;50000000,1)  launch time 51.2637


75M completes the propagation but meets OOM in indexing::

    O[blyth@localhost opticks]$ OContext=INFO OKTest --targetpvn pPoolLining --generateoverride -75 --rngmax 100 --cvd 1 --rtx 1 -e ~8, 

    2021-04-24 22:36:21.684 INFO  [247597] [OContext::launch@820]  entry 0 width 75000000 height 1   printLaunchIndex ( -1 -1 -1) -
    2021-04-24 22:37:36.314 INFO  [247597] [OContext::launch@854] LAUNCH time: 74.6296
    2021-04-24 22:37:36.314 INFO  [247597] [OPropagator::launch@287] 0 : (0;75000000,1)  launch time 74.6296
    2021-04-24 22:37:36.315 INFO  [247597] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    terminate called after throwing an instance of 'thrust::system::detail::bad_alloc'
      what():  std::bad_alloc: cudaErrorMemoryAllocation: out of memory


* OOM at 75M is a bit surprising as managed to get to 400M with Quadro RTX 8000 : which has 48G VRAM (double the TITAN RTX 24G VRAM), 
* so expecting to be able to get to 200M with TITAN RTX
* this suggests that indexing or something else in non-production running more than doubles VRAM usage

  * actually the indexing is of 64-bit "unsigned long long" : so that might cause "double" trouble  


Using production mode allows 75M to complete on TITAN RTX::

    O[blyth@localhost opticks]$ OContext=INFO OKTest --targetpvn pPoolLining --generateoverride -75 --rngmax 100 --cvd 1 --rtx 1 -e ~8, --production
    2021-04-24 22:46:28.332 INFO  [261093] [OPropagator::launch@287] 0 : (0;75000000,1)  launch time 76.6231

Also 100M and 200M::

    O[blyth@localhost opticks]$ OContext=INFO OKTest --targetpvn pPoolLining --generateoverride -100 --rngmax 100 --cvd 1 --rtx 1 -e ~8, --production
    2021-04-24 22:52:26.213 INFO  [269495] [OPropagator::launch@287] 0 : (0;100000000,1)  launch time 102.25

    O[blyth@localhost opticks]$ OContext=INFO OKTest --targetpvn pPoolLining --generateoverride -200 --rngmax 200 --cvd 1 --rtx 1 -e ~8, --production
    2021-04-24 23:12:18.474 INFO  [296907] [OPropagator::launch@287] 0 : (0;200000000,1)  launch time 191.351









Try to find what the name volume 62590 was at that time : its probably pTarget or smth ?
-------------------------------------------------------------------------------------------

::

    epsilon:issues blyth$ opticks-f 62590 
    ./ana/geocache.bash:    62590     62589 [  2:   0/   1]    1 ( 0)     pPoolLining0x4bd25b0  sPoolLining0x4bd1eb0
    ./ana/geocache.bash:    62591     62590 [  3:   0/   1] 2308 ( 0)      pOuterWaterPool0x4bd2b70  sOuterWaterPool0x4bd2960
    ./ana/geocache.bash:    OKTest --target 62590 --xanalytic    
    ./ana/geocache.bash:    ## 62590 : pOuterWaterPool0x4bd2b70  sOuterWaterPool0x4bd2960  
    ./ana/geocache.bash:    OKTest --target 62590 --xanalytic --eye -0.9,0,0
    ./bin/scan.bash:   local cmd="OKTest --target 62590  --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --save --production --savehit --dbghitmask TO,BT,RE,SC,SA --multievent 10 --xanalytic " 
    ./bin/scan.bash:scan-pf-check(){  OKTest --target 62590 --generateoverride -10 --rngmax 10 --cvd 1 --rtx 1 --xanalytic ; }
    ./bin/scan.bash:   local cmd="OKTest --target 62590  --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --save --production --savehit --dbghitmask TO,BT,RE,SC,SA --multievent 10  " 
    epsilon:opticks blyth$ 



::

    epsilon:geocache blyth$ find . -name all_volume_PVNames.txt  -exec wc -l {} \;
      319036 ./OKX4Test_lWorld0x68777d0_PV_g4live/g4ok_gltf/b574f652da8bb005cefa723ecf24b65b/1/GNodeLib/all_volume_PVNames.txt
           3 ./CerenkovMinimal_World_g4live/g4ok_gltf/43bc26d43bba43fc6c680afe1e9df8fa/1/GNodeLib/all_volume_PVNames.txt
       12230 ./G4OKTest_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GNodeLib/all_volume_PVNames.txt
          65 ./OKX4Test_World_LV0x7fe41a880b10_PV_g4live/g4ok_gltf/d550c14c5f5b52b4210e6564133bf938/1/GNodeLib/all_volume_PVNames.txt
      319036 ./OKX4Test_lWorld0x61ee3b0_PV_g4live/g4ok_gltf/d462ec558d40795c0ba134102c68a8b4/1/GNodeLib/all_volume_PVNames.txt
      319036 ./OKX4Test_lWorld0x344f8d0_PV_g4live/g4ok_gltf/732a5daf83a7153b316a2013fcfb1fc2/1/GNodeLib/all_volume_PVNames.txt
      366697 ./G4OKTest_lWorld0x4bc2710_PV_g4live/g4ok_gltf/8068ea569d0c5ca7e26d6db23f17a3fc/1/GNodeLib/all_volume_PVNames.txt
         257 ./OKX4Test_World_LV0x7fe675d2d5f0_PV_g4live/g4ok_gltf/301ccca8d2808b97e14a2ccb14ac3c45/1/GNodeLib/all_volume_PVNames.txt
    epsilon:geocache blyth$ 




geocache-bench360
--------------------

::

     976 geocache-bench360(){ OPTICKS_GROUPCOMMAND="$FUNCNAME $*" geocache-rtxcheck $FUNCNAME $* ; }
     977 geocache-bench360-()
     978 {
     979    type $FUNCNAME
     980    UseOptiX $*
     981 
     982    local factor=4
     983    local cameratype=2  # EQUIRECTANGULAR : all PMTs in view 
     984    local dbg
     985    [ -n "$DBG" ] && dbg="gdb --args" || dbg=""
     986    $dbg OpSnapTest --envkey \
     987                    --target 62594  \
     988                    --eye 0,0,0  \
     989                    --look 0,0,1  \
     990                    --up 1,0,0 \
     991                    --snapconfig "steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25" \
     992                    --size $(geocache-size $factor) \
     993                    --enabledmergedmesh 1,2,3,4,5 \
     994                    --cameratype $cameratype \
     995                    --embedded \
     996                    $*
     997 }

::

    epsilon:opticks blyth$ find . -name OpSnapTest.cc
    ./okop/tests/OpSnapTest.cc

    epsilon:opticks blyth$ okop
    epsilon:okop blyth$ vi tests/OpSnapTest.cc 

    042 const char* TMPDIR = "$TMP/okop/OpSnapTest" ;
     43 
     44 int main(int argc, char** argv)
     45 {
     46     OPTICKS_LOG(argc, argv);
     47     Opticks ok(argc, argv, "--tracer");   // tempted to put --embedded here 
     48     OpMgr op(&ok);
     49     op.snap(TMPDIR);
     50     return 0 ;
     51 }

::

     54 OpMgr::OpMgr(Opticks* ok )
     55     :
     56     m_preinit(Preinit()),
     57     m_ok(ok ? ok : Opticks::GetInstance()),
     58     m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry OR adopt a preexisting GGeo instance
     59     m_idx(new OpticksIdx(m_hub)),
     60     m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
     61     m_gen(m_hub->getGen()),
     62     m_run(m_ok->getRun()),
     63     m_propagator(new OpPropagator(m_hub, m_idx)),
     64     m_count(0)
     65 {
     66     init();
     67 }

     74 void OpMgr::init()
     75 {
     76     LOG(LEVEL);
     77     bool g4gun = m_ok->getSourceCode() == OpticksGenstep_G4GUN ;
     78     if(g4gun)
     79          LOG(fatal) << "OpMgr doesnt support G4GUN, other that via loading (TO BE IMPLEMENTED) " ;
     80     assert(!g4gun);
     81 
     82     //m_ok->dumpParameters("OpMgr::init");
     83 }



::

    epsilon:tmp blyth$ opticks-f makeSimpleTorchStep
    ./opticksgeo/OpticksGen.cc:    TorchStepNPY* torchstep = m_ok->makeSimpleTorchStep(gencode);
    ./optickscore/Opticks.hh:       TorchStepNPY*        makeSimpleTorchStep(unsigned gencode);
    ./optickscore/Opticks.cc:Opticks::makeSimpleTorchStep
    ./optickscore/Opticks.cc:TorchStepNPY* Opticks::makeSimpleTorchStep(unsigned gencode)
    epsilon:opticks blyth$ 



OpSnapTest
-------------

Getting back into the flow::

    OpMgr=INFO OpticksHub=INFO OpticksGen=INFO OpPropagator=INFO OpTracer=INFO OpSnapTest 

    OpticksAim=INFO OpSnapTest --targetpvn pAcrylic


  
    OpSnapTest --snapconfig steps=11,x0=-0.4,x1=-0.3,y0=0,z0=0    ## all the interesting views in this range 


    OpSnapTest --snapconfig steps=101,x0=-0.4,x1=-0.3,y0=0,z0=0



Getting --targetpvn to work
-----------------------------

::

    epsilon:ggeo blyth$ GNodeLibTest --targetpvn pCentralDetector




