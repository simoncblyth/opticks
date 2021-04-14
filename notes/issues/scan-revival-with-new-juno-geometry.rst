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



Getting --targetpvn to work
-----------------------------

::

    epsilon:ggeo blyth$ GNodeLibTest --targetpvn pCentralDetector




