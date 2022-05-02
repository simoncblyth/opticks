remove-OpticksCore-dependency-from-CSGOptiX
=============================================

Reimplementing the Composition tree of classes for the new workflow ?
-----------------------------------------------------------------------------------

* :doc:`reimplement_Composition`


CVD control needs to be moved  : DONE : simple headeronly replacement im SCVD.hh
-----------------------------------------------------------------------------------

::

    epsilon:sysrap blyth$ opticks-f CVD
    ./ana/profilesmry.py:    def UCVD(cls, c ):
    ./ana/profilesmry.py:        default_cvd = os.environ.get("OPTICKS_DEFAULT_INTEROP_CVD", "0")  ## hmm this is broken by scan-rsync when looking as scans from another machine
    ./ana/profilesmry.py:        ucvd = ProfileSmry.UCVD(c)         
    ./CSGOptiX/tests/CSGOptiXRenderTest.py:    cvdver = os.environ.get("CVDVER", "cvd0/50001" ) 
    ./CSGOptiX/tests/CXRaindropTest.cc:TODO: bring over CVD mechanics for GPU in use control 
    ./CSGOptiX/cxr.sh:export CVD=${CVD:-$cvd}    # --cvd 
    ./CSGOptiX/cxr.sh:vars="CVD EMM MOI EYE TOP SLA CAM TMIN ZOOM CAMERATYPE OPTICKS_GEOM OPTICKS_RELDIR SIZE SIZESCALE"
    ./CSGOptiX/cxr.sh:$GDB $bin $DIV --nameprefix "$NAMEPREFIX" --cvd $CVD -e "$EMM" --size "$SIZE" --sizescale "$SIZESCALE" --solid_label "$SLA" $* 
    ./cudarap/CDevice.cu:const char* CDevice::CVD = "CUDA_VISIBLE_DEVICES" ; 
    ./cudarap/CDevice.cu:    char* cvd = getenv(CVD); 
    ./cudarap/CDevice.hh:    static const char* CVD ; 
    ./bin/scan.bash:      #echo cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_2
    ./bin/scan.bash:cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_0
    ./bin/scan.bash:cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_1
    ./bin/scan.bash:cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_0
    ./bin/scan.bash:cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_1
    ./bin/scan.bash:cvd_${OPTICKS_DEFAULT_INTEROP_CVD}_rtx_2
    ./optickscore/Opticks.hh:       const char* getCVD() const ;
    ./optickscore/Opticks.hh:       const char* getDefaultCVD() const ;
    ./optickscore/Opticks.hh:       const char* getUsedCVD() const ;
    ./optickscore/Opticks.hh:       void postconfigureCVD() ;
    ./optickscore/OpticksCfg.cc:    m_cvd(SSys::getenvvar("CVD","")),
    ./optickscore/OpticksCfg.cc:const std::string& OpticksCfg<Listener>::getCVD()
    ./optickscore/OpticksCfg.hh:     const std::string& getCVD();
    ./optickscore/Opticks.cc:TODO: relocate this to SOpticks or SOpticksResource : sticking point is getUsedCVD for "--cvd" option
    ./optickscore/Opticks.cc:    const char* ucvd = getUsedCVD(); 
    ./optickscore/Opticks.cc:Opticks::getCVD
    ./optickscore/Opticks.cc:const char* Opticks::getCVD() const 
    ./optickscore/Opticks.cc:    const std::string& cvdcfg = m_cfg->getCVD();  
    ./optickscore/Opticks.cc:const char* Opticks::getDefaultCVD() const 
    ./optickscore/Opticks.cc:    const char* dk = "OPTICKS_DEFAULT_INTEROP_CVD" ; 
    ./optickscore/Opticks.cc:const char* Opticks::getUsedCVD() const 
    ./optickscore/Opticks.cc:    const char* cvd = getCVD(); 
    ./optickscore/Opticks.cc:    const char* dcvd = getDefaultCVD(); 
    ./optickscore/Opticks.cc:        LOG(error) << " --interop mode with no cvd specified, adopting OPTICKS_DEFAULT_INTEROP_CVD hinted by envvar [" << dcvd << "]" ;   
    ./optickscore/Opticks.cc:    postconfigureCVD(); 
    ./optickscore/Opticks.cc:Opticks::postconfigureCVD
    ./optickscore/Opticks.cc:to avoid failures. To automate that the envvar OPTICKS_DEFAULT_INTEROP_CVD 
    ./optickscore/Opticks.cc:void Opticks::postconfigureCVD()
    ./optickscore/Opticks.cc:    const char* ucvd = getUsedCVD() ;  
    ./thrustrap/tests/TDeviceDumpTest.cc:2021-06-07 16:37:31.044 ERROR [117740] [Opticks::postconfigureCVD@3000]  --cvd [-] option internally sets CUDA_VISIBLE_DEVICES []
    ./thrustrap/tests/TDeviceDumpTest.cc:epsilon:thrustrap blyth$ CVD=- TDeviceDumpTest 
    ./thrustrap/tests/TDeviceDumpTest.cc:2021-06-07 16:37:39.771 ERROR [117796] [Opticks::postconfigureCVD@3000]  --cvd [-] option internally sets CUDA_VISIBLE_DEVICES []
    epsilon:opticks blyth$ 




::

    3200 const char* Opticks::getCVD() const
    3201 {
    3202     const std::string& cvdcfg = m_cfg->getCVD();
    3203     const char* cvd = cvdcfg.empty() ? NULL : cvdcfg.c_str() ;
    3204     return cvd ;
    3205 }
    3206 
    3207 const char* Opticks::getDefaultCVD() const
    3208 {
    3209     const char* dk = "OPTICKS_DEFAULT_INTEROP_CVD" ;
    3210     const char* dcvd = SSys::getenvvar(dk) ;
    3211     return dcvd ;
    3212 }
    3213 
    3214 const char* Opticks::getUsedCVD() const
    3215 {
    3216     const char* cvd = getCVD();
    3217     const char* dcvd = getDefaultCVD();
    3218     const char* ucvd =  cvd == NULL && isInterop() && dcvd != NULL ? dcvd : cvd ;
    3219 
    3220     LOG(LEVEL)
    3221         << " cvd " << cvd
    3222         << " dcvd " << dcvd
    3223         << " isInterop " << isInterop()
    3224         << " ucvd " << ucvd
    3225         ;
    3226 
    3227     if( cvd == NULL && isInterop() && dcvd != NULL )
    3228     {
    3229         LOG(error) << " --interop mode with no cvd specified, adopting OPTICKS_DEFAULT_INTEROP_CVD hinted by envvar [" << dcvd << "]" ;
    3230         ucvd = dcvd ;
    3231     }
    3232     return ucvd ;
    3233 }




    3281 /**
    3282 Opticks::postconfigureCVD
    3283 ---------------------------
    3284 
    3285 When "--cvd" option is on the commandline this internally sets 
    3286 the CUDA_VISIBLE_DEVICES envvar to the string argument provided.
    3287 For example::
    3288  
    3289    --cvd 0 
    3290    --cvd 1
    3291    --cvd 0,1,2,3
    3292 
    3293    --cvd -   # '-' is treated as a special token representing an empty string 
    3294              # which easier to handle than an actual empty string 
    3295 
    3296 In interop mode on multi-GPU workstations it is often necessary 
    3297 to set the --cvd to match the GPU that is driving the monitor
    3298 to avoid failures. To automate that the envvar OPTICKS_DEFAULT_INTEROP_CVD 
    3299 is consulted when no --cvd option is provides, acting as a default value.
    3300 
    3301 **/
    3302 
    3303 void Opticks::postconfigureCVD()
    3304 {
    3305     const char* ucvd = getUsedCVD() ;
    3306     if(ucvd)
    3307     {
    3308         const char* ek = "CUDA_VISIBLE_DEVICES" ;
    3309         LOG(LEVEL) << " setting " << ek << " envvar internally to " << ucvd ;
    3310         char special_empty_token = '-' ;   // when ucvd is "-" this will replace it with an empty string
    3311         SSys::setenvvar(ek, ucvd, true, special_empty_token );    // Opticks::configure setting CUDA_VISIBLE_DEVICES
    3312 
    3313         const char* chk = SSys::getenvvar(ek);
    3314         LOG(error) << " --cvd [" << ucvd << "] option internally sets " << ek << " [" << chk << "]" ;
    3315     }
    3316 }






SBT.cc using SolidSelection vector from Opticks : relocate where ?
----------------------------------------------------------------------

* inside CSGFoundry would seem the natural place as it is the primary user 


::

    epsilon:CSGOptiX blyth$ opticks-f getSolidSelection
    ./CSGOptiX/SBT.cc:1. Opticks::getSolidSelection
    ./CSGOptiX/SBT.cc:    solid_selection(ok->getSolidSelection()),   // vector<unsigned>
    ./CSGOptiX/tests/CSGOptiXRenderTest.cc:    solid_selection(ok->getSolidSelection()), //  NB its not set yet, that happens below 
    ./CSGOptiX/Six.cc:    solid_selection(ok->getSolidSelection()),
    ./sysrap/SOpticks.hh:    std::vector<unsigned>&        getSolidSelection() ;
    ./sysrap/SOpticks.hh:    const std::vector<unsigned>&  getSolidSelection() const ;
    ./sysrap/SOpticks.cc:std::vector<unsigned>&  SOpticks::getSolidSelection() 
    ./sysrap/SOpticks.cc:const std::vector<unsigned>&  SOpticks::getSolidSelection() const 
    ./optickscore/Opticks.hh:       std::vector<unsigned>&  getSolidSelection() ; 
    ./optickscore/Opticks.hh:       const std::vector<unsigned>& getSolidSelection() const ;
    ./optickscore/Opticks.cc:std::vector<unsigned>& Opticks::getSolidSelection()
    ./optickscore/Opticks.cc:const std::vector<unsigned>& Opticks::getSolidSelection() const 



* vector is populated based on solid_label argument, that solid_label can instead come in via envvar  

::

     88 CSGOptiXRenderTest::CSGOptiXRenderTest(int argc, char** argv)
     89     :
     90     ok(InitOpticks(argc, argv)),
     91     solid_label(ok->getSolidLabel()),         // --solid_label   used for selecting solids from the geometry 
     92     solid_selection(ok->getSolidSelection()), //  NB its not set yet, that happens below 
     93     fd(CSGFoundry::Load()),
     94     cx(nullptr),


::

    epsilon:CSG blyth$ opticks-f findSolidIdx
    ./CSGOptiX/tests/CSGOptiXRenderTest.cc:        fd->findSolidIdx(solid_selection, solid_label); 
    ./CSG/CSGFoundry.h:    int findSolidIdx(const char* label) const  ; // -1 if not found
    ./CSG/CSGFoundry.h:    void findSolidIdx(std::vector<unsigned>& solid_idx, const char* label) const ; 
    ./CSG/tests/CMakeLists.txt:    CSGFoundry_findSolidIdx_Test.cc
    ./CSG/tests/CSGFoundry_findSolidIdx_Test.cc:void test_findSolidIdx(const CSGFoundry* fd, int argc, char** argv)
    ./CSG/tests/CSGFoundry_findSolidIdx_Test.cc:        fd->findSolidIdx(solid_selection, sla );   
    ./CSG/tests/CSGFoundry_findSolidIdx_Test.cc:    test_findSolidIdx(fd, argc, argv); 
    ./CSG/CSGFoundry.cc:int CSGFoundry::findSolidIdx(const char* label) const 
    ./CSG/CSGFoundry.cc:CSGFoundry::findSolidIdx
    ./CSG/CSGFoundry.cc:void CSGFoundry::findSolidIdx(std::vector<unsigned>& solid_idx, const char* label) const 
    ./CSG/CSGFoundry.cc:    findSolidIdx(solidIdx, label); 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 



