sajan-optix-missing-init
===========================

Sajan reported an issue
-------------------------

>  (e) As for OKX4Test, as usual I had to activate the OptiX in the main() before 
>      the Opticks was initiated. (This was done by calling things like rtDeviceGetDeviceCount ()).
>      I used the gdml geometry file of CerenkovMinimal example 
>      for this and then it ran OK as can be seen from the snapshot of the prinout listed below.


I have not been able to reproduce this OptiX init problem.
Suggestions for debug are below.


Where OptiX initialization happens in OKX4Test
------------------------------------------------

In okg4/tests/OKX4Test.cc::

    172     ok->profile("_OKX4Test:OKMgr");
    173     OKMgr mgr(argc, argv);  // OpticksHub inside here picks up the gg (last GGeo instanciated) via GGeo::GetInstance 
    174     ok->profile("OKX4Test:OKMgr");
       
Within the instanciation of OKMgr the OptiX context is created via OContext::Create
when OScene is instanciated which subsequently populates the context with the geometry.

     74 OScene::OScene(OpticksHub* hub, const char* cmake_target, const char* ptxrel)
     75     :
     76     m_preinit(preinit()),
     77     m_log(new SLog("OScene::OScene","", LEVEL)),
     78     m_timer(new BTimeKeeper("OScene::")),
     79     m_hub(hub),
     80     m_ok(hub->getOpticks()),
     81     m_ocontext(OContext::Create(m_ok, cmake_target, ptxrel)),


From OContext::Create OContext::CheckDevices the instanciation of VisibleDevices
does device counting and collects metadata about them.

optixrap/OContext.cc::

     233 OContext* OContext::Create(Opticks* ok, const char* cmake_target, const char* ptxrel )
     234 {
     235     LOG(LEVEL) << "[" ;
     236     OKI_PROFILE("_OContext::Create");
     237 
     238     SetupOptiXCachePathEnvvar();
     239 
     240     NMeta* parameters = ok->getParameters();
     241     int rtxmode = ok->getRTX();
     242 #if OPTIX_VERSION_MAJOR >= 6
     243     InitRTX( rtxmode );
     244 #else
     245     assert( rtxmode == 0 && "Cannot use --rtx 1/2/-1 options prior to OptiX 6.0.0 " ) ;
     246 #endif
     247     parameters->add<int>("RTXMode", rtxmode  );
     248 
     249     CheckDevices(ok);
     250 
     251     OKI_PROFILE("_optix::Context::create");
     252     optix::Context context = optix::Context::create();
     253     OKI_PROFILE("optix::Context::create");
     254 
     255     OContext* ocontext = new OContext(context, ok, cmake_target, ptxrel );
     256 
     257     OKI_PROFILE("OContext::Create");
     258     LOG(LEVEL) << "]" ;
     259     return ocontext ;
     260 }
     ...
     204 void OContext::CheckDevices(Opticks* ok)
     205 {
     206     VisibleDevices vdev ;
     207     LOG(info) << std::endl << vdev.desc();
     208 
     209     NMeta* parameters = ok->getParameters();
     210     parameters->add<int>("NumDevices", vdev.num_devices );
     211     parameters->add<std::string>("VisibleDevices", vdev.brief() );
     212 
     ...
     166 struct VisibleDevices
     167 {
     168     unsigned num_devices;
     169     unsigned version;
     170     std::vector<Device> devices ;
     171 
     172     VisibleDevices()
     173     {
     174         RT_CHECK_ERROR(rtDeviceGetDeviceCount(&num_devices));
     175         RT_CHECK_ERROR(rtGetVersion(&version));
     176         for(unsigned i = 0; i < num_devices; ++i)
     177         {
     178             Device d(i);
     179             devices.push_back(d);
     180         }
     181     }




Steps to debug
------------------

To debug this further I suggest you add some debug
in the methods outlined above::

   LOG(LEVEL) << "..." 

Rebuild and install optixrap lib (and others too if necessary)::

    oxrap-
    oxrap--

Set the envvars named after the relevant classes so the output becomes visible, and rerun::

    export OContext=INFO
    oe-   # sources opticks-setup.sh 
    CerenkovMinimal 





