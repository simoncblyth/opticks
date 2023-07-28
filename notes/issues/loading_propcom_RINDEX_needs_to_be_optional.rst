loading_propcom_RINDEX_needs_to_be_optional
==============================================

Setup
------

0. download test GDML into $HOME/.opticks/GEOM/simpleLArTPC/origin.gdml 
1. use "GEOM" to set envvar to "simpleLArTPC"
2. check conversion with, gxt::

    ./G4CXOpticks_setGeometry_Test.sh


Issue
--------

The experimental propcom is inappropriately assuming existance of an LS_ori/RINDEX.npy 
 

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGINT
      * frame #0: 0x00007fff56dccb66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff56f97080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff56cda6fe libsystem_c.dylib`raise + 26
        frame #3: 0x00000001068fe623 libGGeo.dylib`NP::load(this=0x000000010bd665a0, path="/Users/blyth/.opticks/GEOM/simpleLArTPC/GGeo/GScintillatorLib/LS_ori/RINDEX.npy") at NP.hh:4422
        frame #4: 0x00000001068fcbcb libGGeo.dylib`NP::Load_(path="/Users/blyth/.opticks/GEOM/simpleLArTPC/GGeo/GScintillatorLib/LS_ori/RINDEX.npy") at NP.hh:2326
        frame #5: 0x00000001068fc45c libGGeo.dylib`NP::Load(path_="/Users/blyth/.opticks/GEOM/simpleLArTPC/GGeo/GScintillatorLib/LS_ori/RINDEX.npy") at NP.hh:2303
        frame #6: 0x0000000106a80191 libGGeo.dylib`SPropMockup::Combination(base="$HOME/.opticks/GEOM/$GEOM", relp="GGeo/GScintillatorLib/LS_ori/RINDEX.npy") at SPropMockup.h:71
        frame #7: 0x0000000106a7465b libGGeo.dylib`SPropMockup::CombinationDemo() at SPropMockup.h:42
        frame #8: 0x0000000106a73ae9 libGGeo.dylib`GGeo::convertSim_Prop(this=0x000000010ca36f60) const at GGeo.cc:2639
        frame #9: 0x0000000106a73039 libGGeo.dylib`GGeo::convertSim(this=0x000000010ca36f60) const at GGeo.cc:2548
        frame #10: 0x00000001067a5259 libCSG_GGeo.dylib`CSG_GGeo_Convert::convertSim(this=0x00007ffeefbfc770) at CSG_GGeo_Convert.cc:201
        frame #11: 0x00000001067a177a libCSG_GGeo.dylib`CSG_GGeo_Convert::convert(this=0x00007ffeefbfc770) at CSG_GGeo_Convert.cc:121
        frame #12: 0x00000001067a1198 libCSG_GGeo.dylib`CSG_GGeo_Convert::Translate(ggeo=0x000000010ca36f60) at CSG_GGeo_Convert.cc:50
        frame #13: 0x000000010011fa14 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010be113a0, gg_=0x000000010ca36f60) at G4CXOpticks.cc:271
        frame #14: 0x000000010011db42 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010be113a0, world=0x000000010be41aa0) at G4CXOpticks.cc:264
        frame #15: 0x000000010011f359 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010be113a0, gdmlpath="/Users/blyth/.opticks/GEOM/simpleLArTPC/origin.gdml") at G4CXOpticks.cc:218
        frame #16: 0x000000010011d4fe libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010be113a0) at G4CXOpticks.cc:198
        frame #17: 0x000000010011c8c0 libG4CX.dylib`G4CXOpticks::SetGeometry() at G4CXOpticks.cc:60
        frame #18: 0x000000010000fa6f G4CXOpticks_setGeometry_Test`main(argc=1, argv=0x00007ffeefbfe728) at G4CXOpticks_setGeometry_Test.cc:16
        frame #19: 0x00007fff56c7c015 libdyld.dylib`start + 1
        frame #20: 0x00007fff56c7c015 libdyld.dylib`start + 1
    (lldb) f 17
    frame #17: 0x000000010011c8c0 libG4CX.dylib`G4CXOpticks::SetGeometry() at G4CXOpticks.cc:60
       57  	G4CXOpticks* G4CXOpticks::SetGeometry()
       58  	{
       59  	    G4CXOpticks* g4cx = new G4CXOpticks ;
    -> 60  	    g4cx->setGeometry(); 
       61  	    return g4cx ; 
       62  	}
       63  	G4CXOpticks* G4CXOpticks::SetGeometry(const G4VPhysicalVolume* world)
    (lldb) f 16
    frame #16: 0x000000010011d4fe libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010be113a0) at G4CXOpticks.cc:198
       195 	    {
       196 	        // may load GDML directly if "${GEOM}_GDMLPathFromGEOM" is defined
       197 	        LOG(LEVEL) << "[ GDMLPathFromGEOM " ; 
    -> 198 	        setGeometry(SOpticksResource::GDMLPathFromGEOM()) ; 
       199 	        LOG(LEVEL) << "] GDMLPathFromGEOM " ; 
       200 	    }
       201 	    else if(ssys::hasenv_("GEOM"))
    (lldb) f 15
    frame #15: 0x000000010011f359 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010be113a0, gdmlpath="/Users/blyth/.opticks/GEOM/simpleLArTPC/origin.gdml") at G4CXOpticks.cc:218
       215 	{
       216 	    LOG(LEVEL) << " gdmlpath [" << gdmlpath << "]" ;
       217 	    const G4VPhysicalVolume* world = U4GDML::Read(gdmlpath);
    -> 218 	    setGeometry(world); 
       219 	}
       220 	
       221 	/**
    (lldb) p gdmlpath
    (const char *) $0 = 0x00007ffeefbffebc "/Users/blyth/.opticks/GEOM/simpleLArTPC/origin.gdml"
    (lldb) f 14
    frame #14: 0x000000010011db42 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010be113a0, world=0x000000010be41aa0) at G4CXOpticks.cc:264
       261 	    Opticks::Configure("--gparts_transform_offset --allownokey" );  
       262 	    GGeo* gg_ = X4Geo::Translate(wd) ; 
       263 	
    -> 264 	    setGeometry(gg_); 
       265 	}
       266 	void G4CXOpticks::setGeometry(GGeo* gg_)
       267 	{
    (lldb) f 13
    frame #13: 0x000000010011fa14 libG4CX.dylib`G4CXOpticks::setGeometry(this=0x000000010be113a0, gg_=0x000000010ca36f60) at G4CXOpticks.cc:271
       268 	    LOG(LEVEL); 
       269 	    gg = gg_ ; 
       270 	
    -> 271 	    CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ; 
       272 	    setGeometry(fd_); 
       273 	}
       274 	
    (lldb) f 12
    frame #12: 0x00000001067a1198 libCSG_GGeo.dylib`CSG_GGeo_Convert::Translate(ggeo=0x000000010ca36f60) at CSG_GGeo_Convert.cc:50
       47  	    CSGFoundry* fd = new CSGFoundry  ; 
       48  	    LOG(LEVEL) << "[ convert ggeo " ; 
       49  	    CSG_GGeo_Convert conv(fd, ggeo ) ; 
    -> 50  	    conv.convert(); 
       51  	
       52  	    bool ops = SSys::getenvbool("ONE_PRIM_SOLID"); 
       53  	    if(ops) conv.addOnePrimSolid(); 
    (lldb) f 11
    frame #11: 0x00000001067a177a libCSG_GGeo.dylib`CSG_GGeo_Convert::convert(this=0x00007ffeefbfc770) at CSG_GGeo_Convert.cc:121
       118 	{
       119 	    LOG(LEVEL) << "[" ; 
       120 	    convertGeometry(); 
    -> 121 	    convertSim(); 
       122 	    LOG(LEVEL) << "]" ; 
       123 	}
       124 	
    (lldb) f 10
    frame #10: 0x00000001067a5259 libCSG_GGeo.dylib`CSG_GGeo_Convert::convertSim(this=0x00007ffeefbfc770) at CSG_GGeo_Convert.cc:201
       198 	
       199 	void CSG_GGeo_Convert::convertSim() 
       200 	{
    -> 201 	    ggeo->convertSim() ; 
       202 	}
       203 	
       204 	
    (lldb) f 9
    frame #9: 0x0000000106a73039 libGGeo.dylib`GGeo::convertSim(this=0x000000010ca36f60) const at GGeo.cc:2548
       2545	
       2546	    convertSim_BndLib(); 
       2547	    convertSim_ScintillatorLib(); 
    -> 2548	    convertSim_Prop(); 
       2549	    //convertSim_MultiFilm(); 
       2550	
       2551	    SSim* sim = SSim::Get();
    (lldb) f 8
    frame #8: 0x0000000106a73ae9 libGGeo.dylib`GGeo::convertSim_Prop(this=0x000000010ca36f60) const at GGeo.cc:2639
       2636	
       2637	void GGeo::convertSim_Prop() const 
       2638	{
    -> 2639	    const NP* propcom = SPropMockup::CombinationDemo();
       2640	    m_fold->add(snam::PROPCOM, propcom); 
       2641	}
       2642	
    (lldb) f 7
    frame #7: 0x0000000106a7465b libGGeo.dylib`SPropMockup::CombinationDemo() at SPropMockup.h:42
       39  	
       40  	inline const NP* SPropMockup::CombinationDemo() // static
       41  	{
    -> 42  	    const NP* propcom = Combination( DEMO_BASE, DEMO_RELP);
       43  	    return propcom ;  
       44  	}
       45  	
    (lldb) p DEMO_BASE
    error: use of undeclared identifier 'DEMO_BASE'
    (lldb) f 6
    frame #6: 0x0000000106a80191 libGGeo.dylib`SPropMockup::Combination(base="$HOME/.opticks/GEOM/$GEOM", relp="GGeo/GScintillatorLib/LS_ori/RINDEX.npy") at SPropMockup.h:71
       68  	        ;
       69  	
       70  	    if( path == nullptr ) return nullptr ;  // malformed path ?
    -> 71  	    NP* a = NP::Load(path) ; 
       72  	    if( a == nullptr ) return nullptr ;  // non-existing path 
       73  	
       74  	    bool is_double = strcmp( a->dtype, "<f8") == 0; 
    (lldb) p path
    (const char *) $1 = 0x000000010bd66200 "/Users/blyth/.opticks/GEOM/simpleLArTPC/GGeo/GScintillatorLib/LS_ori/RINDEX.npy"
    (lldb) f 5
    frame #5: 0x00000001068fc45c libGGeo.dylib`NP::Load(path_="/Users/blyth/.opticks/GEOM/simpleLArTPC/GGeo/GScintillatorLib/LS_ori/RINDEX.npy") at NP.hh:2303
       2300	    NP* a = nullptr ; 
       2301	    if(npy_ext)
       2302	    {
    -> 2303	        a  = NP::Load_(path);
       2304	    }  
       2305	    else
       2306	    {
    (lldb) f 4
    frame #4: 0x00000001068fcbcb libGGeo.dylib`NP::Load_(path="/Users/blyth/.opticks/GEOM/simpleLArTPC/GGeo/GScintillatorLib/LS_ori/RINDEX.npy") at NP.hh:2326
       2323	{
       2324	    if(!path) return nullptr ; 
       2325	    NP* a = new NP() ; 
    -> 2326	    int rc = a->load(path) ; 
       2327	    return rc == 0 ? a  : nullptr ; 
       2328	}
       2329	
    (lldb) f 3
    frame #3: 0x00000001068fe623 libGGeo.dylib`NP::load(this=0x000000010bd665a0, path="/Users/blyth/.opticks/GEOM/simpleLArTPC/GGeo/GScintillatorLib/LS_ori/RINDEX.npy") at NP.hh:4422
       4419	    if(fp.fail())
       4420	    {
       4421	        std::cerr << "NP::load Failed to load from path " << path << std::endl ; 
    -> 4422	        std::raise(SIGINT); 
       4423	        return 1 ; 
       4424	    }
       4425	
    (lldb) 

