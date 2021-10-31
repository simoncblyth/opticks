deep-tree-balancing-unexpected-un-ze
=======================================

Reproduce
------------

When not using the simplify csg in the standalone PMTSim a height 7 tree 
occurs including a torus. 

Despite this having no chance of working with the torus also 
find a problem in tree balancing, somehow a "ze" placeholder
node gets added to the tree resulting in a nnode::get_primitive_bbox assert.

Root cause is the balancing unexpectedly adding a ze primitive.  


To reproduce::

    cd $HOME/j/PMTSim
    om                        # build and install PMTSim 

    cd ~/opticks/GeoChain
    om                        # CMake finds and links against PMTSim 

    GEOCHAINTEST=PMTSim_etc ./run.sh 


Issue
----------


::

    2021-10-31 14:01:20.142 INFO  [11636647] [*NTreeProcess<nnode>::Process@81] before
    NTreeAnalyse height 7 count 17
                                                                  un    

                                                          un          cy

                                                  un          cy        

                                          un          zs                

                          un                  cy                        

                  un              di                                    

          un          zs      cy      to                                

      zs      cy                                                        


    inorder (left-to-right) 
     [ 0:zs] P PMTSim_etc_I_ellipsoid 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_II_difference 
     [ 0:un] C un 
     [ 0:zs] P PMTSim_etc_III_ellipsoid 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_IV_tube_difference 
     [ 0:di] C di 
     [ 0:to] P PMTSim_etc_IV_torus_torus 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_V_difference 
     [ 0:un] C un 
     [ 0:zs] P PMTSim_etc_VI_ellipsoid 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_VIII_difference 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_IX0_zp_cylinder 


    2021-10-31 14:01:20.142 INFO  [11636647] [*NTreeProcess<nnode>::Process@96] after
    NTreeAnalyse height 5 count 19
                                                                          un    

                                  un                                          ze

                  un                                      un                    

          un              un              un                      in            

      zs      cy      zs      cy      cy          un          cy     !to        

                                              zs      cy                        


    inorder (left-to-right) 
     [ 0:zs] P PMTSim_etc_III_ellipsoid 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_V_difference 
     [ 0:un] C un 
     [ 0:zs] P PMTSim_etc_VI_ellipsoid 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_VIII_difference 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_IX0_zp_cylinder 
     [ 0:un] C un 
     [ 0:zs] P PMTSim_etc_I_ellipsoid 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_II_difference 
     [ 0:un] C un 
     [ 0:cy] P PMTSim_etc_IV_tube_difference 
     [ 0:in] C di 
    [ 0:!to] P PMTSim_etc_IV_torus_torus 
     [ 0:un] C un 
     [ 0:ze] P ze 


    2021-10-31 14:01:20.143 INFO  [11636647] [*NTreeProcess<nnode>::Process@97]  soIdx 0 lvIdx 0 height0 7 height1 5 ### LISTED
    2021-10-31 14:01:20.143 INFO  [11636647] [NNodeNudger::init@72]  init 
    2021-10-31 14:01:20.143 FATAL [11636647] [nnode::get_primitive_bbox@1096] Need to add upcasting for type: 0 name zero
    Assertion failed: (0), function get_primitive_bbox, file /Users/blyth/opticks/npy/NNode.cpp, line 1097.
    Process 21095 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff71632b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff71632b66 <+10>: jae    0x7fff71632b70            ; <+20>
        0x7fff71632b68 <+12>: movq   %rax, %rdi
        0x7fff71632b6b <+15>: jmp    0x7fff71629ae9            ; cerror_nocancel
        0x7fff71632b70 <+20>: retq   
    Target 0: (GeoChainTest) stopped.

    Process 21095 launched: '/usr/local/opticks/lib/GeoChainTest' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff71632b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff717fd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7158e1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff715561ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000100c0f49c libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010830b3d0, bb=0x00007ffeefbfc118) const at NNode.cpp:1097
        frame #5: 0x0000000100c0f840 libNPY.dylib`nnode::bbox(this=0x000000010830b3d0) const at NNode.cpp:1135
        frame #6: 0x0000000100c4fdb6 libNPY.dylib`NNodeNudger::update_prim_bb(this=0x000000010830b760) at NNodeNudger.cpp:114
        frame #7: 0x0000000100c4f8a6 libNPY.dylib`NNodeNudger::init(this=0x000000010830b760) at NNodeNudger.cpp:79
        frame #8: 0x0000000100c4f627 libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000010830b760, root_=0x000000010830a050, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:67
        frame #9: 0x0000000100c4fc2d libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000010830b760, root_=0x000000010830a050, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:65
        frame #10: 0x0000000100cb2c1d libNPY.dylib`NCSG::MakeNudger(msg="Adopt root ctor", root=0x000000010830a050, surface_epsilon=0.00000999999974) at NCSG.cpp:300
        frame #11: 0x0000000100cb2daa libNPY.dylib`NCSG::NCSG(this=0x000000010830b6b0, root=0x000000010830a050) at NCSG.cpp:331
        frame #12: 0x0000000100cb1d3d libNPY.dylib`NCSG::NCSG(this=0x000000010830b6b0, root=0x000000010830a050) at NCSG.cpp:345
        frame #13: 0x0000000100cb1c2c libNPY.dylib`NCSG::Adopt(root=0x000000010830a050, config=0x0000000000000000, soIdx=0, lvIdx=0) at NCSG.cpp:181
        frame #14: 0x00000001001b2d56 libExtG4.dylib`X4PhysicalVolume::ConvertSolid_(ok=0x00007ffeefbfe3f0, lvIdx=0, soIdx=0, solid=0x0000000108502840, lvname="PMTSim_etc_1_9", balance_deep_tree=true) at X4PhysicalVolume.cc:1130
        frame #15: 0x00000001001b1db1 libExtG4.dylib`X4PhysicalVolume::ConvertSolid(ok=0x00007ffeefbfe3f0, lvIdx=0, soIdx=0, solid=0x0000000108502840, lvname="PMTSim_etc_1_9") at X4PhysicalVolume.cc:1033
        frame #16: 0x00000001000d44b8 libGeoChain.dylib`GeoChain::convert(this=0x00007ffeefbfe3c0, solid=0x0000000108502840) at GeoChain.cc:38
        frame #17: 0x000000010000645e GeoChainTest`main(argc=3, argv=0x00007ffeefbfe798) at GeoChainTest.cc:147
        frame #18: 0x00007fff714e2015 libdyld.dylib`start + 1
        frame #19: 0x00007fff714e2015 libdyld.dylib`start + 1
    (lldb) f 17
    frame #17: 0x000000010000645e GeoChainTest`main(argc=3, argv=0x00007ffeefbfe798) at GeoChainTest.cc:147
       144 	    for(int lvIdx=-1 ; lvIdx < 10 ; lvIdx+= 1 ) LOG(info) << " lvIdx " << lvIdx << " ok.isX4TubsNudgeSkip(lvIdx) " << ok.isX4TubsNudgeSkip(lvIdx)  ; 
       145 	
       146 	    GeoChain chain(&ok); 
    -> 147 	    chain.convert(solid);  
       148 	    chain.save(name); 
       149 	
       150 	    return 0 ; 
    (lldb) f 16
    frame #16: 0x00000001000d44b8 libGeoChain.dylib`GeoChain::convert(this=0x00007ffeefbfe3c0, solid=0x0000000108502840) at GeoChain.cc:38
       35  	    int soIdx = 0 ; 
       36  	    std::string lvname = solid->GetName(); 
       37  	
    -> 38  	    mesh = X4PhysicalVolume::ConvertSolid(ok, lvIdx, soIdx, solid, lvname ) ; 
       39  	    LOG(info) << " mesh " << mesh ; 
       40  	
       41  	    ggeo->add(mesh); 
    (lldb) f 15
    frame #15: 0x00000001001b1db1 libExtG4.dylib`X4PhysicalVolume::ConvertSolid(ok=0x00007ffeefbfe3f0, lvIdx=0, soIdx=0, solid=0x0000000108502840, lvname="PMTSim_etc_1_9") at X4PhysicalVolume.cc:1033
       1030	GMesh* X4PhysicalVolume::ConvertSolid( const Opticks* ok, int lvIdx, int soIdx, const G4VSolid* const solid, const std::string& lvname ) // static
       1031	{
       1032	    bool balance_deep_tree = true ;  
    -> 1033	    GMesh* mesh = ConvertSolid_( ok, lvIdx, soIdx, solid, lvname, balance_deep_tree ) ;  
       1034	
       1035	    mesh->setIndex( lvIdx ) ;   
       1036	
    (lldb) f 14
    frame #14: 0x00000001001b2d56 libExtG4.dylib`X4PhysicalVolume::ConvertSolid_(ok=0x00007ffeefbfe3f0, lvIdx=0, soIdx=0, solid=0x0000000108502840, lvname="PMTSim_etc_1_9", balance_deep_tree=true) at X4PhysicalVolume.cc:1130
       1127	     root->set_treeidx( lvIdx ); 
       1128	
       1129	     const NSceneConfig* config = NULL ; 
    -> 1130	     NCSG* csg = NCSG::Adopt( root, config, soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instance
       1131	     assert( csg ) ; 
       1132	     assert( csg->isUsedGlobally() );
       1133	     csg->set_soname( soname.c_str() ) ; 
    (lldb) f 13
    frame #13: 0x0000000100cb1c2c libNPY.dylib`NCSG::Adopt(root=0x000000010830a050, config=0x0000000000000000, soIdx=0, lvIdx=0) at NCSG.cpp:181
       178 	
       179 	    root->set_treeidx(lvIdx) ;  // without this no nudging is done
       180 	
    -> 181 	    NCSG* tree = new NCSG(root);
       182 	
       183 	    tree->setConfig(config);
       184 	    tree->setSOIdx(soIdx); 
    (lldb) f 12
    frame #12: 0x0000000100cb1d3d libNPY.dylib`NCSG::NCSG(this=0x000000010830b6b0, root=0x000000010830a050) at NCSG.cpp:345
       342 	    m_soIdx(0),
       343 	    m_lvIdx(0),
       344 	    m_other(NULL)
    -> 345 	{
       346 	    setBoundary( root->boundary );  // boundary spec
       347 	    LOG(debug) << "[" ; 
       348 	    m_csgdata->init_buffers(root->maxdepth()) ;  
    (lldb) f 11
    frame #11: 0x0000000100cb2daa libNPY.dylib`NCSG::NCSG(this=0x000000010830b6b0, root=0x000000010830a050) at NCSG.cpp:331
       328 	    m_root(root),
       329 	    m_points(NULL),
       330 	    m_uncoincide(make_uncoincide()),
    -> 331 	    m_nudger(MakeNudger("Adopt root ctor", root, SURFACE_EPSILON)),
       332 	    m_csgdata(new NCSGData),
       333 	    m_adopted(true), 
       334 	    m_boundary(NULL),
    (lldb) f 10
    frame #10: 0x0000000100cb2c1d libNPY.dylib`NCSG::MakeNudger(msg="Adopt root ctor", root=0x000000010830a050, surface_epsilon=0.00000999999974) at NCSG.cpp:300
       297 	        << " nudgeskip " << nudgeskip 
       298 	         ; 
       299 	
    -> 300 	    NNodeNudger* nudger = nudgeskip ? nullptr : new NNodeNudger(root, surface_epsilon, root->verbosity);
       301 	    return nudger ; 
       302 	}
       303 	
    (lldb) f 9
    frame #9: 0x0000000100c4fc2d libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000010830b760, root_=0x000000010830a050, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:65
       62  	    verbosity(SSys::getenvint("VERBOSITY",1)),
       63  	    listed(false),
       64  	    enabled(true)
    -> 65  	{
       66  	    root->check_tree( FEATURE_GTRANSFORMS | FEATURE_PARENT_LINKS );
       67  	    init();
       68  	}
    (lldb) f 8
    frame #8: 0x0000000100c4f627 libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000010830b760, root_=0x000000010830a050, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:67
       64  	    enabled(true)
       65  	{
       66  	    root->check_tree( FEATURE_GTRANSFORMS | FEATURE_PARENT_LINKS );
    -> 67  	    init();
       68  	}
       69  	
       70  	void NNodeNudger::init()
    (lldb) f 7
    frame #7: 0x0000000100c4f8a6 libNPY.dylib`NNodeNudger::init(this=0x000000010830b760) at NNodeNudger.cpp:79
       76  	    if( NudgeBuffer == NULL ) NudgeBuffer = NPY<unsigned>::make(0,4) ; 
       77  	
       78  	    root->collect_prim_for_edit(prim);  // recursive collector 
    -> 79  	    update_prim_bb();                   // find z-order of prim using bb.min.z
       80  	    collect_coincidence();
       81  	
       82  	    if(enabled)
    (lldb) f 6
    frame #6: 0x0000000100c4fdb6 libNPY.dylib`NNodeNudger::update_prim_bb(this=0x000000010830b760) at NNodeNudger.cpp:114
       111 	    {
       112 	        const nnode* p = prim[i] ; 
       113 	
    -> 114 	        nbbox pbb = p->bbox(); 
       115 	        bb.push_back(pbb);
       116 	        zorder.push_back(i);
       117 	    }
    (lldb) f 5
    frame #5: 0x0000000100c0f840 libNPY.dylib`nnode::bbox(this=0x000000010830b3d0) const at NNode.cpp:1135
       1132	
       1133	    if(is_primitive())
       1134	    {
    -> 1135	        get_primitive_bbox(bb);
       1136	    } 
       1137	    else 
       1138	    {
    (lldb) f 4
    frame #4: 0x0000000100c0f49c libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010830b3d0, bb=0x00007ffeefbfc118) const at NNode.cpp:1097
       1094	    else
       1095	    {
       1096	        LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSG::Name(node->type) ;  
    -> 1097	        assert(0);
       1098	    }
       1099	}
       1100	
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff71632b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff717fd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7158e1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff715561ac libsystem_c.dylib`__assert_rtn + 320
      * frame #4: 0x0000000100c0f49c libNPY.dylib`nnode::get_primitive_bbox(this=0x000000010830b3d0, bb=0x00007ffeefbfc118) const at NNode.cpp:1097
        frame #5: 0x0000000100c0f840 libNPY.dylib`nnode::bbox(this=0x000000010830b3d0) const at NNode.cpp:1135
        frame #6: 0x0000000100c4fdb6 libNPY.dylib`NNodeNudger::update_prim_bb(this=0x000000010830b760) at NNodeNudger.cpp:114
        frame #7: 0x0000000100c4f8a6 libNPY.dylib`NNodeNudger::init(this=0x000000010830b760) at NNodeNudger.cpp:79
        frame #8: 0x0000000100c4f627 libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000010830b760, root_=0x000000010830a050, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:67
        frame #9: 0x0000000100c4fc2d libNPY.dylib`NNodeNudger::NNodeNudger(this=0x000000010830b760, root_=0x000000010830a050, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:65
        frame #10: 0x0000000100cb2c1d libNPY.dylib`NCSG::MakeNudger(msg="Adopt root ctor", root=0x000000010830a050, surface_epsilon=0.00000999999974) at NCSG.cpp:300
        frame #11: 0x0000000100cb2daa libNPY.dylib`NCSG::NCSG(this=0x000000010830b6b0, root=0x000000010830a050) at NCSG.cpp:331
        frame #12: 0x0000000100cb1d3d libNPY.dylib`NCSG::NCSG(this=0x000000010830b6b0, root=0x000000010830a050) at NCSG.cpp:345
        frame #13: 0x0000000100cb1c2c libNPY.dylib`NCSG::Adopt(root=0x000000010830a050, config=0x0000000000000000, soIdx=0, lvIdx=0) at NCSG.cpp:181
        frame #14: 0x00000001001b2d56 libExtG4.dylib`X4PhysicalVolume::ConvertSolid_(ok=0x00007ffeefbfe3f0, lvIdx=0, soIdx=0, solid=0x0000000108502840, lvname="PMTSim_etc_1_9", balance_deep_tree=true) at X4PhysicalVolume.cc:1130
        frame #15: 0x00000001001b1db1 libExtG4.dylib`X4PhysicalVolume::ConvertSolid(ok=0x00007ffeefbfe3f0, lvIdx=0, soIdx=0, solid=0x0000000108502840, lvname="PMTSim_etc_1_9") at X4PhysicalVolume.cc:1033
        frame #16: 0x00000001000d44b8 libGeoChain.dylib`GeoChain::convert(this=0x00007ffeefbfe3c0, solid=0x0000000108502840) at GeoChain.cc:38
        frame #17: 0x000000010000645e GeoChainTest`main(argc=3, argv=0x00007ffeefbfe798) at GeoChainTest.cc:147
        frame #18: 0x00007fff714e2015 libdyld.dylib`start + 1
        frame #19: 0x00007fff714e2015 libdyld.dylib`start + 1
    (lldb) 

