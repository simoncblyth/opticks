nmskSolidMaskVirtual_G4Polycone_spurious_line_between_hat_and_head
=====================================================================

Related
---------

* prev :doc:`ct_scan_nmskTailInner`

setup for ct scan
--------------------

::

    gc
    ./mtranslate.sh   # after adding nmskSolidMaskVirtual__U1 to the geomlist 

    c
    ./ct.sh   ## CSGSimtraceTest


issue
--------

* line of spurious : between hat and head : but note also one spurious not on that line 

* zooming in the ct scan corner  however shows that standard coincidence avoidance 
  actually can be used because the lower hat radius is less than the head cylinder radius 

  * HMM : actually NO : that is a mis-interpretation

* comparing ct.sh and x4t.sh using extg4/ct_vs_x4t.sh makes it look like the
  uncoincidence alg appears to have run when it should not done : resulting in the shelf 


::

    S_OFFSET=0,0,-1 ./ct_vs_x4t.sh 



after controlling logging for GeoChain running and PMTSim::GetSolid 
-----------------------------------------------------------------------

::

    epsilon:GeoChain blyth$ ncylinder=INFO ./translate.sh 
    PLOG::EnvLevel adjusting loglevel by envvar   key ncylinder level INFO fallback DEBUG
    2022-09-16 13:57:13.014 INFO  [7484313] [main@76]  geom_ [nmskSolidMaskVirtual] geom (after trim)  [nmskSolidMaskVirtual] 
                                     PMTSim::init instanciations yielded output chars :  cout  21600 cerr      0 : set VERBOSE to see them 
                                                PMTSim::getSolid yielded output chars :  cout   1432 cerr      0 : set VERBOSE to see them 
    GeoChain::convertSolid/X4SolidTree::Draw : original G4VSolid tree [-1] nameprefix nmsksMask_virtual  NODE:1 PRIM:1 UNDEFINED:1 EXCLUDE:0 INCLUDE:0 MIXED:0 Order:IN

    Pol             
    U               
    0               
                    
                    
    0       zdelta  
                    
    194     az1     
    -183    az0     
                    
    GeoChain::convertSolid/X4SolidTree::Draw : original G4VSolid tree
      0 :                   nmsksMask_virtual :                                     : [          ]
    GeoChain::convertSolid/X4SolidTree::Draw : original G4VSolid tree
     ix  0 iy  0 idx  0 tag        Pol zcn          U zdelta      0.000 az0   -183.225 az1    194.050 name nmsksMask_virtual
    2022-09-16 13:57:13.076 INFO  [7484313] [ncylinder::increase_z2@122]  _z2 0 dz 1 new_z2 1
    2022-09-16 13:57:13.076 INFO  [7484313] [ncylinder::increase_z2@122]  _z2 97 dz 1 new_z2 98
    2022-09-16 13:57:13.079 INFO  [7484313] [CSGDraw::draw@57] GeoChain::convertSolid/CSGGeometry::Draw : converted CSGNode tree axis Z type 1 CSG::Name(type) union IsTree 1 width 7 height 2

                                   un                                                         
                                  1                                                           
                                     0.00                                                     
                                    -0.00                                                     
                                                                                              
               un                            co                                               
              2                             3                                                 
                 0.00                        194.05                                           
                -0.00                         97.00                                           
                                                                                              
     cy                  cy                                                                   
    4                   5                                                                     
       1.00               98.00                                                               
    -183.22                0.00                                                               
                                                                                              
                                                                       

::
 
    BP=ncylinder::increase_z2 ncylinder=INFO ./translate.sh dbg

    /Applications/Xcode/Xcode_10_1.app/Contents/Developer/usr/bin/lldb -f GeoChainSolidTest -o "b ncylinder::increase_z2" -o b --
    (lldb) target create "/usr/local/opticks/lib/GeoChainSolidTest"
    Current executable set to '/usr/local/opticks/lib/GeoChainSolidTest' (x86_64).
    (lldb) b ncylinder::increase_z2
    Breakpoint 1: where = libNPY.dylib`ncylinder::increase_z2(float) + 27 at NCylinder.cpp:118, address = 0x00000000002c23cb
    (lldb) b
    Current breakpoints:
    1: name = 'ncylinder::increase_z2', locations = 1
      1.1: where = libNPY.dylib`ncylinder::increase_z2(float) + 27 at NCylinder.cpp:118, address = libNPY.dylib[0x00000000002c23cb], unresolved, hit count = 0 

    (lldb) r
    Process 60718 launched: '/usr/local/opticks/lib/GeoChainSolidTest' (x86_64)
    PLOG::EnvLevel adjusting loglevel by envvar   key ncylinder level INFO fallback DEBUG
    2022-09-16 14:00:13.445 INFO  [7487725] [main@76]  geom_ [nmskSolidMaskVirtual] geom (after trim)  [nmskSolidMaskVirtual] 
                                     PMTSim::init instanciations yielded output chars :  cout  21600 cerr      0 : set VERBOSE to see them 
                                                PMTSim::getSolid yielded output chars :  cout   1432 cerr      0 : set VERBOSE to see them 
    GeoChain::convertSolid/X4SolidTree::Draw : original G4VSolid tree [-1] nameprefix nmsksMask_virtual  NODE:1 PRIM:1 UNDEFINED:1 EXCLUDE:0 INCLUDE:0 MIXED:0 Order:IN

    Pol             
    U               
    0               
                    
                    
    0       zdelta  
                    
    194     az1     
    -183    az0     
                    
    GeoChain::convertSolid/X4SolidTree::Draw : original G4VSolid tree
      0 :                   nmsksMask_virtual :                                     : [          ]
    GeoChain::convertSolid/X4SolidTree::Draw : original G4VSolid tree
     ix  0 iy  0 idx  0 tag        Pol zcn          U zdelta      0.000 az0   -183.225 az1    194.050 name nmsksMask_virtual
    Process 60718 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x000000010128b3cb libNPY.dylib`ncylinder::increase_z2(this=0x00000001086486c0, dz=1) at NCylinder.cpp:118
       115 	// grow the cylinder upwards on upper side (z2) or downwards on down side (z1)
       116 	void  ncylinder::increase_z2(float dz)
       117 	{ 
    -> 118 	    assert( dz >= 0.f) ; 
       119 	    float _z2 = z2(); 
       120 	    float new_z2 = _z2 + dz ; 
       121 	
    Target 0: (GeoChainSolidTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x000000010128b3cb libNPY.dylib`ncylinder::increase_z2(this=0x00000001086486c0, dz=1) at NCylinder.cpp:118
        frame #1: 0x00000001012729c5 libNPY.dylib`NNodeNudger::znudge_union_maxmin(this=0x0000000108649440, coin=0x0000000108649620) at NNodeNudger.cpp:490
        frame #2: 0x0000000101271b20 libNPY.dylib`NNodeNudger::znudge(this=0x0000000108649440, coin=0x0000000108649620) at NNodeNudger.cpp:298
        frame #3: 0x000000010126fa0c libNPY.dylib`NNodeNudger::uncoincide(this=0x0000000108649440) at NNodeNudger.cpp:285
        frame #4: 0x000000010126e11b libNPY.dylib`NNodeNudger::init(this=0x0000000108649440) at NNodeNudger.cpp:92
        frame #5: 0x000000010126db86 libNPY.dylib`NNodeNudger::NNodeNudger(this=0x0000000108649440, root_=0x00000001086484d0, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:66
        frame #6: 0x000000010126e46d libNPY.dylib`NNodeNudger::NNodeNudger(this=0x0000000108649440, root_=0x00000001086484d0, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:64
        frame #7: 0x00000001012ffb91 libNPY.dylib`NCSG::MakeNudger(msg="Adopt root ctor", root=0x00000001086484d0, surface_epsilon=0.00000999999974) at NCSG.cpp:281
        frame #8: 0x00000001012ffce7 libNPY.dylib`NCSG::NCSG(this=0x00000001086491b0, root=0x00000001086484d0) at NCSG.cpp:314
        frame #9: 0x00000001012fef3d libNPY.dylib`NCSG::NCSG(this=0x00000001086491b0, root=0x00000001086484d0) at NCSG.cpp:329
        frame #10: 0x00000001012fed2d libNPY.dylib`NCSG::Adopt(root=0x00000001086484d0, config=0x0000000000000000, soIdx=0, lvIdx=0) at NCSG.cpp:174
        frame #11: 0x00000001002901e3 libExtG4.dylib`X4PhysicalVolume::ConvertSolid_FromRawNode(ok=0x00007ffeefbfe360, lvIdx=0, soIdx=0, solid=0x0000000108642ab0, soname="nmsksMask_virtual", lvname="nmsksMask_virtual", balance_deep_tree=true, raw=0x00000001086484d0) at X4PhysicalVolume.cc:1166
        frame #12: 0x000000010028fc64 libExtG4.dylib`X4PhysicalVolume::ConvertSolid_(ok=0x00007ffeefbfe360, lvIdx=0, soIdx=0, solid=0x0000000108642ab0, soname="nmsksMask_virtual", lvname="nmsksMask_virtual", balance_deep_tree=true) at X4PhysicalVolume.cc:1132
        frame #13: 0x000000010028ed8d libExtG4.dylib`X4PhysicalVolume::ConvertSolid(ok=0x00007ffeefbfe360, lvIdx=0, soIdx=0, solid=0x0000000108642ab0, soname="nmsksMask_virtual", lvname="nmsksMask_virtual") at X4PhysicalVolume.cc:1030
        frame #14: 0x0000000100123d56 libGeoChain.dylib`GeoChain::convertSolid(this=0x00007ffeefbfe320, solid=0x0000000108642ab0, meta="") at GeoChain.cc:81
        frame #15: 0x000000010002045d GeoChainSolidTest`main(argc=1, argv=0x00007ffeefbfe7b8) at GeoChainSolidTest.cc:99
        frame #16: 0x00007fff5c0ac015 libdyld.dylib`start + 1
        frame #17: 0x00007fff5c0ac015 libdyld.dylib`start + 1
    (lldb) 
    

::

     271 NNodeNudger* NCSG::MakeNudger(const char* msg, nnode* root, float surface_epsilon )   // static  
     272 {
     273     int treeidx = root->get_treeidx();
     274     bool nudgeskip = root->is_nudgeskip() ;
     275 
     276     LOG(LEVEL)
     277         << " treeidx " << treeidx
     278         << " nudgeskip " << nudgeskip
     279          ;
     280 
     281     NNodeNudger* nudger = nudgeskip ? nullptr : new NNodeNudger(root, surface_epsilon, root->verbosity);
     282     return nudger ;
     283 }
     284 


::
        
    epsilon:npy blyth$ opticks-f set_nudgeskip 
    ./extg4/X4PhysicalVolume.cc:    raw->set_nudgeskip( is_x4nudgeskip );   
    ./extg4/X4PhysicalVolume.cc:    root->set_nudgeskip( is_x4nudgeskip ); 
    ./npy/NNode.cpp:void nnode::set_nudgeskip(bool nudgeskip_)
    ./npy/NNode.hpp:    void set_nudgeskip(bool nudgeskip_); 
    epsilon:opticks blyth$ 


        
    1093 GMesh* X4PhysicalVolume::ConvertSolid_( const Opticks* ok, int lvIdx, int soIdx, const G4VSolid* const solid, const char* soname, const char* lvname, bool balance_deep_tree ) // static
    1094 {
    1095     assert( lvIdx == soIdx );
    1096     bool dbglv = lvIdx == ok->getDbgLV() ;
    1097 
    1098     bool is_x4balanceskip = ok->isX4BalanceSkip(lvIdx) ;
    1099     if( is_x4balanceskip ) LOG(fatal) << " is_x4balanceskip " << " soIdx " << soIdx  << " lvIdx " << lvIdx ;
    1100 
    1101     bool is_x4nudgeskip = ok->isX4NudgeSkip(lvIdx) ;
    1102     if( is_x4nudgeskip ) LOG(fatal) << " is_x4nudgeskip " << " soIdx " << soIdx  << " lvIdx " << lvIdx ;
    1103 
    1104     bool is_x4pointskip = ok->isX4PointSkip(lvIdx) ;
    1105     if( is_x4pointskip ) LOG(fatal) << " is_x4pointskip " << " soIdx " << soIdx  << " lvIdx " << lvIdx ;
    1106 
    1107 
    1108     LOG(LEVEL)
    1109         << "[ "
    1110         << lvIdx
    1111         << ( dbglv ? " --dbglv " : "" )
    1112         << " soname " << soname
    1113         << " lvname " << lvname
    1114         ;
    1115 
    1116     X4Solid::Banner( lvIdx, soIdx, lvname, soname );
    1117 
    1118     const char* boundary = nullptr ;
    1119     nnode* raw = X4Solid::Convert(solid, ok, boundary, lvIdx )  ;
    1120     raw->set_nudgeskip( is_x4nudgeskip );
    1121     raw->set_pointskip( is_x4pointskip );
    1122     raw->set_treeidx( lvIdx );
    1123 
    1124     // At first glance these settings might look too late to do anything, but that 
    1125     // is not the case as for example the *nudgeskip* setting is acted upon by the NCSG::NCSG(nnode*) cto, 
    1126     // which is invoked from NCSG::Adopt below which branches in NCSG::MakeNudger based on the setting.
    1127 



Capture explorations in CSG/nmskSolidMaskVirtual.sh
--------------------------------------------------------

::

     01 #!/bin/bash -l 
      2 usage(){ cat << EOU
      3 nmskSolidMaskVirtual.sh
      4 =========================
      5 
      6 With the nudging, currently ON by default, see sprinkle along z=98 and nothing along z=0, and clear geometry changed "shoulder"::
      7 
      8     ./nmskSolidMaskVirtual.sh withnudge_ct_ana
      9     ./nmskSolidMaskVirtual.sh ana
     10 
     11 Disabling nudging get very clear coincidence lines at z=97 and z=0::
     12 
     13     ./nmskSolidMaskVirtual.sh skipnudge_ct_ana
     14     ./nmskSolidMaskVirtual.sh ana
     15 
     16 After running eg skipnudge_ct the ana appearance will stay the same 
     17 until running eg withnudge_ct 
     18 
     19 EOU
     20 }

::

    In [7]: w = np.logical_and( np.abs(s.simtrace[:,1,0]) < 220, np.abs(s.simtrace[:,1,2]-98) < 1 )
    In [8]: np.count_nonzero(w)
    Out[8]: 18


Looks like the spurious sprinkle have rays that when extended go close to the cone apex::

    NOLEGEND=1 XDIST=400 XX=0 ./nmskSolidMaskVirtual.sh unx


Looking at *intersect_leaf_cone* note that its not using *robust_quadratic_roots*.

::

    z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex




