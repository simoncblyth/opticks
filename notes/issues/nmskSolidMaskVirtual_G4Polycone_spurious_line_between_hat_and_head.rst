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


Almost all the spurious cross x=0 close to the apex. 
The one that doesnt is very nearly shooting up the axis starting from origin.

Using CSGSimtraceSample.sh with simtrace array of the spurious::

     o.x           -237.6450 v.x              0.5402 t0(-o.x/v.x)   439.9518 z0             291.0315
     o.x           -211.2400 v.x              0.5538 t0(-o.x/v.x)   381.4507 z0             291.2147
     o.x           -211.2400 v.x              0.7059 t0(-o.x/v.x)   299.2489 z0             291.1763
     o.x           -132.0250 v.x              0.2816 t0(-o.x/v.x)   468.8077 z0             291.4034
     o.x           -105.6200 v.x             -0.5532 t0(-o.x/v.x)  -190.9166 z0             291.0645
     o.x            -52.8100 v.x              0.1320 t0(-o.x/v.x)   399.9243 z0             290.8022
     o.x            -26.4050 v.x             -0.1410 t0(-o.x/v.x)  -187.2420 z0             290.9908
     o.x            -26.4050 v.x             -0.1643 t0(-o.x/v.x)  -160.7213 z0             290.5624
     o.x            -26.4050 v.x             -0.3141 t0(-o.x/v.x)   -84.0729 z0             291.0587
     o.x              0.0000 v.x             -0.0004 t0(-o.x/v.x)     0.0000 z0               0.0000
     o.x             26.4050 v.x             -0.0621 t0(-o.x/v.x)   425.1504 z0             292.3047
     o.x             52.8100 v.x             -0.1317 t0(-o.x/v.x)   401.0959 z0             291.9841
     o.x             79.2150 v.x             -0.2091 t0(-o.x/v.x)   378.8108 z0             291.2207
     o.x             79.2150 v.x             -0.2866 t0(-o.x/v.x)   276.3970 z0             291.2073
     o.x            132.0250 v.x             -0.3161 t0(-o.x/v.x)   417.7253 z0             290.6928
     o.x            184.8350 v.x             -0.4006 t0(-o.x/v.x)   461.4159 z0             290.7523
     o.x            237.6450 v.x             -0.5399 t0(-o.x/v.x)   440.1857 z0             291.3093
     o.x            264.0500 v.x             -0.6088 t0(-o.x/v.x)   433.7131 z0             291.2607
    2022-09-16 20:40:05.854 INFO  [8515964] [CSGSimtraceSample::intersect@92] CSGSimtraceSample::desc
     fd Y
     fd.geom -
     CSGQuery::Label  not-DEBUG not-DEBUG_RECORD DEBUG_CYLINDER
     path /tmp/simtrace_sample.npy
     simtrace (18, 4, 4, )
     n 18 num_intersect 18
    epsilon:CSG blyth$ ./nmskSolidMaskVirtual.sh sample





Slight variation in direction makes the rays miss : when they should hit the endcap
-------------------------------------------------------------------------------------

Before changing CSG_NEWCONE alg::

    epsilon:tests blyth$ ./intersect_leaf_cone_test.sh
    // r1   100.0000 z1  -100.0000  r2    50.0000 z2   -50.0000 apex z0     0.0000 
    //intersect_leaf_cone r1   100.0000 z1  -100.0000 r2    50.0000 z2   -50.0000 : z0     0.0000 
    //intersect_leaf_cone c2    -0.6000 c1   134.1641 c0 -30000.0000 disc     0.0020 disc > 0.f 1 : tth    -1.0000 
    //intersect_leaf_newcone r1   100.0000 z1  -100.0000 r2    50.0000 z2   -50.0000 : z0     0.0000 
    //intersect_leaf_newcone c2    -0.6000 c1   134.1641 c0 -30000.0000 disc     0.0020 disc > 0.f 1 : tth    -1.0000 
    // ray ( -100.0000     0.0000  -200.0000 ;     0.4472     0.0000     0.8944 ;     0.0000)
    // vi0 1 i0 (    0.0000     0.0000    -1.0000   111.8034)  p0 (  -50.0000     0.0000  -100.0000)
    // vi1 1 i1 (    0.0000     0.0000    -1.0000   111.8034)  p1 (  -50.0000     0.0000  -100.0000)

    epsilon:tests blyth$ RAYDIR=1,0,2.0001 ./intersect_leaf_cone_test.sh run
    // r1   100.0000 z1  -100.0000  r2    50.0000 z2   -50.0000 apex z0     0.0000 
    //intersect_leaf_cone r1   100.0000 z1  -100.0000 r2    50.0000 z2   -50.0000 : z0     0.0000 
    //intersect_leaf_cone c2    -0.6000 c1   134.1676 c0 -30000.0000 disc    -0.0020 disc > 0.f 0 : tth    -1.0000 
    //intersect_leaf_newcone r1   100.0000 z1  -100.0000 r2    50.0000 z2   -50.0000 : z0     0.0000 
    //intersect_leaf_newcone c2    -0.6000 c1   134.1676 c0 -30000.0000 disc    -0.0020 disc > 0.f 0 : tth    -1.0000 
    // ray ( -100.0000     0.0000  -200.0000 ;     0.4472     0.0000     0.8944 ;     0.0000)
    // vi0 0 i0 (    0.0000     0.0000     0.0000     0.0000)  p0 (    0.0000     0.0000     0.0000)
    // vi1 0 i1 (    0.0000     0.0000     0.0000     0.0000)  p1 (    0.0000     0.0000     0.0000)
    epsilon:tests blyth$ 


After adjusting the logic, treating cap itersects as independent of infinite cone ones::

    epsilon:tests blyth$ ./intersect_leaf_cone_test.sh
    // r1   100.0000 z1  -100.0000  r2    50.0000 z2   -50.0000 apex z0     0.0000 
    //intersect_leaf_cone r1   100.0000 z1  -100.0000 r2    50.0000 z2   -50.0000 : z0     0.0000 
    //intersect_leaf_cone c2    -0.6000 c1   134.1641 c0 -30000.0000 disc     0.0020 disc > 0.f 1 : tth    -1.0000 
    //intersect_leaf_newcone r1   100.0000 z1  -100.0000 r2    50.0000 z2   -50.0000 : z0     0.0000 
    //intersect_leaf_newcone c2    -0.6000 c1   134.1641 c0 -30000.0000 disc     0.0020 disc > 0.f 1 : tth    -1.0000 
    // ray ( -100.0000     0.0000  -200.0000 ;     0.4472     0.0000     0.8944 ;     0.0000)
    // vi0 1 i0 (    0.0000     0.0000    -1.0000   111.8034)  p0 (  -50.0000     0.0000  -100.0000)
    // vi1 1 i1 (    0.0000     0.0000    -1.0000   111.8034)  p1 (  -50.0000     0.0000  -100.0000)
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 
    epsilon:tests blyth$ RAYDIR=1,0,2.0001 ./intersect_leaf_cone_test.sh run
    // r1   100.0000 z1  -100.0000  r2    50.0000 z2   -50.0000 apex z0     0.0000 
    //intersect_leaf_cone r1   100.0000 z1  -100.0000 r2    50.0000 z2   -50.0000 : z0     0.0000 
    //intersect_leaf_cone c2    -0.6000 c1   134.1676 c0 -30000.0000 disc    -0.0020 disc > 0.f 0 : tth    -1.0000 
    //intersect_leaf_newcone r1   100.0000 z1  -100.0000 r2    50.0000 z2   -50.0000 : z0     0.0000 
    //intersect_leaf_newcone c2    -0.6000 c1   134.1676 c0 -30000.0000 disc    -0.0020 disc > 0.f 0 : tth    -1.0000 
    // ray ( -100.0000     0.0000  -200.0000 ;     0.4472     0.0000     0.8944 ;     0.0000)
    // vi0 0 i0 (    0.0000     0.0000     0.0000     0.0000)  p0 (    0.0000     0.0000     0.0000)
    // vi1 1 i1 (    0.0000     0.0000    -1.0000   111.8023)  p1 (  -50.0025     0.0000  -100.0000)
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 



Investigate a spurious that is not on the cylinder cone line
---------------------------------------------------------------


* below shows the cause to be catastrophic precision loss in t_near
* this is fixed in newcone using *robust_quadratic_roots*


::

    In [1]: w = np.logical_and( np.abs(s.simtrace[:,1,0] - (-214)) < 5, np.abs(s.simtrace[:,1,2] - (115)) < 5 )

    In [3]: np.where(w)
    Out[3]: (array([421273]),)

    In [5]: s.simtrace[w,:,:3]
    Out[5]: 
    array([[[  -0.546,    0.   ,    0.838],
            [-212.85 ,    0.   ,  114.494],
            [ 158.43 ,    0.   , -158.43 ],
            [  -0.806,    0.   ,    0.592]]], dtype=float32)




::

    epsilon:CSG blyth$ ./nmskSolidMaskVirtual.sh sample
    /Users/blyth/opticks/CSG
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                               gp_ : nmskSolidMaskVirtual_GDMLPath 
                                gp :  
                               cg_ : nmskSolidMaskVirtual_CFBaseFromGEOM 
                                cg : /tmp/blyth/opticks/GEOM/nmskSolidMaskVirtual 
                       TMP_GEOMDIR : /tmp/blyth/opticks/GEOM/nmskSolidMaskVirtual 
                           GEOMDIR : /tmp/blyth/opticks/GEOM/nmskSolidMaskVirtual 

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
    PLOG::EnvLevel adjusting loglevel by envvar   key SArgs level INFO fallback DEBUG
    PLOG::EnvLevel adjusting loglevel by envvar   key CSGFoundry level INFO fallback DEBUG
    PLOG::EnvLevel adjusting loglevel by envvar   key CSGSimtraceSample level INFO fallback DEBUG
    PLOG::EnvLevel adjusting loglevel by envvar   key ncylinder level INFO fallback DEBUG
    2022-09-17 16:18:02.592 INFO  [9048710] [*CSGFoundry::Load@2487] [ argumentless 
    2022-09-17 16:18:02.593 INFO  [9048710] [*CSGFoundry::ResolveCFBase@2550]  cfbase /tmp/blyth/opticks/GEOM/nmskSolidMaskVirtual readable 1
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::setMeta@138]                      : -
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::setMeta@138]                 HOME : /Users/blyth
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::setMeta@138]                 USER : blyth
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::setMeta@138]               SCRIPT : -
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::setMeta@138]                  PWD : /Users/blyth/opticks/CSG
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::setMeta@138]              CMDLINE : -
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::load@2279] [ loaddir /tmp/blyth/opticks/GEOM/nmskSolidMaskVirtual/CSGFoundry
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::loadArray@2648]  ni     1 nj 3 nk 4 solid.npy
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::loadArray@2648]  ni     1 nj 4 nk 4 prim.npy
    2022-09-17 16:18:02.593 INFO  [9048710] [CSGFoundry::loadArray@2648]  ni     7 nj 4 nk 4 node.npy
    2022-09-17 16:18:02.594 INFO  [9048710] [CSGFoundry::loadArray@2648]  ni     3 nj 4 nk 4 tran.npy
    2022-09-17 16:18:02.594 INFO  [9048710] [CSGFoundry::loadArray@2648]  ni     3 nj 4 nk 4 itra.npy
    2022-09-17 16:18:02.594 INFO  [9048710] [CSGFoundry::loadArray@2648]  ni     1 nj 4 nk 4 inst.npy
    2022-09-17 16:18:02.594 INFO  [9048710] [CSGFoundry::load@2303] [ SSim::Load 
    2022-09-17 16:18:02.594 INFO  [9048710] [CSGFoundry::load@2305] ] SSim::Load 
    2022-09-17 16:18:02.594 INFO  [9048710] [CSGFoundry::load@2310] ] loaddir /tmp/blyth/opticks/GEOM/nmskSolidMaskVirtual/CSGFoundry
    2022-09-17 16:18:02.594 INFO  [9048710] [*CSGFoundry::ELVString@2446]  elv_selection_ (null) elv (null)
    2022-09-17 16:18:02.594 INFO  [9048710] [*CSGFoundry::Load@2502] ] argumentless 
    2022-09-17 16:18:02.594 INFO  [9048710] [CSGSimtraceSample::init@37]  axis Z type 1 CSG::Name(type) union IsTree 1 width 7 height 2
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
                                                                                              
                                                                                              
                                                                                              
                                                                                              
                                                                                              
                                                                                              


    2022-09-17 16:18:02.594 INFO  [9048710] [CSGSimtraceSample::init@38]  fd.cfbase /tmp/blyth/opticks/GEOM/nmskSolidMaskVirtual
    2022-09-17 16:18:02.594 INFO  [9048710] [CSGSimtraceSample::init@39]  vv (10, )
    2022-09-17 16:18:02.595 INFO  [9048710] [CSGSimtraceSample::init@40] vv.lpath [/tmp/blyth/opticks/GEOM/nmskSolidMaskVirtual/Values/values.npy]
    2022-09-17 16:18:02.595 INFO  [9048710] [CSGSimtraceSample::init@41] vv.descValues 
    NP::descValues num_val 10
      0 v  -183.2246 k SolidMask.SolidMaskVirtual.zPlane0.-height_virtual          
      1 v     0.0000 k SolidMask.SolidMaskVirtual.zPlane1.0                        
      2 v    97.0000 k SolidMask.SolidMaskVirtual.zPlane2.htop_out/2               
      3 v   194.0500 k SolidMask.SolidMaskVirtual.zPlane3.htop_out+MAGIC_virtual_thickness
      4 v   264.0500 k SolidMask.SolidMaskVirtual.rOuter0.mask_radiu_virtual       
      5 v   264.0500 k SolidMask.SolidMaskVirtual.rOuter1.mask_radiu_virtual       
      6 v   264.0500 k SolidMask.SolidMaskVirtual.rOuter2.mask_radiu_virtual       
      7 v   132.0250 k SolidMask.SolidMaskVirtual.rOuter3.mask_radiu_virtual/2     
      8 v     0.0500 k SolidMask.SolidMaskVirtual.MAGIC_virtual_thickness          
      9 v   194.0000 k SolidMask.SolidMaskVirtual.htop_out                         

    //intersect_leaf_oldcone r1   264.0500 z1    97.0000 r2   132.0250 z2   194.0500 : z0   291.1000 
    //intersect_leaf_oldcone c2    -0.0000 c1   365.0782 c0 -348871.4688 disc 133281.9219 disc > 0.f 1 : tth    -1.3604 
    //intersect_leaf_oldcone c0 -3.489e+05 c1      365.1 c2  -5.96e-07 t_near      460.8 t_far  1.225e+09 sdisc   365.0780 (-c1-sdisc)     -730.2 (-c1+sdisc) -0.0002747 
    //intersect_leaf_oldcone t_near_alt      477.8 t_far_alt  1.225e+09 t_near_alt-t_near         17 t_far_alt-t_far          0 
    //intersect_leaf_oldcone r1   264.0500 z1    97.0000 r2   132.0250 z2   194.0500 : z0   291.1000 
    //intersect_leaf_oldcone c2    -0.0000 c1   365.0782 c0 -348871.4688 disc 133281.9219 disc > 0.f 1 : tth    -1.3604 
    //intersect_leaf_oldcone c0 -3.489e+05 c1      365.1 c2  -5.96e-07 t_near      460.8 t_far  1.225e+09 sdisc   365.0780 (-c1-sdisc)     -730.2 (-c1+sdisc) -0.0002747 
    //intersect_leaf_oldcone t_near_alt      477.8 t_far_alt  1.225e+09 t_near_alt-t_near         17 t_far_alt-t_far          0 
                                 - HIT
                        q0 norm t (   -0.5457    0.0000    0.8380  460.8000)
                       q1 ipos sd ( -212.8504    0.0000  114.4940    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (  158.4300    0.0000 -158.4300)
                  q3 ray_dir gsid (   -0.8057    0.0000    0.5923 C4U (     0    0    0  255 ) )

     o.x            158.4300 v.x             -0.8057 t0(-o.x/v.x)   196.6291 z0             -41.9699
    2022-09-17 16:18:02.595 INFO  [9048710] [CSGSimtraceSample::intersect@89] CSGSimtraceSample::desc
     fd Y
     fd.geom -
     CSGQuery::Label  not-DEBUG not-DEBUG_RECORD not-DEBUG_CYLINDER
     path /tmp/simtrace_sample.npy
     simtrace (1, 4, 4, )
     n 1 num_intersect 1
    epsilon:CSG blyth$ 



