csg_complexity_lvid_desc_with_NRT_triangulation_to_ensure_do_not_have_any_analytic_deep_tree_hogs
=====================================================================================================


Maybe enhancing desc_lvid would be sufficient::

    [stree::desc_lvid
    force_triangulate_lvid.size 6
     i    0 lvid    0 is_force_triangulate NO  count_N/R/T      1     1     0 lvn sTopRock_domeAir
     i    1 lvid    1 is_force_triangulate NO  count_N/R/T      1     1     0 lvn sTopRock_dome_plusUpper
     i    2 lvid    2 is_force_triangulate NO  count_N/R/T      1     1     0 lvn sDomeRockBox
     i    3 lvid    3 is_force_triangulate NO  count_N/R/T      1     1     0 lvn PoolCoversub
     i    4 lvid    4 is_force_triangulate NO  count_N/R/T      1     1     0 lvn sElecRoom1
     i    5 lvid    5 is_force_triangulate NO  count_N/R/T      1     1     0 lvn sElecRoom2
     i    6 lvid    6 is_force_triangulate NO  count_N/R/T      1     1     0 lvn Upper_LS_tube
     i    7 lvid    7 is_force_triangulate NO  count_N/R/T      1     1     0 lvn Upper_Steel_tube
     i    8 lvid    8 is_force_triangulate NO  count_N/R/T      1     1     0 lvn Upper_Chimney
     i    9 lvid    9 is_force_triangulate NO  count_N/R/T      3     3     0 lvn pipeA
     i   10 lvid   10 is_force_triangulate NO  count_N/R/T  31744     0     0 lvn sBar_0
     i   11 lvid   11 is_force_triangulate NO  count_N/R/T  31744     0     0 lvn sBar_1
     i   12 lvid   12 is_force_triangulate NO  count_N/R/T    496     0     0 lvn sPanelTape
     i   13 lvid   13 is_force_triangulate NO  count_N/R/T    496     0     0 lvn sPanel
     i   14 lvid   14 is_force_triangulate NO  count_N/R/T    122   122     0 lvn sPlane_0
     i   15 lvid   15 is_force_triangulate NO  count_N/R/T     30    30     0 lvn sWall_0
     i   16 lvid   16 is_force_triangulate NO  count_N/R/T     31    31     0 lvn sWall_1
     i   17 lvid   17 is_force_triangulate NO  count_N/R/T      4     4     0 lvn sPlane_1
     i   18 lvid   18 is_force_triangulate NO  count_N/R/T      2     2     0 lvn sWall_2
     i   19 lvid   19 is_force_triangulate NO  count_N/R/T      1     1     0 lvn sAirTT
     i   20 lvid   20 is_force_triangulate NO  count_N/R/T      1     1     0 lvn sExpHallPlusER2
     i   21 lvid   21 is_force_triangulate NO  count_N/R/T      1     1     0 lvn sExpRockWithMiddle


Actually desc_solids is better::

    [stree::desc_solids num_soname 472
     lvid   0 is_force_triangulate NO  is_auto_triangulate NO  is_triangulate NO  soname[lvid]                                             sTopRock_domeAir sn::rbrief
      0 : sn::brief tc    2 cm  0 lv   0 xf Y pa N bb N pt N nc  2 dp  0 tg in bb.desc -
      1 : sn::brief tc  105 cm  0 lv   0 xf Y pa Y bb Y pt Y nc  0 dp  1 tg cy bb.desc [-26760.000,-26760.000,-28125.000, 26760.000, 26760.000, 28125.000]
      1 : sn::brief tc  110 cm  1 lv   0 xf Y pa Y bb Y pt Y nc  0 dp  1 tg bo bb.desc [-37520.000,-53520.000,-38125.000, 37520.000, 53520.000, 38125.000]

     lvid   1 is_force_triangulate NO  is_auto_triangulate NO  is_triangulate NO  soname[lvid]                                      sTopRock_dome_plusUpper sn::rbrief
      0 : sn::brief tc    1 cm  0 lv   1 xf Y pa N bb N pt N nc  2 dp  0 tg un bb.desc -
      1 : sn::brief tc    2 cm  0 lv   1 xf Y pa N bb N pt Y nc  2 dp  1 tg in bb.desc -
      2 : sn::brief tc  105 cm  0 lv   1 xf Y pa Y bb Y pt Y nc  0 dp  2 tg cy bb.desc [-36760.000,-36760.000,-38125.000, 36760.000, 36760.000, 38125.000]
      2 : sn::brief tc  110 cm  1 lv   1 xf Y pa Y bb Y pt Y nc  0 dp  2 tg bo bb.desc [-47520.000,-73520.000,-39125.000, 47520.000, 73520.000, 39125.000]
      1 : sn::brief tc  110 cm  0 lv   1 xf Y pa Y bb Y pt Y nc  0 dp  1 tg bo bb.desc [ -2750.000,-54550.000,-15000.000,  2750.000, 54550.000, 15000.000]

     lvid   2 is_force_triangulate NO  is_auto_triangulate NO  is_triangulate NO  soname[lvid]                                                 sDomeRockBox sn::rbrief
      0 : sn::brief tc  110 cm  0 lv   2 xf Y pa Y bb Y pt N nc  0 dp  0 tg bo bb.desc [-38125.000,-34500.000,-13000.000, 38125.000, 34500.000, 13000.000]






Review where the trimesh decision happens
-------------------------------------------

::

    0341 bool CSGFoundry::isSolidTrimesh(int gas_idx) const
     342 {
     343     char intent = getSolidIntent(gas_idx);
     344     bool trimesh = intent == 'T' ;
     345     assert( intent == 'R' || intent == 'F' || intent == 'T' || intent == '\0'  );
     346     return trimesh ;
     347 }


    1439 char CSGFoundry::getSolidIntent(int ridx) const
    1440 {
    1441     const CSGSolid* so = getSolid(ridx);
    1442     assert(so);
    1443     return so ? so->getIntent() : '\0' ;
    1444 }


    087 /**
     88 CSGSolid::setIntent
     89 --------------------
     90
     91 Canonical usage is from CSGImport::importSolid which invokes one of two methods for each solid
     92 depending on stree::get_ridx_type::
     93
     94 1. CSGImport::importSolidGlobal
     95
     96    * for ridx_type 'R' analytic global solid (from stree.h "rem" vector of nodes)
     97    * for ridx_type 'T' triangulated global solid (from stree.h "tri" vector of nodes)
     98
     99 2. CSGImport::importSolidFactor
    100
    101    * for ridx_type 'F' analytic factor solid
    102
    103 **/
    104
    105 void CSGSolid::setIntent(char _intent)
    106 {
    107     assert(_intent == 'R' || _intent == 'F' || _intent == 'T' );
    108     intent = _intent ;
    109 }


    094 void CSGImport::importSolid()
     95 {
     96     int num_ridx = st->get_num_ridx() ;
     97     for(int ridx=0 ; ridx < num_ridx ; ridx++)
     98     {
     99         char ridx_type = st->get_ridx_type(ridx) ;
    100         switch(ridx_type)
    101         {
    102             case 'R': importSolidGlobal( ridx, ridx_type ) ; break ;   // remainder
    103             case 'T': importSolidGlobal( ridx, ridx_type ) ; break ;   // triangulate
    104             case 'F': importSolidFactor( ridx, ridx_type ) ; break ;   // factor
    105         }
    106     }
    107 }
    108


::

    5294 /**
    5295 stree::is_triangulate
    5296 ------------------------
    5297 
    5298 How the result of stree::is_triangulate is used is spread over
    5299 the geometry handling of sysrap, CSG and CSGOptiX packages.
    5300 Relevant methods/fields include:
    5301 
    5302 stree::tri
    5303     vector of snode : subset of nds which are triangulated (currently only global, non-instanced nodes)
    5304     (tri vector is populated by stree::factorize/stree::collectGlobalNodes, based on stree::is_triangulate)
    5305 
    5306 stree::rem
    5307     vector of snode : subset of nds which are remainder nodes : left following factorization
    5308     (rem vector is populated by stree::factorize/stree::collectGlobalNodes, based on stree::is_triangulate)
    5309 
    5310 stree::get_ridx_type
    5311     returns 'R' 'F' or 'T' (depending on get_num_remainder (always 1), get_num_factor, get_num_triangulated (0 or 1))
    5312 
    5313     'R' analytic global solid
    5314     'T' triangulated global solid
    5315     'F' analytic factor solid
    5316     ## NB no 'Q' for triangulated factor solid : NOT YET IMPLEMENTED
    5317 
    5318 
    5319 void CSGSolid::setIntent(char _intent)
    5320    invoked by the below importers, where _intent is one of::
    5321 
    5322 CSGImport::importSolid
    5323 CSGImport::importSolidGlobal
    5324 CSGImport::importSolidFactor
    5325 
    5326 CSGFoundry::isSolidTrimesh
    5327    returns true when CSGFoundry::getSolidIntent yields 'T' ... this is
    5328    used extensively by the below CSGOptiX methods to setup the GPU geometry
    5329 
    5330 CSGOptiX/SBT::createGAS
    5331    depending on CSGFoundry::isSolidTrimesh switches between the mesh and analytic buildInput
    5332    passed to SOPTIX_Accel::Create
    5333 
    5334 CSGOptiX/SBT::createHitgroup
    5335 
    5336 
    5337 In order to dump the triangulation status, need to dump the compound CSGSolid
    5338 
    5339 **/
    5340 
    5341 
    5342 inline bool stree::is_triangulate( int lvid ) const
    5343 {
    5344     return is_force_triangulate(lvid) || is_auto_triangulate(lvid);
    5345 }




    5268 /**
    5269 stree::is_auto_triangulate
    5270 ----------------------------
    5271     
    5272 WIP: to reproduce the manual list of lvid also need to judge
    5273 based on the CSG complexity, depth etc.. even with supported
    5274 nodes
    5275 
    5276 **/ 
    5277     
    5278     
    5279 inline bool stree::is_auto_triangulate( int lvid ) const
    5280 {
    5281     const sn* root = sn::GetLVRoot(lvid);
    5282     assert( root );
    5283 
    5284     const char* names = "torus,notsupported,cutcylinder,phicut,halfspace" ;
    5285     const char* NAMES = ssys::getenvvar(stree__is_auto_triangulate_NAMES, names);
    5286     std::vector<int> tcq ;
    5287     CSG::TypeCodeVec(tcq, NAMES, ',');
    5288     int minsubdepth = 0;  // 0:entire tree
    5289     int count = root->typecodes_count(tcq, minsubdepth );
    5290     return count > 0 ;
    5291 }












