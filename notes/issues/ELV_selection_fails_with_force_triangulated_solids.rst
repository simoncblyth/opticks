ELV_selection_fails_with_force_triangulated_solids
=====================================================

Working in this in::

    modified:   CSG/CSGCopy.cc
    modified:   CSG/CSGFoundry.cc
    modified:   sysrap/SMesh.h
    modified:   sysrap/SMeshGroup.h
    modified:   sysrap/SOPTIX_Accel.h
    modified:   sysrap/SOPTIX_BuildInput.h
    modified:   sysrap/SScene.h
    modified:   sysrap/strid.h
    modified:   u4/U4Mesh.h

Trying the approach of applying the ELV selection to do 
a select copy of the SScene. 


Issue : ELV selection of a force-triangulated solid breaks::

    P[blyth@localhost tests]$ ELV=sWorld,svacSurftube_0V1_0 MOI=svacSurftube_0V1_0:0:-2 ~/o/cxr_min.sh
    /home/blyth/o/cxr_min.sh : FOUND B_CFBaseFromGEOM /home/blyth/.opticks/GEOM/J_2024aug27 containing CSGFoundry/prim.npy
                    GEOM : J_2024aug27 
                     MOI : svacSurftube_0V1_0:0:-2 
                    TMIN : 0.5 
                     EYE : 1,0,0 
                    LOOK : 0,0,0 
                      UP : 0,0,1 
                    ZOOM : 1 
                  LOGDIR : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderInteractiveTest 
                    BASE : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderInteractiveTest 
                    PBAS : /data/blyth/opticks/ 
              NAMEPREFIX : cxr_min__eye_1,0,0__zoom_1__tmin_0.5_ 
            OPTICKS_HASH : FAILED_GIT_REV_PARSE 
                 TOPLINE : ESCALE=extent EYE=1,0,0 TMIN=0.5 MOI=svacSurftube_0V1_0:0:-2 ZOOM=1 CAM=perspective ~/opticks/CSGOptiX/cxr_min.sh  
                 BOTLINE : Mon Sep  9 11:34:55 CST 2024 
    CUDA_VISIBLE_DEVICES : 1 
    /home/blyth/o/cxr_min.sh : run : delete prior LOG CSGOptiXRenderInteractiveTest.log
    CSGFoundry::Load_[/home/blyth/.opticks/GEOM/J_2024aug27]
    2024-09-09 11:34:57.283 INFO  [62582] [main@67] standard CSGFoundry::Load has scene : no need to kludge OverrideScene 
    2024-09-09 11:34:57.747 FATAL [62582] [SBT::_getOffset@716]  UNEXPECTED trimesh with   UNEQUAL:  num_bi 5 numPrim 1 gas_idx 1 mmlabel 322:solidSJCLSanchor
    CSGOptiXRenderInteractiveTest: /home/blyth/opticks/CSGOptiX/SBT.cc:724: int SBT::_getOffset(unsigned int, unsigned int) const: Assertion `are_equal' failed.
    /home/blyth/o/cxr_min.sh: line 275: 62582 Aborted                 (core dumped) $bin
    /home/blyth/o/cxr_min.sh run error




     ELV=sWorld,svacSurftube_0V1_0 MOI=svacSurftube_0V1_0:0:-2 CSGCopy=INFO DUMP_RIDX=1 DUMP_NPS=1  ~/o/cxr_min.sh



ELV selection acts on the analytic CSGFoundry geometry within CSGCopy : but it does 
not act upon the trimesh geometrty ? So the trimesh part of the below is not aware of the selection::

     314 #ifdef WITH_SOPTIX_ACCEL
     315 void SBT::createGAS(unsigned gas_idx)
     316 {
     317     SOPTIX_BuildInput* bi = nullptr ;
     318     SOPTIX_Accel* gas = nullptr ;
     319 
     320     bool trimesh = foundry->isSolidTrimesh(gas_idx); // now based on forced triangulation config 
     321 
     322     const std::string& mmlabel = foundry->getSolidMMLabel(gas_idx);
     323 
     324     LOG(LEVEL)
     325         << " WITH_SOPTIX_ACCEL "
     326         << " gas_idx " << gas_idx
     327         << " trimesh " << ( trimesh ? "YES" : "NO " )
     328         << " mmlabel " << mmlabel
     329         ;
     330 
     331     if(trimesh)
     332     {
     333         // note similarity to SOPTIX_Scene::init_GAS
     334         const SMeshGroup* mg = scene->getMeshGroup(gas_idx) ;
     335         LOG_IF(fatal, mg == nullptr)
     336             << " FAILED to SScene::getMeshGroup"
     337             << " gas_idx " << gas_idx
     338             << "\n"
     339             << scene->desc()
     340             ;
     341         assert(mg);
     342 
     343         SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create( mg ) ;
     344         gas = SOPTIX_Accel::Create(Ctx::context, xmg->bis );   
     345         xgas[gas_idx] = xmg ;
     346     }
     347     else
     348     {
     349         // analytic geometry 
     350         SCSGPrimSpec ps = foundry->getPrimSpec(gas_idx);
     351         bi = new SOPTIX_BuildInput_CPA(ps) ;
     352         gas = SOPTIX_Accel::Create(Ctx::context, bi );
     353     }
     354     vgas[gas_idx] = gas ;
     355 }


So where does scene formation happen, and how to make it respect ELV ?
The SScene is populated via a surprisingly high level call stack::

    G4CXOpticks::setGeometry 
    SSim::initSceneFromTree
    SScene::initFromTree  

ELV selection currently done at CSGFoundry level via CSGCopy with ELV
selection applied. Essentially another CSGFoundry is created and selectively 
populated. 

Recall:: 

     stree +--->  CSGFoundry 
           |
           +--->  SScene
    

Should extra ELV selection be done via copying stree to a selected new one 
or between the full stree and a selected SScene (or Scene to Scene) ?::

     stree -> stree

     stree -> SScene 
       making SScene::initFromTree respect ELV ?     
     
       * relatively easy because stree has everything : but is that what want ?
       * point of ELV selection is for dynamic geometry speed test
         so need to do this postcache 

     Scene -> Scene


::

    3008 CSGFoundry* CSGFoundry::Load() // static
    3009 {
    3010     SProf::Add("CSGFoundry__Load_HEAD");
    3011 
    3012 
    3013 
    3014     LOG(LEVEL) << "[ argumentless " ;
    3015     CSGFoundry* src = CSGFoundry::Load_() ;
    3016     if(src == nullptr) return nullptr ;
    3017 
    3018     SGeoConfig::GeometrySpecificSetup(src->id);
    3019 
    3020     const SBitSet* elv = ELV(src->id);
    3021     CSGFoundry* dst = elv ? CSGFoundry::CopySelect(src, elv) : src  ;
    3022 
    3023     if( elv != nullptr && Load_saveAlt)
    3024     {
    3025         LOG(error) << " non-standard dynamic selection CSGFoundry_Load_saveAlt " ;
    3026         dst->saveAlt() ;
    3027     }
    3028 
    3029     AfterLoadOrCreate();
    3030 
    3031     LOG(LEVEL) << "] argumentless " ;
    3032     SProf::Add("CSGFoundry__Load_TAIL");
    3033     return dst ;
    3034 }


    3126 CSGFoundry* CSGFoundry::Load_() // static
    3127 {
    3128     const char* cfbase = ResolveCFBase() ;
    3129     if(ssys::getenvbool(_Load_DUMP)) std::cout << "CSGFoundry::Load_[" << cfbase << "]\n" ;
    3130 
    3131     LOG(LEVEL) << "[ SSim::Load cfbase " << ( cfbase ? cfbase : "-" )  ;
    3132     SSim* sim = SSim::Load(cfbase, "CSGFoundry/SSim");
    3133     LOG(LEVEL) << "] SSim::Load " ;
    3134 
    3135     LOG_IF(fatal, sim==nullptr ) << " sim(SSim) required before CSGFoundry::Load " ;
    3136     assert(sim);
    3137 
    3138     CSGFoundry* fd = Load(cfbase, "CSGFoundry");
    3139     return fd ;
    3140 }



    141 SSim::SSim()
    142     :
    143     relp(ssys::getenvvar("SSim__RELP", RELP_DEFAULT )), // alt: "extra/GGeo"
    144     top(nullptr),
    145     extra(nullptr),
    146     tree(new stree),
    147     scene(new SScene)
    148 {
    149     init(); // just sets tree level 
    150 }
     
    398 void SSim::load_(const char* dir)
    399 {
    400     LOG(LEVEL) << "[" ;
    401     LOG_IF(fatal, top != nullptr)  << " top is NOT nullptr : cannot SSim::load into pre-serialized instance " ;
    402     top = new NPFold ;
    403 
    404     LOG(LEVEL) << "[ top.load [" << dir << "]" ;
    405 
    406     top->load(dir) ;
    407 
    408     LOG(LEVEL) << "] top.load [" << dir << "]" ;
    409 
    410     NPFold* f_tree = top->get_subfold( stree::RELDIR ) ;
    411     tree->import( f_tree );
    412 
    413     NPFold* f_scene = top->get_subfold( SScene::RELDIR ) ;
    414     scene->import( f_scene );
    415 
    416     LOG(LEVEL) << "]" ;
    417 }


Does the SScene have the lvid info needed to do ELV selection ? 
Probably not, but the stree does. 

Where to set SMesh.h lvid ?::

    0559 inline void U4Tree::initSolids_Mesh()
     560 {
     561     st->mesh = U4Mesh::MakeFold(solids, st->soname ) ;
     562 }


    108 inline NPFold* U4Mesh::MakeFold(
    109     const std::vector<const G4VSolid*>& solids,
    110     const std::vector<std::string>& keys
    111    ) // static
    112 {
    113     NPFold* mesh = new NPFold ;
    114     int num_solid = solids.size();
    115     int num_key = keys.size();
    116     assert( num_solid == num_key );
    117 
    118     for(int i=0 ; i < num_solid ; i++)
    119     {
    120         int lvid = i ; 
    121         const G4VSolid* so = solids[i];
    122         const char* _key = keys[i].c_str();
    123 
    124         NPFold* sub = Serialize(so) ;
    125         sub->set_meta<int>("lvid", lvid ); 
    126         
    127         mesh->add_subfold( _key, sub );
    128     }
    129     return mesh ;
    130 }


