how_to_integrate_triangulated_geom_with_analytic
==================================================


What level of ana/tri split ? CSGSolid
----------------------------------------

1:1 CSGSolid:GAS

As each GAS must be either analytic or triangulated 
have to split at CSGSolid level. 

That means if have a G4VSolid (eg the guide tube torus) 
that must be triangulated then must arrange for the corresponding 
CSGPrim to be isolated into its own CSGSolid. 

Initially can just assert that selected CSGPrim must be isolated, 
as that will be the case for the guide tube. 


Recent addition triangulated geom is dev in sysrap/SOPTIX,SScene
--------------------------------------------------------------------

* :doc:`sysrap/SOPTIX`


High level geometry workflow
------------------------------


::

    227 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    228 {
    229     LOG(LEVEL) << "[ G4VPhysicalVolume world " << world ;
    230     assert(world);
    231     wd = world ;
    232 
    233     assert(sim && "sim instance should have been grabbed/created in ctor" );
    234     stree* st = sim->get_tree();
    235 
    236     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    237     LOG(LEVEL) << "Completed U4Tree::Create " ;
    238 
    239     sim->initSceneFromTree(); // not so easy to do at lower level  
    240 
    241 
    242     CSGFoundry* fd_ = CSGFoundry::CreateFromSim() ; // adopts SSim::INSTANCE  
    243     setGeometry(fd_);
    244 
    245     LOG(info) << Desc() ;
    246 
    247     LOG(LEVEL) << "] G4VPhysicalVolume world " << world ;
    248 }



::

     079 /**
      80 CSGFoundry::CSGFoundry
      81 ------------------------
      82 
      83 HMM: the dependency between CSGFoundry and SSim is a bit mixed up
      84 because of the two possibilities:
      85 
      86 1. "Import" : create CSGFoundry from SSim/stree using CSGImport
      87 2. "Load"   : load previously created and persisted CSGFoundry + SSim from file system 
      88 
      89 sim(SSim) used to be a passive passenger of CSGFoundry but now that CSGFoundry 
      90 can be CSGImported from SSim it is no longer so passive. 
      91 
      92 **/
      93 
      94 CSGFoundry::CSGFoundry()
      95     :
      96     d_prim(nullptr),
      97     d_node(nullptr),
      98     d_plan(nullptr),
      99     d_itra(nullptr),
     100     sim(SSim::Get()),
     101     import(new CSGImport(this)),




Workflow : how to add tri ?
-------------------------------

SSim
   holds stree(ana) and SScene(tri)

CSGFoundry 
   has sim member giving access to both stree and SScene

CSGFoundry::CreateFromSim/CSGFoundry::importSim
   populates CSGFoundry from stree 


* HMM: simpler to have parallel ana+tri throughout the geometry workflow with the 
  ana/tri switch done at the GAS handle creation stage 

* ana at all stages is very small, so no resource issue, 
  tri could be large for the remainder instance : so want to 
  do ana/tri switch before GPU (hmm might not be so easy with SOPTIX)

  * this might need SOPTIX_MeshGroup reworking to defer uploads : unless
    just deferred usage of that until GAS-handle stage  
 


DONE : made a more vertical API for tri/ana integration
--------------------------------------------------------

::

   SOPTIX_MeshGroup* Create( OptixDeviceContext& ctx, const SMeshGroup* mg );

   SMeshGroup* mg = scene->meshgroup[i] ;  
   SOPTIX_MeshGroup* xmg = SOPTIX_MeshGroup::Create( ctx, mg ) ; 
   xmg->gas->handle  



NEXT: name based ana/tri control 
-------------------------------------



Analytic in stree/CSG/CSGOptiX 
---------------------------------

::

     551 void CSGOptiX::initGeometry()
     552 {
     553     LOG(LEVEL) << "[" ;
     554     params->node = foundry->d_node ;
     555     params->plan = foundry->d_plan ;
     556     params->tran = nullptr ;
     557     params->itra = foundry->d_itra ;
     558 
     559     bool is_uploaded =  params->node != nullptr ;
     560     LOG_IF(fatal, !is_uploaded) << "foundry must be uploaded prior to CSGOptiX::initGeometry " ;
     561     assert( is_uploaded );
     562 
     563 #if OPTIX_VERSION < 70000
     564     six->setFoundry(foundry);
     565 #else
     566     LOG(LEVEL) << "[ sbt.setFoundry " ;
     567     sbt->setFoundry(foundry);
     568     LOG(LEVEL) << "] sbt.setFoundry " ;
     569 #endif
     570     const char* top = Top();
     571     setTop(top);
     572     LOG(LEVEL) << "]" ;
     573 }


::

   CSGOptiX::initGeometry
   SBT::setFoundry
   SBT::createGeom
   SBT::createGAS_Standard



Where+how to ana/tri branch ?
-------------------------------

EMM is integer based.  Need name based gas_idx control for greater longevity. 

::

     261 void SBT::createGAS_Standard()
     262 {
     263     unsigned num_solid = foundry->getNumSolid();   // STANDARD_SOLID
     264     for(unsigned i=0 ; i < num_solid ; i++)
     265     {
     266         unsigned gas_idx = i ;
     267 
     268         bool enabled = SGeoConfig::IsEnabledMergedMesh(gas_idx) ;
     269         bool enabled2 = emm & ( 0x1 << gas_idx ) ;
     270         bool enabled_expect = enabled == enabled2 ;
     271         assert( enabled_expect );
     272         if(!enabled_expect) std::raise(SIGINT);
     273 
     274         if( enabled )
     275         {
     276             LOG(LEVEL) << " emm proceed " << gas_idx ;
     277             createGAS(gas_idx);
     278         }
     279         else
     280         {
     281             LOG(LEVEL) << " emm skip " << gas_idx ;
     282         }
     283     } 
     284     LOG(LEVEL) << descGAS() ;
     285 }  


Commonality between ana and tri is the handle
---------------------------------------------------

* HMM: SOPTIX side "gas" is SOPTIX_Accel instance
* WIP: maybe standardize by using the handle in the  vgas map ?

  * NOPE: NEED NUMBER OF buildInputs FOR SBT MECHANICS
  * added reference to the vector in SOPTIX_Accel MAYBE NEEDS TO BE pointer to vector on heap ?


::

   00305 void SBT::createGAS(unsigned gas_idx)
     306 {
     307     CSGPrimSpec ps = foundry->getPrimSpec(gas_idx);
     308     GAS gas = {} ;
     309     GAS_Builder::Build(gas, ps);
     310     vgas[gas_idx] = gas ;
     311 }

   0005 struct AS
      6 {
      7     CUdeviceptr             d_buffer;
      8     OptixTraversableHandle  handle ;
      9 };


* IAS_Builder::CollectInstances sets gas.handle into OptixInstance



Should CSGOptiX adopt some of SOPTIX ? 
---------------------------------------------

SOPTIX_Accel
    builds acceleration structure GAS or IAS from the buildInputs

    * could replace:: 

       GAS_Builder::BoilerPlate 
       IAS_Builder::Build


HMM: many of the CSGOptiX::initXXX and SBT.h PIP.h could be 
replaced by SOPTIX but not much motivation unless can show better
performance.  


Need to check perf as make such changes
------------------------------------------



