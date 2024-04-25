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



The new triangulated is dev in sysrap/SOPTIX,SScene
-----------------------------------------------------

* :doc:`sysrap/SOPTIX`


Analytic in stree/CSG/CSGOptiX
---------------------------------


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



