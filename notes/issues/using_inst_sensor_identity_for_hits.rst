using_inst_sensor_identity_for_hits
======================================

TODO : get gxt gxs working again
----------------------------------

Need more flexibility in control of output dir for event and geometry.


* DONE: added SEventConfig::GeoFold()

TODO need to split geo and evt dirs::

    1391 void SEvt::save()
    1392 {
    1393     const char* dir = SGeo::DefaultDir();
    1394     LOG(LEVEL) << "DefaultDir " << dir ;
    1395     save(dir);
    1396 }
    1397 void SEvt::load()
    1398 {
    1399     const char* dir = SGeo::DefaultDir();
    1400     LOG(LEVEL) << "DefaultDir " << dir ;
    1401     load(dir);
    1402 }
    1403 


DONE : save SEvt arrays with G4CXOpticks::saveEvent and check identity info in photon+hit arrays
--------------------------------------------------------------------------------------------------

Added G4CXOpticks__simulate_saveEvent to switch on G4CXOpticks::saveEvent 
within G4CXOpticks::simulate.


::

    N[blyth@localhost offline]$ l /tmp/blyth/opticks/ntds3/ALL/z000/
    total 1008
      0 drwxr-xr-x. 4 blyth blyth     30 Aug 16 22:24 ..
      0 drwxr-xr-x. 2 blyth blyth    203 Aug 16 22:24 .
      4 -rw-rw-r--. 1 blyth blyth     21 Aug 16 22:24 sframe_meta.txt
      4 -rw-rw-r--. 1 blyth blyth    384 Aug 16 22:24 sframe.npy
      4 -rw-rw-r--. 1 blyth blyth     38 Aug 16 22:24 domain_meta.txt
      4 -rw-rw-r--. 1 blyth blyth    256 Aug 16 22:24 domain.npy
    240 -rw-rw-r--. 1 blyth blyth 242880 Aug 16 22:24 hit.npy
      4 -rw-rw-r--. 1 blyth blyth    131 Aug 16 22:24 NPFold_meta.txt
     44 -rw-rw-r--. 1 blyth blyth  43952 Aug 16 22:24 seed.npy
     12 -rw-rw-r--. 1 blyth blyth  11264 Aug 16 22:24 genstep.npy
      4 -rw-rw-r--. 1 blyth blyth     51 Aug 16 22:24 NPFold_index.txt
    688 -rw-rw-r--. 1 blyth blyth 701312 Aug 16 22:24 photon.npy

    N[blyth@localhost offline]$ l /tmp/blyth/opticks/ntds3/ALL/p001/
    total 1008
      0 drwxr-xr-x. 2 blyth blyth    203 Aug 16 22:24 .
      4 -rw-rw-r--. 1 blyth blyth     38 Aug 16 22:24 domain_meta.txt
      4 -rw-rw-r--. 1 blyth blyth    256 Aug 16 22:24 domain.npy
      4 -rw-rw-r--. 1 blyth blyth    131 Aug 16 22:24 NPFold_meta.txt
      4 -rw-rw-r--. 1 blyth blyth     21 Aug 16 22:24 sframe_meta.txt
      4 -rw-rw-r--. 1 blyth blyth    384 Aug 16 22:24 sframe.npy
      0 drwxr-xr-x. 4 blyth blyth     30 Aug 16 22:24 ..
     12 -rw-rw-r--. 1 blyth blyth   9920 Aug 16 22:24 genstep.npy
    240 -rw-rw-r--. 1 blyth blyth 244928 Aug 16 22:24 hit.npy
      4 -rw-rw-r--. 1 blyth blyth     51 Aug 16 22:24 NPFold_index.txt
    688 -rw-rw-r--. 1 blyth blyth 701120 Aug 16 22:24 photon.npy
     44 -rw-rw-r--. 1 blyth blyth  43940 Aug 16 22:24 seed.npy


gxt::

    ./sev.sh 

    In [3]: f.hit.view(np.int32)[:,1,3]
    Out[3]: array([29082, 31842, 32525, 32355, 35884, ..., 40408, 34641, 29988, 29182, 37875], dtype=int32)



Passing sensor info via the sframe ? Just as transforms are 
-------------------------------------------------------------

::

    2859 int CSGFoundry::getFrame(sframe& fr, int inst_idx) const
    2860 {
    2861     return target->getFrame( fr, inst_idx );
    2862 }

The identity info is already there::

    139 int CSGTarget::getFrame(sframe& fr, int inst_idx ) const
    140 {
    141     const qat4* _t = foundry->getInst(inst_idx);
    142 
    143     int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
    144     _t->getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );
    145 
    146     assert( ins_idx == inst_idx );
    147     fr.set_inst(inst_idx);
    148   
    149     // HMM: these values are already there inside the matrices ? 
    150     fr.set_identity(ins_idx, gas_idx, sensor_identifier, sensor_index ) ;
    151 
    152     qat4 t(_t->cdata());   // copy the instance (transform and identity info)
    153     const qat4* v = Tran<double>::Invert(&t);     // identity gets cleared in here 
    154 
    155     qat4::copy(fr.m2w,  t);
    156     qat4::copy(fr.w2m, *v);
    157 
    158     const CSGSolid* solid = foundry->getSolid(gas_idx);
    159     fr.ce = solid->center_extent ;
    160 
    161     return 0 ;
    162 }

    264 inline int sframe::ins() const { return aux.q0.i.x ; }
    265 inline int sframe::gas() const { return aux.q0.i.y ; }
    266 inline int sframe::sensor_identifier() const { return aux.q0.i.z ; }
    267 inline int sframe::sensor_index() const {      return aux.q0.i.w ; }
    268 


Get it into U4Hit via sphit::

     52 inline void U4HitGet::FromEvt(U4Hit& hit, unsigned idx )
     53 {
     54     sphoton global, local  ;
     55     SEvt* sev = SEvt::Get();
     56     sev->getHit( global, idx);
     57     
     58     sphit ht ;
     59     sev->getLocalHit( ht, local,  idx);
     60     
     61     ConvertFromPhoton(hit, global, local, ht );
     62 }



U4HitGet::ConvertFromPhoton
-------------------------------

::

     23 inline void U4HitGet::ConvertFromPhoton(U4Hit& hit,  const sphoton& global, const sphoton& local )
     24 {
     25     hit.zero();
     26 
     27     U4ThreeVector::FromFloat3( hit.global_position,      global.pos );
     28     U4ThreeVector::FromFloat3( hit.global_direction,     global.mom );
     29     U4ThreeVector::FromFloat3( hit.global_polarization,  global.pol );
     30 
     31     hit.time = double(global.time) ;
     32     hit.weight = 1. ;
     33     hit.wavelength = double(global.wavelength);
     34 
     35     U4ThreeVector::FromFloat3( hit.local_position,      local.pos );
     36     U4ThreeVector::FromFloat3( hit.local_direction,     local.mom );
     37     U4ThreeVector::FromFloat3( hit.local_polarization,  local.pol );
     38 
     39     // TODO: derive the below 3 from global.iindex using the stree nodes 
     40     // HMM: how to access the stree ? it belong with SGeo like the transforms needed for SEvt::getLocalHit 
     41     //hit.sensorIndex = ;   
     42     //hit.nodeIndex = ;    
     43     //hit.sensor_identifier  ; 
     44 


::

    1607 /**
    1608 SEvt::getLocalPhoton SEvt::getLocalHit
    1609 -----------------------------------------
    1610 
    1611 sphoton::iindex instance index used to get instance frame
    1612 from (SGeo*)cf which is used to transform the photon  
    1613 
    1614 **/
    1615 
    1616 void SEvt::getLocalPhoton(sphoton& lp, unsigned idx) const
    1617 {
    1618     getPhoton(lp, idx);
    1619     applyLocalTransform_w2m(lp);
    1620 }
    1621 void SEvt::getLocalHit(sphoton& lp, unsigned idx) const
    1622 {
    1623     getHit(lp, idx);
    1624     applyLocalTransform_w2m(lp);
    1625 }
    1626 void SEvt::applyLocalTransform_w2m( sphoton& lp) const
    1627 {
    1628     sframe fr ;
    1629     getPhotonFrame(fr, lp);
    1630     fr.transform_w2m(lp);
    1631 }
    1632 void SEvt::getPhotonFrame( sframe& fr, const sphoton& p ) const
    1633 {
    1634     assert(cf);
    1635     cf->getFrame(fr, p.iindex);
    1636     fr.prepare();
    1637 }

    0274 void SEvt::setGeo(const SGeo* cf_)
     275 {
     276     cf = cf_ ;
     277 }

     23 struct SYSRAP_API SGeo
     24 {
     ...
     35         virtual unsigned           getNumMeshes() const = 0 ;
     36         virtual const char*        getMeshName(unsigned midx) const = 0 ;
     37         virtual int                getMeshIndexWithName(const char* name, bool startswith) const = 0 ;
     38         virtual int                getFrame(sframe& fr, int ins_idx ) const = 0 ;
     39         virtual std::string        descBase() const = 0 ;
     ...
     43 };



review identity info 
----------------------

HMM: suspect that OptixInstance::instanceId is currently the same as the automatic instanceIndex ?

YES: confirmed this::

   gxt
   ./sev.sh 

    In [20]: ev.hit[:,3,1].view(np.int32)   ## sphoton::iindex
    Out[20]: array([203125146, 203127906, 203128589, 203128419, 203131948, ..., 203595224, 203130705, 203126052, 203125246, 203133939], dtype=int32)

    In [21]: ev.hit[:,3,1].view(np.int32) & 0xffff   ## lower half of sphoton::identity 
    Out[21]: array([29082, 31842, 32525, 32355, 35884, ..., 40408, 34641, 29988, 29182, 37875], dtype=int32)

    In [22]: ev.hit[:,1,3].view(np.int32)   
    Out[22]: array([29082, 31842, 32525, 32355, 35884, ..., 40408, 34641, 29988, 29182, 37875], dtype=int32)

    In [23]: np.all( (ev.hit[:,3,1].view(np.int32) & 0xffff) == ev.hit[:,1,3].view(np.int32) )
    Out[23]: True


    In [24]: ev.hit[:,3,1].view(np.int32) >> 16
    Out[24]: array([3099, 3099, 3099, 3099, 3099, ..., 3106, 3099, 3099, 3099, 3099], dtype=int32)

    In [24]: ev.hit[:,3,1].view(np.int32) >> 16
    Out[24]: array([3099, 3099, 3099, 3099, 3099, ..., 3106, 3099, 3099, 3099, 3099], dtype=int32)

    In [25]: np.unique( ev.hit[:,3,1].view(np.int32) >> 16 , return_counts=True )
    Out[25]: (array([3091, 3099, 3106], dtype=int32), array([ 138, 2587, 1068]))

    In [27]: cf.primname[[3091,3099,3106]]
    Out[27]: array(['PMT_3inch_inner1_solid_ell_helper', 'NNVTMCPPMT_PMT_20inch_inner1_solid_head', 'HamamatsuR12860_PMT_20inch_inner1_solid_I'], dtype=object)


    In [11]: iid[hit_ii]
    Out[11]: 
    array([[29082,     2,  4938,  4938],
           [31842,     2,  8753,  8753],
           [32525,     2,  9656,  9656],
           [32355,     2,  9485,  9485],
           [35884,     2, 14395, 14395],
           ...,
           [40408,     3,  7566,  7566],
           [34641,     2, 12663, 12663],
           [29988,     2,  6174,  6174],
           [29182,     2,  5082,  5082],
           [37875,     2, 17165, 17165]], dtype=int32)

    In [18]: np.unique( iid[hit_ii,1], return_counts=True )
    Out[18]: (array([1, 2, 3], dtype=int32), array([ 138, 2587, 1068]))












::

     42 void IAS_Builder::Build(IAS& ias, const std::vector<qat4>& ias_inst, const SBT* sbt) // static 
     43 {
     44     unsigned num_ias_inst = ias_inst.size() ;
     45     LOG(LEVEL) << "num_ias_inst " << num_ias_inst ;
     46     assert( num_ias_inst > 0);
     47 
     48     unsigned flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;
     49 
     50     std::vector<OptixInstance> instances ;
     51     for(unsigned i=0 ; i < num_ias_inst ; i++)
     52     {
     53         const qat4& q = ias_inst[i] ;
     54         int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
     55         q.getIdentity(ins_idx, gas_idx, sensor_identifier, sensor_index );
     56         unsigned prim_idx = 0u ;  // need offset for the outer prim(aka layer) of the GAS 
     57 
     58         const GAS& gas = sbt->getGAS(gas_idx);
     59 
     60         OptixInstance instance = {} ;
     61         q.copy_columns_3x4( instance.transform );
     62 
     63         instance.instanceId = ins_idx ;  // perhaps bitpack gas_idx, ias_idx ?
     64         instance.sbtOffset = sbt->getOffset(gas_idx, prim_idx );
     65         instance.visibilityMask = 255;
     66         instance.flags = flags ;
     67         instance.traversableHandle = gas.handle ;
     68    
     69         instances.push_back(instance);
     70     }
     71     Build(ias, instances);
     72 }



cx::

    404 extern "C" __global__ void __closesthit__ch()
    405 {
    406     unsigned iindex = optixGetInstanceIndex() ;    // 0-based index within IAS
    407     unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build and InstanceId.h 
    408     unsigned prim_idx = optixGetPrimitiveIndex() ; // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    409     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ;
    410 
    411 #ifdef WITH_PRD
    412     quad2* prd = getPRD<quad2>();
    413 
    414     prd->set_identity( identity ) ;
    415     prd->set_iindex(   iindex ) ;
    416     //printf("//__closesthit__ch prd.boundary %d \n", prd->boundary() );  // boundary set in IS for WITH_PRD
    417     float3* normal = prd->normal();
    418     *normal = optixTransformNormalFromObjectToWorldSpace( *normal ) ;
    419 




cx::

    epsilon:CSGOptiX blyth$ grep iindex *.*
    CSGOptiX7.cu:    prd->set_iindex(p7) ;  
    CSGOptiX7.cu:static __forceinline__ __device__ void setPayload( float normal_x, float normal_y, float normal_z, float distance, unsigned identity, unsigned boundary, float lposcost, unsigned iindex )
    CSGOptiX7.cu:    optixSetPayload_7( iindex   );  
    CSGOptiX7.cu:    unsigned iindex = optixGetInstanceIndex() ;    // 0-based index within IAS
    CSGOptiX7.cu:    prd->set_iindex(   iindex ) ;
    CSGOptiX7.cu:    setPayload( normal.x, normal.y, normal.z, distance, identity, boundary, lposcost, iindex );  // communicate from ch->rg


qu::

    epsilon:qudarap blyth$ grep iindex *.*
    qcerenkov.h:    p.iindex = 0u ; 
    qscint.h:    p.iindex = 0u ; 
    qsim.h:    const unsigned iindex = ctx.prd->iindex() ; 
    qsim.h:    ctx.p.set_prd(boundary, identity, cosTheta, iindex ); 
    epsilon:qudarap blyth$ 


::

    1258 inline QSIM_METHOD int qsim::propagate(const int bounce, curandStateXORWOW& rng, sctx& ctx )
    1259 {
    1260     const unsigned boundary = ctx.prd->boundary() ;
    1261     const unsigned identity = ctx.prd->identity() ;
    1262     const unsigned iindex = ctx.prd->iindex() ;
    1263     const float3* normal = ctx.prd->normal();
    1264     float cosTheta = dot(ctx.p.mom, *normal ) ;
    ....
    1272     ctx.p.set_prd(boundary, identity, cosTheta, iindex );




