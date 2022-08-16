using_inst_sensor_identity_for_hits
======================================

TODO: save SEvt arrays with G4CXOpticks::saveEvent and check identity info in photon+hit arrays
--------------------------------------------------------------------------------------------------

review identity info 
----------------------

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

HMM: suspect that OptixInstance::instanceId is currently the same as the automatic instanceIndex ?


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




