CSGOptiX_simulate_avoid_normalizing_every_normal
===================================================

Related
---------

* ~/o/notes/issues/intersect_leaf_normals_from_Ellipsoid_not_normalized.rst
* ~/o/CSG/tests/csg_intersect_leaf_test.sh 


Use blame to find where I added the kludge 
---------------------------------------------

* https://bitbucket.org/simoncblyth/opticks/annotate/master/CSGOptiX/CSGOptiX7.cu?at=master
* https://bitbucket.org/simoncblyth/opticks/commits/ae8415ff75cc71719703440eb02dec58dbcc2b71

Commit::

    "normalizing the normal after trace in CSGOptiX7.cu looks to have fixed the discrepancy"


Issue : kludge normalizing every normal
-------------------------------------------

As a fix for non-normalized ellipsoid normals added normalization of every normal in CSGOptiX7.cu::

    273     while( bounce < evt->max_bounce )
    274     {   
    275         trace( params.handle, ctx.p.pos, ctx.p.mom, params.tmin, params.tmax, prd);  // geo query filling prd      
    276         if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
    277         // propagate can do nothing meaningful without a boundary 
    278 
    279         // HMM: normalize here or within CSG ? Actually only needed for 
    280         // geometry with active scaling, such as ellipsoid.  
    281         // TODO: move this so its only done when needed
    282         float3* normal = prd->normal();
    283         *normal = normalize(*normal);
    284 

Thats a kludge, as normally no need to normalize the normal. 
Should be happening at lower level only for ellipsoid::

    478 extern "C" __global__ void __closesthit__ch()
    479 {
    480     unsigned iindex = optixGetInstanceIndex() ;    // 0-based index within IAS
    481     unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build 
    482     unsigned prim_idx = optixGetPrimitiveIndex() ; // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    483 
    484     //unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ; 
    485     unsigned identity = instance_id ;  // CHANGED July 2023, as now carrying sensor_identifier, see sysrap/sqat4.h 
    486 
    487 #ifdef WITH_PRD
    488     quad2* prd = getPRD<quad2>();
    489 
    490     prd->set_identity( identity ) ;
    491     prd->set_iindex(   iindex ) ;
    492     //printf("//__closesthit__ch prd.boundary %d \n", prd->boundary() );  // boundary set in IS for WITH_PRD
    493     float3* normal = prd->normal();
    494     *normal = optixTransformNormalFromObjectToWorldSpace( *normal ) ;
    495 
    496 #else
    497     const float3 local_normal =    // geometry object frame normal at intersection point 
    498         make_float3(
    499                 uint_as_float( optixGetAttribute_0() ),
    500                 uint_as_float( optixGetAttribute_1() ),
    501                 uint_as_float( optixGetAttribute_2() )
    502                 );
    503 
    504     const float distance = uint_as_float(  optixGetAttribute_3() ) ;
    505     unsigned boundary = optixGetAttribute_4() ;
    506     const float lposcost = uint_as_float( optixGetAttribute_5() ) ;
    507     float3 normal = optixTransformNormalFromObjectToWorldSpace( local_normal ) ;
    508 
    509     setPayload( normal.x, normal.y, normal.z, distance, identity, boundary, lposcost, iindex );  // communicate from ch->rg
    510 #endif
    511 }




