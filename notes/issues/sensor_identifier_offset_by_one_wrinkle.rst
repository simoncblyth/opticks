sensor_identifier_offset_by_one_wrinkle
==========================================

Overview
----------

Much of the inconvenience of the offsetting 
will go away after adopting the current 
SEvt::getLocalHit_ALT impl as SEvt::getLocalHit

Thats because the stree/iinst identity is never incremented
unlike the CSGFoundry one which need to be uploaded.  


Why the -1 fiddle ? How to contain it better ?
------------------------------------------------

Comment from SEvt::getLocalHit_ALT

Dec 19,2023 : Added sensor_identifier subtract one 
to SEvt::getLocalHit corresponding to the addition of one in::

   CSGFoundry::addInstance firstcall:true
   CSGFoundry::addInstanceVector

That was done in a hurry as a bug fix: need to reconsider
how to do this better. 

The need for the offset by one arises from the optixInstance identifier 
range limitation meaning that need zero to mean not-a-sensor
and not -1 0xffffffff

BUT: the limitation is on the geometry optixInstance identifier, 
the limitation does not apply to the photon/hit struct.

So could this offsetting be done GPU side when info from the geometry 
is copied into the photon to avoid the need to offset hits way 
out in SEvt::getLocalHit ? This would avoid complications of testing 
when running from pre-persisted hits would need to do something different
than when running from hits fresh from the GPU.  



Where the incrementSensorIdentifier +1 happens ?
-------------------------------------------------

::

    1939 /**
    1940 CSGFoundry::addInstanceVector
    1941 ------------------------------
    1942 
    1943 stree.h/snode.h uses sensor_identifier -1 to indicate not-a-sensor, but 
    1944 that is not convenient on GPU due to OptixInstance.instanceId limits.
    1945 Hence here make transition by adding 1 and treating 0 as not-a-sensor, 
    1946 with the sqat4::incrementSensorIdentifier method
    1947 
    1948 **/
    1949 
    1950 void CSGFoundry::addInstanceVector( const std::vector<glm::tmat4x4<float>>& v_inst_f4 )
    1951 {
    1952     assert( inst.size() == 0 );
    1953     int num_inst = v_inst_f4.size() ;
    1954 
    1955     for(int i=0 ; i < num_inst ; i++)
    1956     {
    1957         const glm::tmat4x4<float>& inst_f4 = v_inst_f4[i] ;
    1958         const float* tr16 = glm::value_ptr(inst_f4) ;
    1959         qat4 instance(tr16) ;
    1960         instance.incrementSensorIdentifier() ; // GPU side needs 0 to mean "not-a-sensor"
    1961         inst.push_back( instance );
    1962     }
    1963 }



When the sensorIdentifier is read from geometry and recorded into photon/hit ?
----------------------------------------------------------------------------------

* first into quad2(prd)unsigned

In CH the unsigned identity (sensor_identifier+1 [0 for not-a-sensor]) is
copied from geometry into the quad2(prd)unsigned::

    493 extern "C" __global__ void __closesthit__ch()
    494 {
    495     unsigned iindex = optixGetInstanceIndex() ;
    496     unsigned identity = optixGetInstanceId() ; 
    497     
    498 #ifdef WITH_PRD
    499     quad2* prd = getPRD<quad2>();
    500     
    501     prd->set_identity( identity ) ;
    502     prd->set_iindex(   iindex ) ; 
    503     float3* normal = prd->normal();
    504     *normal = optixTransformNormalFromObjectToWorldSpace( *normal ) ;
    505 
    506 #else
    507     const float3 local_normal =    // geometry object frame normal at intersection point 
    508         make_float3(
    509                 uint_as_float( optixGetAttribute_0() ),
    510                 uint_as_float( optixGetAttribute_1() ),
    511                 uint_as_float( optixGetAttribute_2() )
    512                 );
    513 
    514     const float distance = uint_as_float(  optixGetAttribute_3() ) ;
    515     unsigned boundary = optixGetAttribute_4() ;
    516     const float lposcost = uint_as_float( optixGetAttribute_5() ) ;
    517     float3 normal = optixTransformNormalFromObjectToWorldSpace( local_normal ) ;
    518 
    519     setPayload( normal.x, normal.y, normal.z, distance, identity, boundary, lposcost, iindex );  // communicate from ch->rg
    520 #endif
    521 }


::

    2180 inline QSIM_METHOD int qsim::propagate(const int bounce, curandStateXORWOW& rng, sctx& ctx )
    2181 {
    2182     const unsigned boundary = ctx.prd->boundary() ;
    2183     const unsigned identity = ctx.prd->identity() ; // sensor_identifier+1, 0:not-a-sensor 
    2184     const unsigned iindex = ctx.prd->iindex() ;
    2185     const float lposcost = ctx.prd->lposcost() ;  // local frame intersect position cosine theta 
    2186 
    2187     const float3* normal = ctx.prd->normal();
    2188     float cosTheta = dot(ctx.p.mom, *normal ) ;
    2189 
    2190 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    2191     if( ctx.idx == base->pidx )
    2192     {
    2193     printf("\n//qsim.propagate.head idx %d : bnc %d cosTheta %10.8f \n", ctx.idx, bounce, cosTheta );
    2194 
    2195     printf("//qsim.propagate.head idx %d : mom = np.array([%10.8f,%10.8f,%10.8f]) ; lmom = %10.8f  \n",
    2196                  ctx.idx, ctx.p.mom.x, ctx.p.mom.y, ctx.p.mom.z, length(ctx.p.mom) ) ; 
    2197 
    2198     printf("//qsim.propagate.head idx %d : pos = np.array([%10.5f,%10.5f,%10.5f]) ; lpos = %10.8f \n",
    2199                  ctx.idx, ctx.p.pos.x, ctx.p.pos.y, ctx.p.pos.z, length(ctx.p.pos) ) ; 
    2200 
    2201     printf("//qsim.propagate.head idx %d : nrm = np.array([(%10.8f,%10.8f,%10.8f]) ; lnrm = %10.8f  \n",
    2202                  ctx.idx, normal->x, normal->y, normal->z, length(*normal) ); 
    2203 
    2204     }
    2205 #endif
    2206 
    2207     ctx.p.set_prd(boundary, identity, cosTheta, iindex );  // HMM: lposcost not passed along 
    2208 



HMM sphoton::set_prd looks like a good place to remove the +1 as it 
corresponds to the transition between geometry and event info.
That means changing identity member to int::

    struct sphoton
    {
        float3 pos ;        // 0
        float  time ;

        float3 mom ;        // 1 
        unsigned iindex ;   // instance index, t.record[:,:,1,3].view(np.int32)  

        float3 pol ;         // 2
        float  wavelength ;

        unsigned boundary_flag ;  // 3   
        unsigned identity ;       // [:,3,1]
        unsigned orient_idx ;
        unsigned flagmask ;
    ...
    };


    SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_, unsigned iindex_ )
    {
        set_boundary(boundary_);
        identity = identity_ ;
        set_orient( orient_ );
        iindex = iindex_ ;
    }



BUT where is the identity used GPU side ?
-------------------------------------------

It is used direct from ctx.prd::

    1785 inline QSIM_METHOD int qsim::propagate_at_surface_CustomART(unsigned& flag, curandStateXORWOW& rng, sctx& ctx) const
    1786 {
    1787 
    1788     const sphoton& p = ctx.p ;
    1789     const float3* normal = (float3*)&ctx.prd->q0.f.x ;  // geometrical outwards normal 
    1790     int lpmtid = ctx.prd->identity() - 1 ;  // identity comes from optixInstance.instanceId where 0 means not-a-sensor  
    1791     float minus_cos_theta = dot(p.mom, *normal);
    1792     float dot_pol_cross_mom_nrm = dot(p.pol,cross(p.mom,*normal)) ;





