identity-review
==================

* from :doc:`cxsim-shakedown`

Where does the sphoton identity info come from ?
----------------------------------------------------

ana/p.py::

     22 identity_ = lambda p:p.view(np.uint32)[3,1]
     23 primIdx_   = lambda p:identity_(p) >> 16
     24 instanceId_  = lambda p:identity_(p) & 0xffff
     25 
     26 idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff      ## HMM: looks to be always zero 
     27 orient_   = lambda p:p.view(np.uint32)[3,2] >> 31
     28 
     29 flagmask_ = lambda p:p.view(np.uint32)[3,3]
     30 
     31 flagdesc_ = lambda p:" %6d prd(%3d %4d %5d %1d)  %3s  %15s " % ( idx_(p),  boundary_(p),primIdx_(p),instanceId_(p), orient_(p),  hm.label(flag_(p)),hm.label( flagmask_(p) ))


::

    402 extern "C" __global__ void __closesthit__ch()
    403 {
    404     //unsigned instance_index = optixGetInstanceIndex() ;  0-based index within IAS
    405     unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build and InstanceId.h 
    406     unsigned prim_idx = optixGetPrimitiveIndex() ;  // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    407     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ;
    408 
    409 #ifdef WITH_PRD
    410     quad2* prd = getPRD<quad2>();
    411 
    412     prd->set_identity( identity ) ;
    413     //printf("//__closesthit__ch prd.boundary %d \n", prd->boundary() );  // boundary set in IS for WITH_PRD
    414     float3* normal = prd->normal();


    1145 inline QSIM_METHOD int qsim::propagate(const int bounce, sphoton& p, qstate& s, const quad2* prd, curandStateXORWOW& rng, unsigned idx )
    1146 {
    1147     const unsigned boundary = prd->boundary() ;
    1148     const unsigned identity = prd->identity() ;
    1149     const float3* normal = prd->normal();
    1150     float cosTheta = dot(p.mom, *normal ) ;    
    1151    
    1152 #ifdef DEBUG_COSTHETA
    1153     if( idx == pidx ) printf("//qsim.propagate idx %d bnc %d cosTheta %10.4f dir (%10.4f %10.4f %10.4f) nrm (%10.4f %10.4f %10.4f) \n",
    1154                  idx, bounce, cosTheta, p.mom.x, p.mom.y, p.mom.z, normal->x, normal->y, normal->z );
    1155 #endif
    1156 
    1157     p.set_prd(boundary, identity, cosTheta);
    1158 

    050 struct sphoton
     51 {
     52     float3 pos ;
     53     float  time ;
     54 
     55     float3 mom ;
     56     float  weight ;
     57 
     58     float3 pol ;
     59     float  wavelength ;
     60 
     61     unsigned boundary_flag ;  // p.view(np.uint32)[3,0] 
     62     unsigned identity ;       // p.view(np.uint32)[3,1]
     63     unsigned orient_idx ;     // p.view(np.uint32)[3,2]
     64     unsigned flagmask ;       // p.view(np.uint32)[3,3]
     65 
     66 
     67     SPHOTON_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient );


     72     SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it 
     73     SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 
     74 
    105 SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_ )
    106 {
    107     set_boundary(boundary_);
    108     identity = identity_ ;
    109     set_orient( orient_ );
    110 }




