#pragma once

/**
qevent : host/device communication instance
=============================================

Instantiation of qevent is done by QEvent::init 
and the instance is subsequently uploaded to the device after 
device buffer allocations hence the qevent instance
provides event config and device buffer pointers 
both on device and host. 

Note that *num_seed* and *num_photon* will be equal in 
normal operation which uses QEvent::setGensteps. 
However for clarity separate fields are used to 
distinguish photon test running that directly uses
QEvent::setNumPhoton 

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define QEVENT_METHOD __device__ __forceinline__
#else
#    define QEVENT_METHOD inline 
#endif



struct float4 ; 
struct float2 ; 
struct quad4 ; 
struct quad6 ; 
struct srec ; 
struct sphoton ; 

struct qevent
{
    static constexpr unsigned genstep_itemsize = 6*4 ; 
    static constexpr unsigned genstep_numphoton_offset = 3 ; 

    float4 center_extent ; 
    float2 time_domain ; 
    float2 wavelength_domain ; 


    // values here come from SEventConfig 
    int      max_genstep ; // eg:      100,000
    int      max_photon  ; // eg: 100,000,0000
    int      max_bounce  ; // eg:            9 
    int      max_record  ; // eg:           10  full step record 
    int      max_rec     ; // eg:           10  compressed step record


    int      num_genstep ; 
    quad6*   genstep ; 

    int      num_seed ; 
    int*     seed ;     

    int      num_photon ; 
    sphoton* photon ; 

    int      num_record ; 
    sphoton* record ; 

    int      num_rec ; 
    srec*    rec ; 

    int      num_hit ; 
    sphoton* hit ; 


    QEVENT_METHOD void add_rec( srec& r, unsigned idx, unsigned bounce, const sphoton& p); 

    // not including prd here as that is clearly for debugging only 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    QEVENT_METHOD void zero(); 
#endif 

}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
QEVENT_METHOD void qevent::zero()
{
    num_genstep = 0 ; 
    num_seed  = 0 ; 
    num_photon = 0 ; 
    num_record = 0 ; 
    num_rec = 0 ; 
    num_hit = 0 ; 

    genstep = nullptr ; 
    seed = nullptr ; 
    photon = nullptr ; 
    record = nullptr ; 
    rec = nullptr ; 
    hit = nullptr ; 
    
}
#endif 


QEVENT_METHOD void  qevent::add_rec( srec& r, unsigned idx, unsigned bounce, const sphoton& p )
{
    r.set_position(     p.pos,  center_extent ); 
    r.set_time(         p.time, time_domain ); 
    r.set_polarization( p.pol ); 
    r.set_wavelength(   p.wavelength, wavelength_domain ); 
    // flags ?

    rec[max_rec*idx+bounce] = r ;      
}
