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


struct quad4 ; 
struct quad6 ; 
struct srec ; 

struct qevent
{
    static constexpr unsigned genstep_itemsize = 6*4 ; 
    static constexpr unsigned genstep_numphoton_offset = 3 ; 

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
    quad4*   photon ; 

    int      num_record ; 
    quad4*   record ; 

    int      num_rec ; 
    srec*    rec ; 

    int      num_hit ; 
    quad4*   hit ; 


    // not including prd here as that is clearly for debugging only 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void zero(); 
#endif 

}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline void qevent::zero()
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

