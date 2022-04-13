#pragma once

/**
qevent
=======

Instance used to communicate device buffer pointers 
and numbers of items between host and device. 

Note that *num_seed* and *num_photon* will be equal in 
normal operation which uses QEvent::setGensteps. 
However for clarity separate fields are used to 
distinguish photon test running that directly uses
QEvent::setNumPhoton 

**/


struct quad4 ; 
struct quad6 ; 

struct qevent
{
    static constexpr unsigned genstep_itemsize = 6*4 ; 
    static constexpr unsigned genstep_numphoton_offset = 3 ; 

    int      max_genstep ; // eg:      100,000
    int      max_photon  ; // eg: 100,000,0000
    int      max_bounce  ; // eg:            9 
    int      max_record  ; // eg:           10

    int      num_genstep ; 
    quad6*   genstep ; 

    int      num_seed ; 
    int*     seed ;     

    int      num_photon ; 
    quad4*   photon ; 

    int      num_record ; 
    quad4*   record ; 


    // not including prd here as that is clearly for debugging only 

}; 


