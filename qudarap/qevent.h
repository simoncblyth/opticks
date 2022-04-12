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


struct quad6 ; 
struct quad4 ; 

struct qevent
{
    static constexpr unsigned genstep_itemsize = 6*4 ; 
    static constexpr unsigned genstep_numphoton_offset = 3 ; 

    int      max_genstep ;  
    int      max_photon  ;  

    int      num_genstep ; 
    quad6*   genstep ; 

    int      num_seed ; 
    int*     seed ;     

    int      num_photon ; 
    quad4*   photon ; 


}; 


