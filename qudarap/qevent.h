#pragma once

struct quad6 ; 
struct quad4 ; 

struct qevent
{
    unsigned num_genstep ; 
    quad6*   genstep ; 

    unsigned num_seed ; 
    int*     seed ;     

    unsigned num_photon ; 
    quad4*   photon ; 
}; 


