#pragma once

#include "QUDARAP_API_EXPORT.hh"

/**
QCK : carrier struct holding Cerenkov lookup arrays, created by QCerenkov  
===========================================================================

Contains NP arrays:

bis
    (ny,) : BetaInverse domain values in range from from 1. to RINDEX_max

s2c
    (ny,nx,2) : *ny* cumulative Cerenkov sin^2 theta "s2" energy integrals up to *nx* energy cuts. 
    Last "payload" dimension is 2 as the energy domain values are kept together with the 
    cumulative integral values. This is necessary as the energies are not common, 
    with each BetaInverse having different energy ranges over which Cerenkov photons 
    can be produced. 

s2cn
    (ny,nx,2) : normalized version of *s2c* with all values divided by the maximum energy 
    integral values giving the inverse-CDF ICDF. This is usable via NP::pdomain
    to perform domain lookups to generate Cerenkov wavelengths given input BetaInverse
    and a random number.   

**/

struct NP ; 

struct QUDARAP_API QCK
{
    NP* bis ;  // BetaInverse 
    NP* s2c ;  // cumulative s2 integral 
    NP* s2cn ; // normalized *s2c*, aka *icdf* 

     // template<typename T> T energy_lookup( const T BetaInverse, const T u) ;  

    // TODO: add save and load to allow using testing separate from the QCerenkov creator 

}; 

