#pragma once
/**
QCurandState.hh : creates states
=====================================

* loading from file is handled by QRng::Load 
* HMM:maybe move into here ?

The curandState originate on the device as a result of 
calling curand_init and they need to be downloaded and stored
into files named informatively with seeds, counts, offsets etc..

A difficulty is that calling curand_init is a very heavy kernel, 
so need to split up into multiple launches that all write into the same file. 
OR could split up into multiple files ?

* HMM : start with small sizes and verify that reproduces the cudarap states 
  before scaling it up 

* curandState Content size is 44 bytes which get padded to 48 bytes in the file. 

**/

#include <string>
#include <cstdint>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct qcurandstate ; 

struct QUDARAP_API QCurandState
{
    static const plog::Severity LEVEL ; 

    qcurandstate* cs ; 
    qcurandstate* d_cs ; 

    QCurandState(unsigned long long num=100u, unsigned long long seed=0u, unsigned long long offset=0u); 
    std::string desc() const ; 

};
