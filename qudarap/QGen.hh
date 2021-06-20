#pragma once

/**
QGen
====

**/

#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct QRng ; 

struct QUDARAP_API QGen
{
    static const plog::Severity LEVEL ; 

    const QRng*   rng ; 

    QGen(); 
    void generate(float* dst, unsigned num_gen); 
    void dump(float* dst, unsigned num_gen); 
};


