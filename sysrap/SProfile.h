#pragma once
/**
SProfile.h
===========

For example used to collect timestamps from junoSD_PMT_v2::ProcessHits 
for persisting into .npy files for analysis, using NPX.h functionality

See tests/SProfile_test.cc for example of use. 

The static fixing of template parameter N is inconvenient when 
needing to change the N. But trying to do that dynamically for example 
with placement new would get complicated and would need in anycase 
an oversized buffer. Hence, just be pragmatic and fix N larger 
than typically needed (eg 16) so will then not have to change it very often.  

**/

#include <cstdint>
#include <vector>
#include <chrono>
#include "NPX.h"

template<int N>
struct SProfile
{
    uint64_t idx ; 
    uint64_t t[N] ; 

    static std::vector<SProfile<N>> RECORD ; 
    void add(){ RECORD.push_back(*this) ; }

    static uint64_t Now()
    {
        std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t0.time_since_epoch()).count() ; 
    }
    void zero(){ idx = 0 ; for(int i=0 ; i < N ; i++) t[i] = 0 ; }
    void stamp(int i){ t[i] = Now(); }

    static constexpr const char* NAME = "SProfile.npy" ; 
    static NP* Array(){ return NPX::ArrayFromVec<uint64_t,SProfile<N>>(RECORD,1+N); }
    static void Save(const char* dir, const char* reldir=nullptr){ NP* a = Array(); a->save(dir, reldir, NAME) ; } 
    static void Clear(){ RECORD.clear() ; }
}; 

