#pragma once
/**
SProfile.h
===========

For example used to collect timestamps from junoSD_PMT_v2::ProcessHits 
for persisting into .npy files for analysis, using NPX.h functionality

See tests/SProfile_test.cc for example of use. 

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

