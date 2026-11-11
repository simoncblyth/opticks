#pragma once

/**
CSGFoundryCompare.h
====================

**/

#include "plog/Severity.h"
#include <string>
#include <vector>
struct CSGFoundry ;
struct float4 ;

#include "CSG_API_EXPORT.hh"



struct CSG_API CSGFoundryCompare
{
    static const plog::Severity LEVEL ;

    static int Compare(const CSGFoundry* a , const CSGFoundry* b );
    static int WIP_CompareStruct(const CSGFoundry* a , const CSGFoundry* b );
    static std::string DescCompare( const CSGFoundry* a, const CSGFoundry* b );

    template<typename T>
    static int CompareVec( const char* name, const std::vector<T>& a, const std::vector<T>& b );


    template<typename T>
    static int CompareStruct( const char* name, const std::vector<T>& a, const std::vector<T>& b );

    static int CompareFloat4( const char* name, const std::vector<float4>& aa, const std::vector<float4>& bb );
    static bool Float4_IsDiff( const float4& a , const float4& b );


    static int CompareBytes(const void* a, const void* b, unsigned num_bytes);
};






