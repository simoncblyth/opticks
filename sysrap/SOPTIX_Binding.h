#pragma once
/**
SOPTIX_Binding.h
==================

**/


#include <optix.h>


template <typename T>
struct SOPTIX_Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};


struct SOPTIX_RaygenData {} ;

struct SOPTIX_MissData
{
    float3 bg_color;
};

struct SOPTIX_TriMesh
{
    uint3*  indice ; 
    float3* vertex ; 
    float3* normal ; 
};

struct SOPTIX_HitgroupData 
{
    SOPTIX_TriMesh mesh ; 
};


typedef SOPTIX_Record<SOPTIX_RaygenData>   SOPTIX_RaygenRecord;
typedef SOPTIX_Record<SOPTIX_MissData>     SOPTIX_MissRecord;
typedef SOPTIX_Record<SOPTIX_HitgroupData> SOPTIX_HitgroupRecord;


