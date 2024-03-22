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

struct SOPTIX_EmptyData {}; 

typedef SOPTIX_Record<SOPTIX_EmptyData> SOPTIX_EmptyRecord;


