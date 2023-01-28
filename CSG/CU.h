#pragma once
/**
CU.h : UploadArray/DownloadArray/UploadVec/DownloadVec 
========================================================

* used for CSGFoundry upload 

::

    epsilon:CSG blyth$ opticks-f CU.h 
    ./CSGOptiX/SBT.cc:#include "CU.h"
    ./CSG/CMakeLists.txt:    CU.h
    ./CSG/CSGPrimSpec.cc:#include "CU.h"
    ./CSG/tests/CSGPrimImpTest.cc:#include "CU.h"
    ./CSG/tests/CUTest.cc:#include "CU.h"
    ./CSG/CU.cc:#include "CU.h"
    ./CSG/CSGFoundry.cc:#include "CU.h"



**/


#ifdef WITH_SLOG
#include "plog/Severity.h"
#endif

#include <vector>
#include "CSG_API_EXPORT.hh"

struct CSG_API CU
{
#ifdef WITH_SLOG
    static const plog::Severity LEVEL ; 
#endif

    template <typename T>
    static T* UploadArray(const T* array, unsigned num_items ) ; 

    template <typename T>
    static T* DownloadArray(const T* array, unsigned num_items ) ; 


    template <typename T>
    static T* UploadVec(const std::vector<T>& vec);

    template <typename T>
    static void DownloadVec(std::vector<T>& vec, const T* d_array, unsigned num_items);

};
