#pragma once

#ifdef WITH_PLOG
#include "plog/Severity.h"
#endif

#include <vector>
#include "CSG_API_EXPORT.hh"

struct CSG_API CU
{
#ifdef WITH_PLOG
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
