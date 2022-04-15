#pragma once

template <typename T> struct qselector ; 

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SU 
{
    template<typename T>
    static T* upload(const T* h, unsigned num_items ); 

    template<typename T>
    static void select_copy_device_to_host( T** h, unsigned& num_select,  T* d, unsigned num_d, const qselector<T>& selector  ); 

    template<typename T>
    static unsigned select_count( T* d, unsigned num_d,  qselector<T>& selector ) ; 

    template<typename T>
    static void select_copy_device_to_host_presized( T* h, T* d, unsigned num_d, const qselector<T>& selector, unsigned num_select  ); 

}; 


