#pragma once

template <typename T> struct qselector ; 

struct sphoton ; 
struct sphoton_selector ; 


#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SU 
{
    template<typename T>
    static T* upload(const T* h, unsigned num_items ); 

    template<typename T>
    static void deprecated_select_copy_device_to_host( T** h, unsigned& num_select,  T* d, unsigned num_d, const qselector<T>& selector  ); 

    template<typename T>
    static unsigned count_if( const T* d, unsigned num_d,  const qselector<T>& selector ) ; 

    template<typename T>
    static T* device_alloc( unsigned num ); 

    template<typename T>
    static void device_zero( T* d, unsigned num ); 

    template<typename T>
    static void copy_if_device_to_device_presized( T* d_select, const T* d, unsigned num_d, const qselector<T>& selector ); 

    template<typename T>
    static void copy_device_to_host_presized( T* h, const T* d, unsigned num ); 



    static unsigned count_if_sphoton( const sphoton* d, unsigned num_d, const sphoton_selector& selector ); 

    static void copy_if_device_to_device_presized_sphoton( sphoton* d_select, const sphoton* d, unsigned num_d, const sphoton_selector& selector ); 

}; 




