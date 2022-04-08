#include "SU.hh"

#include "scuda.h"
#include "squad.h"

#include <thrust/device_ptr.h>
#include <thrust/copy.h>

template<typename T>
unsigned SU::select_count( T* d, unsigned num_d,  qselector<T>& selector )
{
    thrust::device_ptr<T> td(d);
    return thrust::count_if(td, td+num_d , selector );
}


template<typename T>
T* SU::upload(const T* h, unsigned num_items )
{
    T* d ;
    cudaMalloc(&d, num_items*sizeof(T));
    cudaMemcpy(d, h, num_items*sizeof(T), cudaMemcpyHostToDevice);
    return d ; 
}



/**
SU::select_copy_device_to_host
-------------------------------

This API is awkward because the number selected is not known when making the call.
For example it would be difficult to populate an NP array using this without 
making copies. 

**/

template<typename T>
void SU::select_copy_device_to_host( T** h, unsigned& num_select,  T* d, unsigned num_d, const qselector<T>& selector  )
{   
    thrust::device_ptr<T> td(d);
    num_select = thrust::count_if(td, td+num_d , selector );
    std::cout << " num_select " << num_select << std::endl ;
    
    T* d_select ;   
    cudaMalloc(&d_select,     num_select*sizeof(T));
    //cudaMemset(d_select, 0,   num_select*sizeof(T));
    thrust::device_ptr<T> td_select(d_select);
    
    thrust::copy_if(td, td+num_d , td_select, selector );
    
    *h = new T[num_select] ; 
    cudaMemcpy(*h, d_select, num_select*sizeof(T), cudaMemcpyDeviceToHost);
}



/**
SU::select_copy_device_to_host_presized
-----------------------------------------

The host array must be presized to fit the selection, do so using *select_count* with the same selector. 

**/

template<typename T>
void SU::select_copy_device_to_host_presized( T* h, T* d, unsigned num_d, const qselector<T>& selector, unsigned num_select  )
{
    thrust::device_ptr<T> td(d);

    T* d_select ;
    cudaMalloc(&d_select,     num_select*sizeof(T));
    //cudaMemset(d_select, 0,   num_select*sizeof(T));
    thrust::device_ptr<T> td_select(d_select);

    thrust::copy_if(td, td+num_d , td_select, selector );
    cudaMemcpy(h, d_select, num_select*sizeof(T), cudaMemcpyDeviceToHost);
}

template SYSRAP_API unsigned SU::select_count( quad4* , unsigned, qselector<quad4>& ); 
template SYSRAP_API quad4*   SU::upload(const quad4* , unsigned ); 
template SYSRAP_API void     SU::select_copy_device_to_host( quad4** h, unsigned& ,  quad4* , unsigned , const qselector<quad4>&  ); 
template SYSRAP_API void     SU::select_copy_device_to_host_presized( quad4*, quad4*, unsigned, const qselector<quad4>& , unsigned ); 

 



