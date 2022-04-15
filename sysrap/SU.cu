#include "SU.hh"

#include "scuda.h"
#include "squad.h"

#include <thrust/device_ptr.h>
#include <thrust/copy.h>


template<typename T>
T* SU::upload(const T* h, unsigned num_items )
{
    T* d ;
    cudaMalloc(&d, num_items*sizeof(T));
    cudaMemcpy(d, h, num_items*sizeof(T), cudaMemcpyHostToDevice);
    return d ; 
}

template SYSRAP_API quad4*   SU::upload(const quad4* , unsigned ); 


/**
SU::deprecated_select_copy_device_to_host
--------------------------------------------

1. apply thrust::count_if to *d* with *selector* functor yielding *num_select* 
2. allocate *d_select* with num_select*sizeof(T) bytes
3. thrust::copy_if from *d* to *d_select* using the *selector* functor
4. host new T[num_select] allocation 
5. copies from *d_select* to the *num_select* host array *h* using the selector 

This API is deprecated because its awkward as the number selected is not known when making the call.
For example it would be difficult to populate an NP array using this without 
making copies. 

**/

template<typename T>
void SU::deprecated_select_copy_device_to_host( T** h, unsigned& num_select,  T* d, unsigned num_d, const qselector<T>& selector  )
{   
    thrust::device_ptr<T> td(d);
    num_select = thrust::count_if(td, td+num_d , selector );
    std::cout << " num_select " << num_select << std::endl ;
    
    T* d_select ;   
    cudaMalloc(&d_select,     num_select*sizeof(T));
    thrust::device_ptr<T> td_select(d_select);
    
    thrust::copy_if(td, td+num_d , td_select, selector );
    
    *h = new T[num_select] ; 
    cudaMemcpy(*h, d_select, num_select*sizeof(T), cudaMemcpyDeviceToHost);
}
template SYSRAP_API void     SU::deprecated_select_copy_device_to_host( quad4** h, unsigned& ,  quad4* , unsigned , const qselector<quad4>&  ); 


/**
SU::count_if
------------------

1. apply thrust::count_if to *d* with *selector* functor yielding *num_select* 

**/

template<typename T>
unsigned SU::count_if( const T* d, unsigned num_d,  qselector<T>& selector )
{
    thrust::device_ptr<const T> td(d);
    return thrust::count_if(td, td+num_d , selector );
}

template SYSRAP_API unsigned SU::count_if( const quad4* , unsigned, qselector<quad4>& ); 


/**
SU::device_alloc
-------------------

1. allocates *d* with num*sizeof(T) bytes

**/

template<typename T>
T* SU::device_alloc( unsigned num  )
{
    T* d ;
    cudaMalloc(&d,  num*sizeof(T));
    return d ; 
}
template SYSRAP_API quad4* SU::device_alloc( unsigned ); 


template<typename T>
void SU::device_zero( T* d, unsigned num )
{
    cudaMemset(d, 0,  num*sizeof(T));
}
template SYSRAP_API void SU::device_zero( quad4*, unsigned ); 


/**
SU::copy_if_device_to_device_presized
-----------------------------------------

The d_select array must be presized to fit the selection, determine the size using *select_count* with the same selector. 

2. thrust::copy_if from *d* to *d_select* using the selector functor
3. copies from *d_select* to the *num_select* presized host array *h* using the selector

**/

template<typename T>
void SU::copy_if_device_to_device_presized( T* d_select, const T* d, unsigned num_d, const qselector<T>& selector )
{
    thrust::device_ptr<const T> td(d);
    thrust::device_ptr<T> td_select(d_select);
    thrust::copy_if(td, td+num_d , td_select, selector );
}

template SYSRAP_API void SU::copy_if_device_to_device_presized( quad4*, const quad4*, unsigned, const qselector<quad4>& ); 


template<typename T>
void SU::copy_device_to_host_presized( T* h, const T* d, unsigned num  )
{
    cudaMemcpy(h, d, num*sizeof(T), cudaMemcpyDeviceToHost);
}
template SYSRAP_API void SU::copy_device_to_host_presized( quad4*, const quad4*, unsigned ); 



