#pragma once
/**
SCU_BufferView.h
=================

Able to *upload* data from multiple NP arrays into a 
contiguous GPU side buffer and record item counts from each 
input array into the item vector allowing access to the 
GPU side device pointers for the separate input array data. 

TODO: investigate alignment performance impact for different T: float/float3/float4
with and without padding 

**/

#include "NP.hh"
#include "CUDA_CHECK.h"

template <typename T>
struct SCU_BufferView
{
    T* data = nullptr ; 
    std::vector<size_t> item ;   // HMM "value" more appropriate 

    void upload(   const std::vector<const NP*>& aa ); 
    void hostcopy( const std::vector<const NP*>& aa ); 

    std::string hostdump(size_t part) const ; 
    std::string hostdump() const ; 

    std::string devdump(size_t part) const ; 
    std::string devdump() const ; 

    size_t num_part() const ;
    size_t item_total() const ; 
    size_t item_offset(size_t part) const ; 
    size_t item_num(   size_t part) const ; 

    T* _pointer(size_t part) const ; 
    CUdeviceptr pointer(size_t part) const ; 
    void free() ; 

    std::string desc() const ; 
    std::string descItem() const ; 
};


/**
SCU_BufferView::hostcopy
--------------------------

Hostside malloc and copy from arrays into BufferView.
Mainly for testing before using for copies to device. 

**/

template <typename T>
inline void SCU_BufferView<T>::hostcopy( const std::vector<const NP*>& aa )
{
    assert( item.size() == 0 );    

    int num_a = aa.size() ; 
    for(int i=0 ; i < num_a ; i++) item.push_back( aa[i]->num_items() );  
    size_t tot_bytes = item_total()*sizeof(T) ; 

    data = (T*)malloc( tot_bytes );
 
    for(int i=0 ; i < num_a ; i++) memcpy( _pointer(i), aa[i]->cvalues<T>(), aa[i]->arr_bytes() );  
}

/**
SCU_BufferView::upload
------------------------

**/

template <typename T>
inline void SCU_BufferView<T>::upload( const std::vector<const NP*>& aa )
{
    assert( item.size() == 0 );    

    int num_a = aa.size() ; 
    for(int i=0 ; i < num_a ; i++) item.push_back( aa[i]->num_values() );  
    size_t tot_bytes = item_total()*sizeof(T) ; 

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &data ), tot_bytes )); 

    size_t tot_arr_bytes = 0 ; 

    for(int i=0 ; i < num_a ; i++) 
    {
        const NP* a = aa[i] ;
        CUdeviceptr d = pointer(i); 

        size_t arr_bytes = a->arr_bytes() ; 
        tot_arr_bytes += arr_bytes ;
        assert( tot_arr_bytes <= tot_bytes ); 
 
        CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d ), a->cvalues<T>(), arr_bytes, cudaMemcpyHostToDevice )); 
    }
}


/**
SCU_BufferView::hostdump
--------------------------

*data* assumed to be a valid host pointer, eg after using *hostcopy* 

**/


template <typename T>
inline std::string SCU_BufferView<T>::hostdump(size_t part) const 
{
    size_t num = item_num(part); 
    const T* p = _pointer(part); 
    std::stringstream ss ;
    ss 
       << "[SCU_BufferView::hostdump"
       << " part " << part 
       << " num " << num
       << "\n"  
       ;

    for(size_t i=0 ; i < num ; i++) ss << p[i] << "\n" ; 
 
    ss << "]SCU_BufferView::hostdump" ; 
    std::string str = ss.str(); 
    return str ; 
}

template <typename T>
inline std::string SCU_BufferView<T>::hostdump() const 
{
    std::stringstream ss ;
    for(size_t i=0 ; i < item.size() ; i++) ss << hostdump(i) ; 
    std::string str = ss.str(); 
    return str ; 
}






/**
SCU_BufferView::devdump
-------------------------

*data* assumed to be a valid dev pointer, eg after using *upload* 

**/

template <typename T>
inline std::string SCU_BufferView<T>::devdump(size_t part) const 
{
    size_t num = item_num(part); 
    CUdeviceptr ptr = pointer(part); 

    std::vector<T> tmp(num) ; 
    T* tt = tmp.data() ;
   
    CUDA_CHECK( cudaMemcpy( tt, reinterpret_cast<void*>(ptr), sizeof(T)*num, cudaMemcpyDeviceToHost )); 

    std::stringstream ss ;
    ss 
       << "[SCU_BufferView::devdump"
       << " part " << part 
       << " num " << num
       << "\n"  
       ;

    for(size_t i=0 ; i < num ; i++) ss << tmp[i] << "\n" ; 
 
    ss << "]SCU_BufferView::devdump \n" ; 
    std::string str = ss.str(); 
    return str ; 
}

template <typename T>
inline std::string SCU_BufferView<T>::devdump() const 
{
    std::stringstream ss ;
    for(size_t i=0 ; i < item.size() ; i++) ss << devdump(i) ; 
    std::string str = ss.str(); 
    return str ; 
}

template <typename T>
inline size_t SCU_BufferView<T>::num_part() const 
{
    return item.size(); 
}

template <typename T>
inline size_t SCU_BufferView<T>::item_total() const
{
    size_t tot = 0 ; 
    for(size_t i=0 ; i < item.size() ; i++) tot += item[i] ; 
    return tot ; 
}

template <typename T>
inline size_t SCU_BufferView<T>::item_offset(size_t part) const
{
    assert( part < item.size() ); 
    size_t off = 0 ; 
    for(size_t i=0 ; i < part ; i++) off += item[i] ; 
    return off ; 
}

template <typename T>
inline size_t SCU_BufferView<T>::item_num(size_t part) const
{
    assert( part < item.size() ); 
    return item[part] ; 
}

template <typename T>
inline T* SCU_BufferView<T>::_pointer(size_t part) const
{
    assert( part < item.size() ); 
    size_t off = item_offset(part) ; 
    return  ( data + off ) ; 
}

template <typename T>
inline CUdeviceptr SCU_BufferView<T>::pointer(size_t part) const
{
    return (CUdeviceptr)(uintptr_t) _pointer(part) ;  
}

/**
SCU_BufferView::free
---------------------

*dat* assumed to be a device pointer, eg after *upload*

**/

template <typename T>
inline void SCU_BufferView<T>::free()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( data) ) );
    data = nullptr ; 
    item.clear(); 
}

template <typename T>
inline std::string SCU_BufferView<T>::desc() const
{
    std::stringstream ss ; 
    ss << "SCU_BufferView"
       << " (uintptr_t)data  0x" 
       << std::setw(9) << std::hex << (uintptr_t)data << std::dec
       << " sizeof(T) " << std::setw(5) << sizeof(T)
       << " item_total "  << std::setw(7) << item_total()
       << " num_part "    << std::setw(7) << item.size() 
       << " " << descItem() << "\n" ; 
       ;
    std::string str = ss.str(); 
    return str ; 
}

template <typename T>
inline std::string SCU_BufferView<T>::descItem() const
{
    std::stringstream ss ; 
    ss << "{" ; 
    for(int i=0 ; i < int(item.size()) ; i++) ss << item[i] << " " ; 
    ss << "}" ; 
    std::string str = ss.str(); 
    return str ; 
}


