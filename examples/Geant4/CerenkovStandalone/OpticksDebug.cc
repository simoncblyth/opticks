#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "NP.hh"
#include "OpticksDebug.hh"

template<typename T>
OpticksDebug<T>::OpticksDebug(unsigned itemsize_, const char* name_)
    :
    itemsize(itemsize_), 
    name(strdup(name_))
{
}

template<typename T>
void OpticksDebug<T>::append( T x, const char* name )
{
    values.push_back(x);  
    if( names.size() <= itemsize ) names.push_back(name); 
} 
template<typename T>
void OpticksDebug<T>::append( unsigned x, unsigned y, const char* name )
{
    assert(0); 
}

template<>
void OpticksDebug<double>::append( unsigned x, unsigned y, const char* name )
{
    assert( sizeof(unsigned)*2 == sizeof(double) ); 
    DUU duu ; 
    duu.uu.x = x ; 
    duu.uu.y = y ;            // union trickery to place two unsigned into the slot of a double  
    append(duu.d, name); 
}


template<>
void OpticksDebug<float>::append( unsigned x, unsigned y, const char* name )
{
    assert( sizeof(uint16_t)*2 == sizeof(float) ); 
    FHH fhh ; 
    fhh.hh.x = x ;  // narrowing : potentially loosing info
    fhh.hh.y = y ;            
    append(fhh.f, name); 
}






template<typename T>
void OpticksDebug<T>::write(const char* dir, const char* reldir, unsigned nj, unsigned nk )
{
    bool expected_size = values.size() % itemsize == 0  ; 
    unsigned ni = values.size() / itemsize  ; 
    assert( nj*nk == itemsize ); 

    if(!expected_size)
    {
       std::cout 
           << " UNEXPECTED SIZE "
           << " values.size " << values.size()
           << " itemsize " << itemsize 
           << " ni " << ni
           << std::endl 
           ;
    }
    assert( expected_size ); 

    std::string stem = name ; 
    std::string npy = stem + ".npy" ; 
    std::string txt = stem + ".txt" ; 

    std::cout 
        << "OpticksDebug::write"
        << " ni " << ni 
        << " dir " << dir
        << " reldir " << reldir
        << std::endl
        ; 

    if( ni > 0 )
    {
        NP::Write(     dir, reldir, npy.c_str(), values.data(), ni, nj, nk ); 
        NP::WriteNames(dir, reldir, txt.c_str(), names, itemsize ); 
    }
}



template struct OpticksDebug<float>;
template struct OpticksDebug<double>;


