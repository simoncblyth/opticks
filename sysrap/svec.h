#pragma once

/**
svec.h : static vector<T> utilities 
=======================================

Header only replacement fort SVec.hh SVec.cc

**/

#include <string>
#include <vector>
#include <iostream>

template <typename T>
struct svec
{
    static std::string Desc(const char* label, const std::vector<T>& a, int width=7);    
    static void Dump(const char* label, const std::vector<T>& a );    
    static void Dump2(const char* label, const std::vector<T>& a );    
    static T MaxDiff(const std::vector<T>& a, const std::vector<T>& b, bool dump);    
    static int FindIndexOfValue( const std::vector<T>& a, T value, T tolerance ); 
    static int FindIndexOfValue( const std::vector<T>& a, T value ); 
    static void MinMaxAvg(const std::vector<T>& a, T& mn, T& mx, T& av) ; 
    static void MinMax(const std::vector<T>& a, T& mn, T& mx ) ; 
    static void Extract(std::vector<T>& a, const char* str, const char* ignore="(),[]") ; 

    static int Compare( const char* name, const std::vector<T>& a, const std::vector<T>& b, std::ostream* out ); 
    static int CompareBytes(const void* a, const void* b, unsigned num_bytes, std::ostream* out); 
};


#include <cassert>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <sstream>

#include "sstr.h"

template <typename T>
inline void svec<T>::Dump( const char* label, const std::vector<T>& a  )
{
    std::cout << std::setw(10) << label  ;
    for(unsigned i=0 ; i < a.size() ; i++) std::cout << std::setw(10) << a[i] << " " ; 
    std::cout << std::endl ; 
} 

template <typename T>
inline std::string svec<T>::Desc( const char* label, const std::vector<T>& a, int width  )
{
    std::stringstream ss ; 
    ss << std::setw(10) << label  ;
    for(unsigned i=0 ; i < a.size() ; i++) ss << std::setw(width) << a[i] << " " ; 
    return ss.str(); 
} 

template <typename T>
inline void svec<T>::Dump2( const char* label, const std::vector<T>& a  )
{
    std::cout << std::setw(10) << label ;
    std::copy( a.begin(), a.end(), std::ostream_iterator<float>(std::cout, " ")) ;
    std::cout << std::endl ; 
} 

template <typename T>
inline T svec<T>::MaxDiff(const std::vector<T>& a, const std::vector<T>& b, bool dump)
{
    assert( a.size() == b.size() );
    T mx = 0.f ;     
    for(unsigned i=0 ; i < a.size() ; i++)
    {
        T df = a[i] - b[i] ;   // std::abs ambiguous when T=unsigned 
        if( df < 0 ) df = -df ;  

        if(df > mx) mx = df ; 

        if(dump)
        std::cout 
            << " a " << a[i] 
            << " b " << b[i] 
            << " df " << df
            << " mx " << mx 
            << std::endl 
            ; 

    }
    return mx ; 
}


template <typename T>
inline int svec<T>::FindIndexOfValue(const std::vector<T>& a, T value, T tolerance)
{
    int idx = -1 ; 
    for(unsigned i=0 ; i < a.size() ; i++)
    {   
        //T df = std::abs(a[i] - value) ;   // std::abs ambiguous when T=unsigned 
        T df = a[i] - value ; 
        if(df < 0) df = -df ; 

        if(df < tolerance)
        {   
            idx = i ; 
            break ; 
        }   
    }           
    return idx ; 
}


template <typename T>
inline int svec<T>::FindIndexOfValue(const std::vector<T>& a, T value )
{
    size_t idx = std::distance( a.begin(), std::find( a.begin(), a.end(), value )) ; 
    return idx < a.size() ? idx : -1 ; 
}
 


template <typename T>
inline void svec<T>::MinMaxAvg(const std::vector<T>& t, T& mn, T& mx, T& av) 
{
    typedef typename std::vector<T>::const_iterator IT ;    
    IT mn_ = std::min_element( t.begin(), t.end()  );  
    IT mx_ = std::max_element( t.begin(), t.end()  );  
    double sum = std::accumulate(t.begin(), t.end(), T(0.) );   

    mn = *mn_ ; 
    mx = *mx_ ; 
    av = t.size() > 0 ? sum/T(t.size()) : T(-1.) ;   
}

template <typename T>
inline void svec<T>::MinMax(const std::vector<T>& t, T& mn, T& mx ) 
{
    typedef typename std::vector<T>::const_iterator IT ;    
    IT mn_ = std::min_element( t.begin(), t.end()  );  
    IT mx_ = std::max_element( t.begin(), t.end()  );  
    mn = *mn_ ; 
    mx = *mx_ ; 
}

template <typename T>
inline void svec<T>::Extract(std::vector<T>& a, const char* str0, const char* ignore ) 
{
    char swap = ' '; 
    const char* str1 = sstr::ReplaceChars(str0, ignore, swap); 
    std::stringstream ss(str1);  
    std::string s ; 
    T value ; 

    while(std::getline(ss, s, ' '))
    {
        if(strlen(s.c_str()) == 0 ) continue;  
        //std::cout << "[" << s << "]" << std::endl ; 
        
        std::stringstream tt(s);  
        tt >> value ; 
        a.push_back(value); 
    }
}











template<typename T>
inline int svec<T>::Compare( const char* name, const std::vector<T>& a, const std::vector<T>& b, std::ostream* out )
{
    int mismatch = 0 ;  

    bool size_match = a.size() == b.size() ; 
    if(!size_match && out) *out << "svec::Compare " << name << " size_match FAIL " << a.size() << " vs " << b.size() << "\n"    ;    
    if(!size_match) mismatch += 1 ;  
    if(!size_match) return mismatch ;  // below will likely crash if sizes are different 

    int data_match = memcmp( a.data(), b.data(), a.size()*sizeof(T) ) ;  
    if(data_match != 0 && out) *out << "svec::Compare " << name << " sizeof(T) " << sizeof(T) << " data_match FAIL " << "\n" ; 
    if(data_match != 0) mismatch += 1 ;  

    int byte_match = CompareBytes( a.data(), b.data(), a.size()*sizeof(T), out ) ;
    if(byte_match != 0 && out) *out << "svec::Compare " << name << " sizeof(T) " << sizeof(T) << " byte_match FAIL : num different bytes : " << byte_match << "\n" ; 
    if(byte_match != 0) mismatch += 1 ;  

    int num_item = a.size(); 
    for(int i=0 ; i < num_item ; i++)
    {
       int item_byte_match = CompareBytes( &a[i], &b[i], sizeof(T), out ); 
       if(item_byte_match == 0) continue ; 
       if(out) *out << "svec::Compare " << name << " sizeof(T) " << sizeof(T) << " num_item " << num_item << " item " << i << " item_byte_match FAIL : num different bytes : " << item_byte_match << "\n" ; 
       mismatch += 1 ;  
    }

    if(mismatch != 0 && out) *out << "svec::Compare " << name <<  " mismatch FAIL " << "\n" ;  
    return mismatch ; 
}

template<typename T>
inline int svec<T>::CompareBytes(const void* a, const void* b, unsigned num_bytes, std::ostream* out)
{
    const char* ca = (const char*)a ;
    const char* cb = (const char*)b ;
    int mismatch = 0 ;
    for(int i=0 ; i < int(num_bytes) ; i++ ) 
    {
        bool diff = ca[i] != cb[i] ; 
        if( diff )
        {
            if(out) *out << "svec::CompareBytes " << std::setw(3) << i << " [" <<  std::setw(3) << int(ca[i]) << "|" << std::setw(3) << int(cb[i]) << "]\n" ; 
            mismatch += 1 ;
        } 
    }
    return mismatch ;
}





template struct svec<int>;
template struct svec<unsigned>;
template struct svec<float>;
template struct svec<double>;


