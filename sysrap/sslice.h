#pragma once

#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

struct sslice
{
    int gs_start ; 
    int gs_stop ; 
    int ph_offset ; 
    int ph_count ; 

    bool matches(int start, int stop, int offset, int count ) const ; 

    static std::string Label() ; 
    std::string desc() const ; 
    static std::string Desc(const std::vector<sslice>& sl ); 
    static int TotalCount(const std::vector<sslice>& sl, int i0, int i1); 
    static void SetOffset(std::vector<sslice>& slice); 
}; 

inline bool sslice::matches(int start, int stop, int offset, int count ) const
{
    return gs_start == start && gs_stop == stop && ph_offset == offset && ph_count == count ; 
}

inline std::string sslice::Label()
{
    std::stringstream ss ;
    ss << "       "
       << " "
       << std::setw(4) << "star"
       << " "
       << std::setw(4) << "stop"
       << " "
       << std::setw(7) << "offset "
       << " "
       << std::setw(7) << "count "
       ;
    std::string str = ss.str() ;
    return str ; 
}
inline std::string sslice::desc() const
{
    std::stringstream ss ;
    ss << "sslice "
       << "{"
       << std::setw(4) << gs_start
       << ","
       << std::setw(4) << gs_stop
       << ","
       << std::setw(7) << ph_offset
       << ","
       << std::setw(7) << ph_count
       << "}"
       ;
    std::string str = ss.str() ;
    return str ; 
}

inline std::string sslice::Desc(const std::vector<sslice>& sl)
{
    std::stringstream ss ;
    for(int i=0 ; i < int(sl.size()) ; i++ ) ss << std::setw(3) << i << " : " << sl[i].desc() << "\n" ;  
    ss << std::setw(3) << " " << "   " << Label() << "\n" ; 

    std::string str = ss.str() ;
    return str ; 
}

inline int sslice::TotalCount(const std::vector<sslice>& slice, int i0, int i1)
{
    assert( i1 <= int(slice.size())) ; 
    int tot = 0 ; 
    for(int i=i0 ; i < i1 ; i++ ) tot += slice[i].ph_count ; 
    return tot ; 
}

inline void sslice::SetOffset(std::vector<sslice>& slice)
{
    for(int i=0 ; i < int(slice.size()) ; i++ ) slice[i].ph_offset = TotalCount(slice,0,i) ; 
}

