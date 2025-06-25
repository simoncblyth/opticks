#pragma once
/**
sslice.h : python style gs[start:stop] slice of genstep arrays/vectors
========================================================================

gs_start
   gs index starting the slice

gs_stop
   gs index stopping the slice, ie one beyond the last index

ph_offset
   total photons before this slice

ph_count
   total photons within this slice

**/

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
    std::string idx_desc(int idx) const ;
    static std::string Desc(const std::vector<sslice>& sl );

    static int TotalPhoton(const std::vector<sslice>& sl );
    static int TotalPhoton(const std::vector<sslice>& sl, int i0, int i1);

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
       << std::setw(8) << "start"
       << " "
       << std::setw(8) << "stop "
       << " "
       << std::setw(10) << "offset "
       << " "
       << std::setw(10) << "count "
       << " "
       << std::setw(10) << "count/M "
       ;
    std::string str = ss.str() ;
    return str ;
}
inline std::string sslice::desc() const
{
    std::stringstream ss ;
    ss << "sslice "
       << "{"
       << std::setw(8) << gs_start
       << ","
       << std::setw(8) << gs_stop
       << ","
       << std::setw(10) << ph_offset
       << ","
       << std::setw(10) << ph_count
       << "}"
       << std::setw(10) << std::fixed << std::setprecision(6) << double(ph_count)/1.e6
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string sslice::idx_desc(int idx) const
{
    std::stringstream ss ;
    ss << std::setw(4) << idx << " : " <<  desc() ;
    std::string str = ss.str() ;
    return str ;
}


inline std::string sslice::Desc(const std::vector<sslice>& sl)
{
    int tot_photon = TotalPhoton(sl) ;
    std::stringstream ss ;
    ss << "sslice::Desc"
       << " num_slice " << sl.size()
       << " TotalPhoton " << std::setw(10) << tot_photon
       << " TotalPhoton/M " << std::setw(10) << std::fixed << std::setprecision(6) << double(tot_photon)/1.e6
       << "\n"
        ;
    ss << std::setw(4) << "" << "   " << Label() << "\n" ;
    for(int i=0 ; i < int(sl.size()) ; i++ ) ss << sl[i].idx_desc(i) << "\n" ;
    ss << std::setw(4) << "" << "   " << Label() << "\n" ;
    std::string str = ss.str() ;
    return str ;
}

inline int sslice::TotalPhoton(const std::vector<sslice>& slice)
{
    return TotalPhoton(slice, 0, slice.size() );
}

/**
sslice::TotalPhoton
----------------------

NB i0, i1 use python style slice indexing, ie::

    In [4]: np.arange(10)[0:4]
    Out[4]: array([0, 1, 2, 3])

    In [5]: np.arange(10)[0:9]
    Out[5]: array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    In [6]: np.arange(10)[0:10]
    Out[6]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    In [7]: np.arange(10)
    Out[7]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    In [8]: np.arange(10)[0:0]
    Out[8]: array([], dtype=int64)


**/


inline int sslice::TotalPhoton(const std::vector<sslice>& slice, int i0, int i1)
{
    assert( i1 <= int(slice.size())) ;
    int tot = 0 ;
    for(int i=i0 ; i < i1 ; i++ ) tot += slice[i].ph_count ;
    return tot ;
}

inline void sslice::SetOffset(std::vector<sslice>& slice)
{
    for(int i=0 ; i < int(slice.size()) ; i++ ) slice[i].ph_offset = TotalPhoton(slice,0,i) ;
}

