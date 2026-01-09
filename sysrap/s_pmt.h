#pragma once
/**
s_pmt.h
========

HMM : NOW THAT THE GAPS ARE NOT SO LARGE IS TEMPTING TO DIRECTLY USE PMTID AND HAVE ZEROS IN THE GAPS


Methods:

s_pmt::desc
    dump the basis num and offsets

s_pmt::check_pmtid
    asserts

s_pmt::in_range
    internal

s_pmt::id_CD_LPMT
s_pmt::id_CD_SPMT
s_pmt::id_WP_PMT
s_pmt::id_WP_ATM_LPMT
s_pmt::id_WP_ATM_MPMT
s_pmt::id_WP_WAL_PMT
    return true when *id:pmtid* is within corresponding ranges of the six pmt types

s_pmt::ix_CD_LPMT
s_pmt::ix_CD_SPMT
s_pmt::ix_WP_PMT
s_pmt::ix_WP_ATM_LPMT
s_pmt::ix_WP_ATM_MPMT
s_pmt::ix_WP_WAL_PMT
    return true when *ix:oldcontiguousidx* is within corresponding ranges of the six pmt types

s_pmt::oldcontiguousidx_from_pmtid
    return *ix:oldcontiguousidx* from *id:pmtid*

s_pmt::pmtid_from_oldcontiguousidx
    return *id:pmtid* from *ix:oldcontiguousidx*

s_pmt::contiguousidx_from_pmtid
    return *ix:contiguousidx* from *id:pmtid* (DIFFERS IN THE ORDERING OF PMT TYPES, WITH SPMT SHIFTED TO LAST NOT 2ND)


s_pmt::lpmtidx_from_pmtid
    return *iy:lpmtidx* from *pmtid* (-1 for SPMT) [*iy:lpmtidx*-IS-MISNOMER-ACTUALLY-ALL-PMT-EXCLUDING-SPMT]


s_pmt::iy_CD_PMT
s_pmt::iy_WP_PMT
s_pmt::iy_WP_ATM_LPMT
s_pmt::iy_WP_ATM_MPMT
s_pmt::iy_WP_WAL_PMT
s_pmt::iy_CD_SPMT
    return true when *iy:contiguousidx* is within corresponding ranges of six pmt types, SPMT last

s_pmt::pmtid_from_contiguousidx
    uses iy_ methods to find pmt type for *iy:contiguousidx* then uses offset and num of prior pmt types to return absolute pmtid

s_pmt::pmtid_from_lpmtidx (very similar to above, just with SPMT excluded : they yield -1)
    uses iy_ methods to find pmt type for *iy:lpmtidx* thne uses offfset and num of prior pmt types to return absolute pmtid


s_pmt::is_spmtid
    returns true for absolute *pmtid* within offset range of SPMT

s_pmt::is_spmtidx
    returns true for *pmtidx* in range 0:NUM_SPMT

s_pmt::pmtid_from_spmtidx
    returns absolute *pmtid* from *spmtidx*

s_pmt::spmtidx_from_pmtid
    returns *spmtidx* from absolute *pmtid*




Used from::

   qudarap/QPMT.hh
   qudarap/qpmt.h
   sysrap/SPMT.h


jcv CopyNumber::

        014 enum PMTID_OFFSET_DETSIM {
         15   kOFFSET_CD_LPMT=0,     kOFFSET_CD_LPMT_END=17612, // CD-LPMT
         16   kOFFSET_CD_SPMT=20000, kOFFSET_CD_SPMT_END=45600,  // CD-SPMT
         17   kOFFSET_WP_PMT=50000,  kOFFSET_WP_PMT_END=52400,  // WP-LPMT
         18   kOFFSET_WP_ATM_LPMT=52400,  kOFFSET_WP_ATM_LPMT_END=52748, //WP-Atmosphere-LPMT
         19   kOFFSET_WP_ATM_MPMT=53000,  kOFFSET_WP_ATM_MPMT_END=53600,  //WP-Atmosphere-MPMT (Medium, 8 inch)
         20   kOFFSET_WP_WAL_PMT=54000,  kOFFSET_WP_WAL_PMT_END=54005 //WP-Water-attenuation-length
         21
         22 };

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SPMT_FUNCTION __host__ __device__ __forceinline__
#else
#    define SPMT_FUNCTION inline
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cassert>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#endif

/**
NB for backward compat with an old branch, had to revert NUM_WP_ATM_MPMT to 0 from 600
https://code.ihep.ac.cn/JUNO/offline/junosw/-/merge_requests/1061
**/


/**
The below WITH_MPMT line needs to be commented for use with older branches
to avoid runtime translation asserts
**/

#define WITH_MPMT 1

namespace s_pmt
{
    enum
    {
        NUM_CAT                = 3,
        NUM_LAYR               = 4,
        NUM_PROP               = 2,
        NUM_CD_LPMT            = 17612,
        NUM_SPMT               = 25600,
        NUM_WP                 = 2400,
        NUM_WP_ATM_LPMT        = 348,
        NUM_WP_ATM_MPMT_ALREADY = 600,      // unfortunately some branches partially have the MPMT
#ifdef WITH_MPMT
        NUM_WP_ATM_MPMT        = 600,      // 0:newer ones have MPMT
#else
        NUM_WP_ATM_MPMT        = 0,        // 0:older branches lack MPMT
#endif
        NUM_WP_WAL_PMT         = 5,
        OFFSET_CD_LPMT         = 0,
        OFFSET_CD_LPMT_END     = 17612,
        OFFSET_CD_SPMT         = 20000,
        OFFSET_CD_SPMT_END     = 45600,
        OFFSET_WP_PMT          = 50000,
        OFFSET_WP_PMT_END      = 52400,
        OFFSET_WP_ATM_LPMT     = 52400,
        OFFSET_WP_ATM_LPMT_END = 52748,
        OFFSET_WP_ATM_MPMT     = 53000,
        OFFSET_WP_ATM_MPMT_END = 53600,
        OFFSET_WP_WAL_PMT      = 54000,
        OFFSET_WP_WAL_PMT_END  = 54005
    };


    enum
    {
        NUM_CD_LPMT_AND_WP   = NUM_CD_LPMT + NUM_WP  +               0 +              0  +              0 +        0,   // 17612 + 2400 +   0 +      0  + 0  +    0 = 20012
        NUM_LPMTIDX          = NUM_CD_LPMT + NUM_WP  + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT +        0,   // 17612 + 2400 + 348 + {0,600} + 5  +    0 = {20365,20965}
        NUM_ALL              = NUM_CD_LPMT + NUM_WP  + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT + NUM_SPMT,   // 17612 + 2400 + 348 + {0,600} + 5 + 25600 = {45965,46565}
        NUM_ALL_EXCEPT_MPMT  = NUM_ALL - NUM_WP_ATM_MPMT,                                                               // 46565 - 600 = 45965
        NUM_CONTIGUOUSIDX    = NUM_CD_LPMT + NUM_WP  + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT         + NUM_WP_WAL_PMT + NUM_SPMT,   // 17612 + 2400 + 348 + {0,600} + 5 + 25600 = {45965,46565}
        NUM_OLDCONTIGUOUSIDX = NUM_CD_LPMT + NUM_WP  + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT_ALREADY + NUM_WP_WAL_PMT + NUM_SPMT,   // 17612 + 2400 + 348 + 600     + 5 + 25600 =        46565
    };


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   SPMT_FUNCTION std::string desc()
   {
       std::stringstream ss ;
       ss
          << "[s_pmt::desc\n"
#ifdef WITH_MPMT
          << " WITH_MPMT\n"
#else
          << " NOT:WITH_MPMT\n"
#endif
          << std::setw(25) << "NUM_CAT"                   << std::setw(7) << NUM_CAT                  << "\n"
          << std::setw(25) << "NUM_LAYR"                  << std::setw(7) << NUM_LAYR                 << "\n"
          << std::setw(25) << "NUM_PROP"                  << std::setw(7) << NUM_PROP                 << "\n"
          << std::setw(25) << "NUM_CD_LPMT"               << std::setw(7) << NUM_CD_LPMT              << "\n"
          << std::setw(25) << "NUM_SPMT"                  << std::setw(7) << NUM_SPMT                 << "\n"
          << std::setw(25) << "NUM_WP"                    << std::setw(7) << NUM_WP                   << "\n"
          << std::setw(25) << "NUM_WP_ATM_LPMT"           << std::setw(7) << NUM_WP_ATM_LPMT          << "\n"
          << std::setw(25) << "NUM_WP_ATM_MPMT_ALREADY"   << std::setw(7) << NUM_WP_ATM_MPMT_ALREADY  << "\n"
          << std::setw(25) << "NUM_WP_ATM_MPMT"           << std::setw(7) << NUM_WP_ATM_MPMT          << "\n"
          << std::setw(25) << "NUM_WP_WAL_PMT"            << std::setw(7) << NUM_WP_WAL_PMT           << "\n"
          << std::setw(25) << "NUM_CD_LPMT_AND_WP"        << std::setw(7) << NUM_CD_LPMT_AND_WP       << "\n"
          << std::setw(25) << "NUM_ALL"                   << std::setw(7) << NUM_ALL                  << "\n"
          << std::setw(25) << "NUM_LPMTIDX"               << std::setw(7) << NUM_LPMTIDX              << "\n"
          << std::setw(25) << "NUM_CONTIGUOUSIDX"         << std::setw(7) << NUM_CONTIGUOUSIDX        << "\n"
          << std::setw(25) << "NUM_OLDCONTIGUOUSIDX"      << std::setw(7) << NUM_OLDCONTIGUOUSIDX     << "\n"
          << std::setw(25) << "OFFSET_CD_LPMT"            << std::setw(7) << OFFSET_CD_LPMT           << "\n"
          << std::setw(25) << "OFFSET_CD_LPMT_END"        << std::setw(7) << OFFSET_CD_LPMT_END       << "\n"
          << std::setw(25) << "OFFSET_CD_SPMT"            << std::setw(7) << OFFSET_CD_SPMT           << "\n"
          << std::setw(25) << "OFFSET_CD_SPMT_END"        << std::setw(7) << OFFSET_CD_SPMT_END       << "\n"
          << std::setw(25) << "OFFSET_WP_PMT"             << std::setw(7) << OFFSET_WP_PMT            << "\n"
          << std::setw(25) << "OFFSET_WP_PMT_END"         << std::setw(7) << OFFSET_WP_PMT_END        << "\n"
          << std::setw(25) << "OFFSET_WP_ATM_LPMT"        << std::setw(7) << OFFSET_WP_ATM_LPMT       << "\n"
          << std::setw(25) << "OFFSET_WP_ATM_LPMT_END"    << std::setw(7) << OFFSET_WP_ATM_LPMT_END   << "\n"
          << std::setw(25) << "OFFSET_WP_ATM_MPMT"        << std::setw(7) << OFFSET_WP_ATM_MPMT       << "\n"
          << std::setw(25) << "OFFSET_WP_ATM_MPMT_END"    << std::setw(7) << OFFSET_WP_ATM_MPMT_END   << "\n"
          << std::setw(25) << "OFFSET_WP_WAL_PMT"         << std::setw(7) << OFFSET_WP_WAL_PMT        << "\n"
          << std::setw(25) << "OFFSET_WP_WAL_PMT_END"     << std::setw(7) << OFFSET_WP_WAL_PMT_END    << "\n"
          << "]s_pmt::desc\n"
          ;

       std::string str = ss.str() ;
       return str ;
   }

    SPMT_FUNCTION void check_pmtid( int pmtid )
    {
        assert(  pmtid >= 0 && pmtid < OFFSET_WP_WAL_PMT_END );
        assert(!(pmtid >= OFFSET_CD_LPMT_END && pmtid < OFFSET_CD_SPMT ));
        assert(!(pmtid >= OFFSET_CD_SPMT_END && pmtid < OFFSET_WP_PMT ));

        assert( OFFSET_CD_LPMT_END - OFFSET_CD_LPMT == NUM_CD_LPMT );
        assert( OFFSET_CD_SPMT_END - OFFSET_CD_SPMT == NUM_SPMT );
        assert( OFFSET_WP_PMT_END  - OFFSET_WP_PMT  == NUM_WP );

#ifdef WITH_MPMT
        assert( OFFSET_WP_ATM_MPMT_END - OFFSET_WP_ATM_MPMT == NUM_WP_ATM_MPMT ) ;
#endif

        assert( OFFSET_WP_WAL_PMT_END - OFFSET_WP_WAL_PMT == NUM_WP_WAL_PMT ) ;

        assert( NUM_LPMTIDX          == NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT         + NUM_WP_WAL_PMT );
        assert( NUM_CONTIGUOUSIDX    == NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT         + NUM_WP_WAL_PMT + NUM_SPMT ) ;
        assert( NUM_ALL              == NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT         + NUM_WP_WAL_PMT + NUM_SPMT ) ;

        assert( NUM_OLDCONTIGUOUSIDX == NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT_ALREADY + NUM_WP_WAL_PMT + NUM_SPMT ) ;
        // oldcontiguousidx matches implicit pmtCat index, which includes the MPMT:600 even before fully impl 


    }




#endif





    SPMT_FUNCTION bool in_range( int ix, int ix0, int ix1 ){ return ix >= ix0 && ix < ix1 ; }

    // returns true when id argument is within the pmtid range of that type of pmt
    SPMT_FUNCTION bool id_CD_LPMT(     int id ){ return in_range( id, OFFSET_CD_LPMT     , OFFSET_CD_LPMT_END     ) ; }
    SPMT_FUNCTION bool id_CD_SPMT(     int id ){ return in_range( id, OFFSET_CD_SPMT     , OFFSET_CD_SPMT_END     ) ; }
    SPMT_FUNCTION bool id_WP_PMT(      int id ){ return in_range( id, OFFSET_WP_PMT      , OFFSET_WP_PMT_END      ) ; }
    SPMT_FUNCTION bool id_WP_ATM_LPMT( int id ){ return in_range( id, OFFSET_WP_ATM_LPMT , OFFSET_WP_ATM_LPMT_END ) ; }
    SPMT_FUNCTION bool id_WP_ATM_MPMT( int id ){ return in_range( id, OFFSET_WP_ATM_MPMT , OFFSET_WP_ATM_MPMT_END ) ; }
    SPMT_FUNCTION bool id_WP_WAL_PMT(  int id ){ return in_range( id, OFFSET_WP_WAL_PMT  , OFFSET_WP_WAL_PMT_END  ) ; }





    /**
    s_pmt::oldcontiguousidx_from_pmtid
    ----------------------------------

    *ix:oldcontiguousidx* follows the numerical ascending order of the absolute pmtid, but without gaps.

    Transform non-contiguous pmtid::

          +------------------+      +-----------------------------+       +--------+--------------+       +-------------+          +-------------+
          |     CD_LPMT      |      |     SPMT                    |       |  WP    | WP_ATM_LPMT  |       | WP_ATM_MPMT |          | WP_WAL_PMT  |
          |     17612        |      |     25600                   |       |  2400  |   348        |       |    600      |          |     5       |
          +------------------+      +-----------------------------+       +--------+--------------+       +-------------+          +-------------+
          ^                  ^      ^                             ^       ^        ^              ^       ^             ^          ^             ^
          0                  17612  20000                         45600   50000    52400          52748   53000         53600      54000         54005
          |                  |      |                             |       |        |              |       |             |          |             |
          |                  |      |                             |       |        |              |       |             |          |             |
          |                  |      |                             |       |        |              |       |             |          |             OFFSET_WP_WAL_PMT_END
          |                  |      |                             |       |        |              |       |             |          OFFSET_WP_WAL_PMT
          |                  |      |                             |       |        |              |       |             |
          |                  |      |                             |       |        |              |       |             OFFSET_WP_ATM_MPMT_END
          |                  |      |                             |       |        |              |       OFFSET_WP_ATM_MPMT
          |                  |      |                             |       |        |              |
          |                  |      |                             |       |        |              OFFSET_WP_ATM_LPMT_END
          |                  |      |                             |       |        OFFSET_WP_ATM_LPMT
          |                  |      |                             |       |        |
          |                  |      |                             |       |        OFFSET_WP_PMT_END
          |                  |      |                             |       OFFSET_WP_PMT
          |                  |      |                             |
          |                  |      |                             OFFSET_CD_SPMT_END
          |                  |      OFFSET_CD_SPMT
          |                  |
          |                  OFFSET_CD_LPMT_END
          OFFSET_CD_LPMT


          pmtid >= OFFSET_CD_LPMT_END && pmtid < OFFSET_CD_SPMT    # 1st gap
          pmtid >= OFFSET_CD_SPMT_END && pmtid < OFFSET_WP_PMT     # 2nd gap

          17612+25600+2400 = 45612

          20000 - 17612 = 2388   # first gap
          50000 - 45600 = 4400   # second gap
          2388 + 4400   = 6788   # total gap
          52400 - 45612 = 6788   # diff between max pmtid kOFFSET_WP_PMT_END and NUM_ALL


    Into oldcontiguousidx::

          +------------------+-----------------------------+--------+--------------+---------------+-------------+
          |     CD_LPMT      |     SPMT                    |  WP    | WP_ATM_LPMT  |  WP_ATM_MPMT  |  WP_WAL_PMT |
          |     17612        |     25600                   |  2400  |   348        |     600       |     5       |
          +------------------+-----------------------------+--------+--------------+---------------+-------------+
          ^                  ^                             ^        ^
          0                17612                         43212    45612

         17612 + 25600 = 43212


    NB the oldcontiguousidx corresponds to the ordering using by pmtCat (from _PMTParamData/m_pmt_categories),
    even in an older branch without MPMT fully impl they are already present in the pmtCat::


        In [27]: np.all( f.pmtCat[17612+25600+2400+348:17612+25600+2400+348+600][:,1] == 4 )  ## MPMT all 600 are cat:4
        Out[27]: np.True_

        In [28]: f.pmtCat[17612+25600+2400+348+600:17612+25600+2400+348+600+5]
        Out[28]:
        array([[54000,     3],
               [54001,     0],
               [54002,     0],
               [54003,     0],
               [54004,     0]], dtype=int32)

    So the implicit contiguous index of pmtCat follows the order, and included MPMT::

        CD-LPMT:17612
        CD-SPMT:25600
        WP-LPMT:2400
        WP-Atmosphere-LPMT:348
        WP-Atmosphere-MPMT:600
        WP-Water-attenuation-length:5

    **/


    // returns true when *ix:oldcontiguousidx* argument is within the oldcontinuousidx ranges
    SPMT_FUNCTION bool ix_CD_LPMT(     int ix ){ return in_range(ix, 0                                                                           , NUM_CD_LPMT           ) ; }
    SPMT_FUNCTION bool ix_CD_SPMT(     int ix ){ return in_range(ix, NUM_CD_LPMT                                                                 , NUM_CD_LPMT + NUM_SPMT) ; }
    SPMT_FUNCTION bool ix_WP_PMT(      int ix ){ return in_range(ix, NUM_CD_LPMT + NUM_SPMT                                                      , NUM_CD_LPMT + NUM_SPMT + NUM_WP)  ; }
    SPMT_FUNCTION bool ix_WP_ATM_LPMT( int ix ){ return in_range(ix, NUM_CD_LPMT + NUM_SPMT + NUM_WP                                             , NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT)  ; }
    SPMT_FUNCTION bool ix_WP_ATM_MPMT( int ix ){ return in_range(ix, NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT                           , NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT_ALREADY)  ; }
    SPMT_FUNCTION bool ix_WP_WAL_PMT(  int ix ){ return in_range(ix, NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT_ALREADY , NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT_ALREADY + NUM_WP_WAL_PMT ) ; }


    /**
    oldcontiguousidx_from_pmtid
    -----------------------------

    1. find PMT type from pmtid id_ methods
    2. subtract the PMT type offset to give local index within the type
    3. add the NUM of other types below this one in the order of PMT types

    **/


    SPMT_FUNCTION int oldcontiguousidx_from_pmtid( int id )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        check_pmtid(id);
#endif

        int ix = -1 ;

        if(      id_CD_LPMT(id) )    ix = id ;
        else if( id_CD_SPMT(id) )    ix = id - OFFSET_CD_SPMT     + NUM_CD_LPMT ;
        else if( id_WP_PMT(id)  )    ix = id - OFFSET_WP_PMT      + NUM_CD_LPMT + NUM_SPMT ;
        else if( id_WP_ATM_LPMT(id)) ix = id - OFFSET_WP_ATM_LPMT + NUM_CD_LPMT + NUM_SPMT + NUM_WP ;
        else if( id_WP_ATM_MPMT(id)) ix = id - OFFSET_WP_ATM_MPMT + NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT ;
        else if( id_WP_WAL_PMT(id))  ix = id - OFFSET_WP_WAL_PMT  + NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT_ALREADY ;

        return ix ;
    }

    /**
    pmtid_from_oldcontiguousidx
    ----------------------------

    1. find PMT type from *ix:oldcontiguousidx* using ix_ methods
    2. get local index within the PMT type by subtracting total NUM of PMTs for PMT types prior to this PMT type in the order of PMT types
    3. add the offset for this PMT type to give the absolute PMT id

    **/

    SPMT_FUNCTION int pmtid_from_oldcontiguousidx( int ix )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( ix >= 0 && ix < NUM_OLDCONTIGUOUSIDX );
#endif

        int id = -1 ;

        if(      ix_CD_LPMT(ix) )     id = OFFSET_CD_LPMT     + ix ;
        else if( ix_CD_SPMT(ix) )     id = OFFSET_CD_SPMT     + ix - ( NUM_CD_LPMT ) ;
        else if( ix_WP_PMT(ix) )      id = OFFSET_WP_PMT      + ix - ( NUM_CD_LPMT + NUM_SPMT ) ;
        else if( ix_WP_ATM_LPMT(ix) ) id = OFFSET_WP_ATM_LPMT + ix - ( NUM_CD_LPMT + NUM_SPMT + NUM_WP ) ;
        else if( ix_WP_ATM_MPMT(ix) ) id = OFFSET_WP_ATM_MPMT + ix - ( NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT )  ;
        else if( ix_WP_WAL_PMT(ix) )  id = OFFSET_WP_WAL_PMT  + ix - ( NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT_ALREADY ) ;

        return id ;
    }

    /**
    s_pmt::contiguousidx_from_pmtid( int pmtid )
    ---------------------------------------------


    Transform non-contiguous pmtid::

          +------------------+      +-----------------------------+       +--------+--------------+       +-------------+          +-------------+
          |     CD_LPMT      |      |     SPMT                    |       |  WP    | WP_ATM_LPMT  |       | WP_ATM_MPMT |          | WP_WAL_PMT  |
          |     17612        |      |     25600                   |       |  2400  |   348        |       |    600      |          |     5       |
          +------------------+      +-----------------------------+       +--------+--------------+       +-------------+          +-------------+
          ^                  ^      ^                             ^       ^        ^              ^       ^             ^          ^             ^
          0                  17612  20000                         45600   50000    52400          52748   53000         53600      54000         54005
          |                  |      |                             |       |        |              |       |             |          |             |
          |                  |      |                             |       |        |              |       |             |          |             |
          |                  |      |                             |       |        |              |       |             |          |             OFFSET_WP_WAL_PMT_END
          |                  |      |                             |       |        |              |       |             |          OFFSET_WP_WAL_PMT
          |                  |      |                             |       |        |              |       |             |
          |                  |      |                             |       |        |              |       |             OFFSET_WP_ATM_MPMT_END
          |                  |      |                             |       |        |              |       OFFSET_WP_ATM_MPMT
          |                  |      |                             |       |        |              |
          |                  |      |                             |       |        |              OFFSET_WP_ATM_LPMT_END
          |                  |      |                             |       |        OFFSET_WP_ATM_LPMT
          |                  |      |                             |       |        |
          |                  |      |                             |       |        OFFSET_WP_PMT_END
          |                  |      |                             |       OFFSET_WP_PMT
          |                  |      |                             |
          |                  |      |                             OFFSET_CD_SPMT_END
          |                  |      OFFSET_CD_SPMT
          |                  |
          |                  OFFSET_CD_LPMT_END
          OFFSET_CD_LPMT



          pmtid >= OFFSET_CD_LPMT_END && pmtid < OFFSET_CD_SPMT    # 1st gap
          pmtid >= OFFSET_CD_SPMT_END && pmtid < OFFSET_WP_PMT     # 2nd gap

          17612+25600+2400 = 45612

          20000 - 17612 = 2388   # first gap
          50000 - 45600 = 4400   # second gap
          2388 + 4400   = 6788   # total gap
          52400 - 45612 = 6788   # diff between max pmtid kOFFSET_WP_PMT_END and NUM_ALL


    Into contiguousidx::

          +------------------+---------+----------------+----------------+---------------+---------------+
          |     CD_LPMT      |   WP    |  WP_ATM_LPMT   |   WP_ATM_MPMT  |   WP_WAL_PMT  |  SPMT         |
          |     17612        |   2400  |    348         |      600       |        5      |  25600        |
          +------------------+---------+----------------+------------------------------------------------+
          ^                  ^         ^
          0                17612     20012

         17612 + 2400 = 20012
         17612 + 2400 + 25600 = 45612


     The ordering CD_LPMT, WP, SPMT must match that used in::

         PMTSimParamSvc::init_all_pmtID_qe_scale "jcv PMTSimParamSvc"
         [MPMT ARE NOT INCLUDED IN THAT IN OLDER BRANCHES EG yupd_bottompipe_adjust]

    **/


    SPMT_FUNCTION int contiguousidx_from_pmtid( int id )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        check_pmtid(id);
#endif

        int iy = -1 ;

        if(      id_CD_LPMT(id) )    iy = id ;
        else if( id_WP_PMT(id)  )    iy = id - OFFSET_WP_PMT      + NUM_CD_LPMT ;
        else if( id_WP_ATM_LPMT(id)) iy = id - OFFSET_WP_ATM_LPMT + NUM_CD_LPMT + NUM_WP ;
        else if( id_WP_ATM_MPMT(id)) iy = id - OFFSET_WP_ATM_MPMT + NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT ;
        else if( id_WP_WAL_PMT(id))  iy = id - OFFSET_WP_WAL_PMT  + NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT ;
        else if( id_CD_SPMT(id) )    iy = id - OFFSET_CD_SPMT     + NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT ;

        return iy ;
    }


    /**
    s_pmt::lpmtidx_from_pmtid (SPMT yields -1)
    -------------------------------------------

    Rearrange the non-contiguous pmtid::

          +------------------+      +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+       +--------+--------------+     +-------------+     +-------------+
          |     CD_LPMT      |      :     SPMT                    :       |  WP    | WP_ATM_LPMT  |     | WP_ATM_MPMT |     | WP_WAL_PMT  |
          |     17612        |      :     25600                   :       |  2400  |    348       |     |    600      |     |     5       |
          +------------------+      +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+       +--------+--------------+     +-------------+     +-------------+
          ^                  ^      ^                             ^       ^        ^              ^     ^             ^     ^             ^
          0                17612  20000                         45600     50000    52400        52748  53000        53600  54000        54005

          |                  |      |                             |       |        |
          |                  |      |                             |       |        OFFSET_WP_PMT_END
          |                  |      |                             |       OFFSET_WP_PMT
          |                  |      |                             |
          |                  |      |                             OFFSET_CD_SPMT_END
          |                  |      OFFSET_CD_SPMT
          |                  |
          |                  OFFSET_CD_LPMT_END
          OFFSET_CD_LPMT

    Into a contiguous 0-based index lpmtidx with SPMT excluded::

          +------------------+--------+----------------+---------------+-------------+
          |     CD_LPMT      |  WP    |   WP_ATM_LPMT  |  WP_ATM_MPMT  | WP_WAL_PMT  |
          |     17612        | 2400   |       348      |     600       |     5       |
          +------------------+--------+----------------+---------------+-------------+
          ^                  ^        ^                ^               ^             ^
          0                 17612   20012           20360            20960         20965

          np.cumsum([17612, 2400, 348, 600, 5])
          array([17612, 20012, 20360, 20960, 20965])


     s_pmt::lpmtidx_from_pmtid is used for example from qpmt::get_lpmtid_stackspec_ce_acosf

    **/


    SPMT_FUNCTION int lpmtidx_from_pmtid( int id )
    {
        int iy = -1 ;

        if(      id_CD_LPMT(id) )    iy = id ;
        else if( id_WP_PMT(id)  )    iy = id - OFFSET_WP_PMT      + NUM_CD_LPMT ;
        else if( id_WP_ATM_LPMT(id)) iy = id - OFFSET_WP_ATM_LPMT + NUM_CD_LPMT + NUM_WP ;
        else if( id_WP_ATM_MPMT(id)) iy = id - OFFSET_WP_ATM_MPMT + NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT ;
        else if( id_WP_WAL_PMT(id))  iy = id - OFFSET_WP_WAL_PMT  + NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT ;

        return iy ;
    }



    /**
    iy_CD_LPMT/iy_WP_PMT/iy_WP_ATM_LPMT/iy_WP_ATM_MPMT/iy_WP_WAL_PMT/iy_CD_SPMT + pmtid_from_contiguousidx
    --------------------------------------------------------------------------------------------------------

    iy_methods returns true when iy "continuousidx" argument is within ranges with ordering and NUM assumptions for the six PMT types:

    * CD_LPMT, WP_PMT, WP_ATM_LPMT, WP_ATM_MPMT, WP_WAL_PMT, CD_SPMT

    *pmtid_from_contiguousidx* converts iy "continuousidx" into non-contiguous absolute standard pmtid
    using OFFSET for each PMT type and following the PMT type ordering and NUM assumptions

    **/


    SPMT_FUNCTION bool iy_CD_LPMT(     int iy ){ return in_range(iy, 0                                                                         , NUM_CD_LPMT ) ; }
    SPMT_FUNCTION bool iy_WP_PMT(      int iy ){ return in_range(iy, NUM_CD_LPMT                                                               , NUM_CD_LPMT + NUM_WP ) ; }
    SPMT_FUNCTION bool iy_WP_ATM_LPMT( int iy ){ return in_range(iy, NUM_CD_LPMT + NUM_WP                                                      , NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT) ; }
    SPMT_FUNCTION bool iy_WP_ATM_MPMT( int iy ){ return in_range(iy, NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT                                    , NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT) ; }
    SPMT_FUNCTION bool iy_WP_WAL_PMT(  int iy ){ return in_range(iy, NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT                  , NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT ) ; }
    SPMT_FUNCTION bool iy_CD_SPMT(     int iy ){ return in_range(iy, NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT , NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT + NUM_SPMT) ; }

    SPMT_FUNCTION int pmtid_from_contiguousidx( int iy )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( iy >= 0 && iy < NUM_CONTIGUOUSIDX );
#endif

        int id = -1 ;

        if(      iy_CD_LPMT(iy) )     id = OFFSET_CD_LPMT     + iy ;
        else if( iy_WP_PMT(iy) )      id = OFFSET_WP_PMT      + iy - ( NUM_CD_LPMT ) ;
        else if( iy_WP_ATM_LPMT(iy) ) id = OFFSET_WP_ATM_LPMT + iy - ( NUM_CD_LPMT + NUM_WP ) ;
        else if( iy_WP_ATM_MPMT(iy) ) id = OFFSET_WP_ATM_MPMT + iy - ( NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT )  ;
        else if( iy_WP_WAL_PMT(iy) )  id = OFFSET_WP_WAL_PMT  + iy - ( NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT ) ;
        else if( iy_CD_SPMT(iy) )     id = OFFSET_CD_SPMT     + iy - ( NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT ) ;

        return id ;
    }




    /**
    s_pmt::pmtid_from_lpmtidx   (NB very similar to s_pmt::pmtid_from_contiguousidx but with SPMT excluded)
    ------------------------------------------------------------------------------------------------------------


    Convert a contiguous 0-based index lpmtidx with SPMT excluded::

          +------------------+--------+----------------+----------------+--------------+~~~~~~~~~~~~~~~EXCLUDED~~~~~~
          |     CD_LPMT      |  WP    |   WP_ATM_LPMT  |  WP_ATM_MPMT   |   WP_WAL_PMT |      SPMT                  :
          |     17612        | 2400   |      348       |    600         |      5       |       25600                :
          +------------------+--------+----------------+----------------+--------------+~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
          ^                  ^        ^                ^                ^              ^                            ^
          0                 17612   20012            20360             20960         20965                        46565

                            |        |                 |                |              |
                            |        |                 |                |              NUM_CD_LPMT+NUM_WP+NUM_WP_ATM_PMT+NUM_WP_ATM_MPMT+NUM_WP_WAL_PMT
                            |        |                 |                NUM_CD_LPMT+NUM_WP+NUM_WP_ATM_PMT+NUM_WP_ATM_MPMT
                            |        |                NUM_CD_LPMT+NUM_WP+NUM_WP_ATM_LPMT
                            |       NUM_CD_LPMT+NUM_WP
                            NUM_CD_LPMT

    np.cumsum([17612,2400,348,600,5,25600])
    array([17612, 20012, 20360, 20960, 20965, 46565])



    Into the the non-contiguous lpmtid::

          +------------------+                                            +--------+--------------+     +-------------+     +-------------+
          |     CD_LPMT      |                                            |  WP    | WP_ATM_LPMT  |     | WP_ATM_MPMT |     | WP_WAL_PMT  |
          |     17612        |                                            |  2400  |    348       |     |    600      |     |     5       |
          +------------------+                                            +--------+--------------+     +-------------+     +-------------+
          ^                  ^      ^                             ^       ^        ^              ^     ^             ^     ^             ^
          0                17612  20000                         45600     50000    52400        52748  53000        53600  54000        54005
                                                                          |
                                                                          OFFSET_WP_PMT


   OR -1 for invalid lpmtidx arguments such as from SPMT.

          17612+25600+2400 = 45612

    This is used from SPMT::init_lcqs

    The assumptions made are:

    1. iy spans the range of expected PMT types
    2. ordering of PMT types for the input index is as expected


    **/


    SPMT_FUNCTION int pmtid_from_lpmtidx( int iy )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( iy >= 0 && iy < NUM_LPMTIDX  );
#endif

        int id = -1 ;

        if(      iy_CD_LPMT(iy) )     id = OFFSET_CD_LPMT     + iy ;
        else if( iy_WP_PMT(iy) )      id = OFFSET_WP_PMT      + iy - ( NUM_CD_LPMT ) ;
        else if( iy_WP_ATM_LPMT(iy) ) id = OFFSET_WP_ATM_LPMT + iy - ( NUM_CD_LPMT + NUM_WP ) ;
        else if( iy_WP_ATM_MPMT(iy) ) id = OFFSET_WP_ATM_MPMT + iy - ( NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT )  ;
        else if( iy_WP_WAL_PMT(iy) )  id = OFFSET_WP_WAL_PMT  + iy - ( NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT ) ;

        return id ;
    }







    /**
    s_pmt::is_spmtid
     ----------------

    Used for example from SPMT::get_s_qescale_from_spmtid

    **/

    SPMT_FUNCTION bool is_spmtid( int pmtid )
    {
        return pmtid >= OFFSET_CD_SPMT && pmtid < OFFSET_CD_SPMT_END ;
    }
    SPMT_FUNCTION bool is_spmtidx( int pmtidx )
    {
        return pmtidx >= 0 && pmtidx < NUM_SPMT ;
    }


    SPMT_FUNCTION int pmtid_from_spmtidx( int spmtidx )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( spmtidx >= 0 && spmtidx < NUM_SPMT );
#endif
        return spmtidx + OFFSET_CD_SPMT ;
    }

    SPMT_FUNCTION int spmtidx_from_pmtid( int pmtid )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( pmtid >= OFFSET_CD_SPMT && pmtid < OFFSET_CD_SPMT_END );
#endif
        return pmtid - OFFSET_CD_SPMT ;
    }

}

