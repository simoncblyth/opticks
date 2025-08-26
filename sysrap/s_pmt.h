#pragma once
/**
s_pmt.h
========

HMM : NOW THAT THE GAPS ARE NOT SO LARGE IS TEMPTING TO DIRECTLY USE PMTID AND HAVE ZEROS IN THE GAPS


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
        NUM_WP_ATM_MPMT        = 0,            // 600 in future
        NUM_WP_WAL_PMT         = 5,
        NUM_CD_LPMT_AND_WP     = 20012,   // 17612 + 2400 +   0 +   0 + 0 +     0 = 20012
        NUM_LPMTIDX            = 20365,   // 17612 + 2400 + 348 +   0 + 5  +    0 = 20365
        OLD_NUM_ALL            = 45612,   // 17612 + 2400 +   0 +   0 + 0 + 25600 = 45612
        NUM_ALL                = 45965,   // 17612 + 2400 + 348 +   0 + 5 + 25600 = 45965
        NUM_CONTIGUOUSIDX      = 45965,   // 17612 + 2400 + 348 +   0 + 5 + 25600 = 45965
        NUM_OLDCONTIGUOUSIDX   = 45965,   // 17612 + 2400 + 348 +   0 + 5 + 25600 = 45965
        FUTURE_NUM_ALL         = 46565,   // 17612 + 2400 + 348 + 600 + 5 + 25600 = 46565
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

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   SPMT_FUNCTION std::string desc()
   {
       std::stringstream ss ;
       ss
          << "[s_pmt::desc\n"
          << std::setw(25) << "NUM_CAT"                << std::setw(7) << NUM_CAT                << "\n"
          << std::setw(25) << "NUM_LAYR"               << std::setw(7) << NUM_LAYR               << "\n"
          << std::setw(25) << "NUM_PROP"               << std::setw(7) << NUM_PROP               << "\n"
          << std::setw(25) << "NUM_CD_LPMT"            << std::setw(7) << NUM_CD_LPMT            << "\n"
          << std::setw(25) << "NUM_SPMT"               << std::setw(7) << NUM_SPMT               << "\n"
          << std::setw(25) << "NUM_WP"                 << std::setw(7) << NUM_WP                 << "\n"
          << std::setw(25) << "NUM_WP_ATM_LPMT"        << std::setw(7) << NUM_WP_ATM_LPMT        << "\n"
          << std::setw(25) << "NUM_WP_ATM_MPMT"        << std::setw(7) << NUM_WP_ATM_MPMT        << "\n"
          << std::setw(25) << "NUM_WP_WAL_PMT"         << std::setw(7) << NUM_WP_WAL_PMT         << "\n"
          << std::setw(25) << "NUM_CD_LPMT_AND_WP"     << std::setw(7) << NUM_CD_LPMT_AND_WP     << "\n"
          << std::setw(25) << "NUM_ALL"                << std::setw(7) << NUM_ALL                << "\n"
          << std::setw(25) << "NUM_LPMTIDX"            << std::setw(7) << NUM_LPMTIDX            << "\n"
          << std::setw(25) << "NUM_CONTIGUOUSIDX"      << std::setw(7) << NUM_CONTIGUOUSIDX      << "\n"
          << std::setw(25) << "NUM_OLDCONTIGUOUSIDX"   << std::setw(7) << NUM_OLDCONTIGUOUSIDX   << "\n"
          << std::setw(25) << "OFFSET_CD_LPMT"         << std::setw(7) << OFFSET_CD_LPMT         << "\n"
          << std::setw(25) << "OFFSET_CD_LPMT_END"     << std::setw(7) << OFFSET_CD_LPMT_END     << "\n"
          << std::setw(25) << "OFFSET_CD_SPMT"         << std::setw(7) << OFFSET_CD_SPMT         << "\n"
          << std::setw(25) << "OFFSET_CD_SPMT_END"     << std::setw(7) << OFFSET_CD_SPMT_END     << "\n"
          << std::setw(25) << "OFFSET_WP_PMT"          << std::setw(7) << OFFSET_WP_PMT          << "\n"
          << std::setw(25) << "OFFSET_WP_PMT_END"      << std::setw(7) << OFFSET_WP_PMT_END      << "\n"
          << std::setw(25) << "OFFSET_WP_ATM_LPMT"     << std::setw(7) << OFFSET_WP_ATM_LPMT     << "\n"
          << std::setw(25) << "OFFSET_WP_ATM_LPMT_END" << std::setw(7) << OFFSET_WP_ATM_LPMT_END << "\n"
          << std::setw(25) << "OFFSET_WP_ATM_MPMT"     << std::setw(7) << OFFSET_WP_ATM_MPMT     << "\n"
          << std::setw(25) << "OFFSET_WP_ATM_MPMT_END" << std::setw(7) << OFFSET_WP_ATM_MPMT_END << "\n"
          << std::setw(25) << "OFFSET_WP_WAL_PMT"      << std::setw(7) << OFFSET_WP_WAL_PMT      << "\n"
          << std::setw(25) << "OFFSET_WP_WAL_PMT_END"  << std::setw(7) << OFFSET_WP_WAL_PMT_END  << "\n"
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
        //assert( OFFSET_WP_ATM_MPMT_END - OFFSET_WP_ATM_MPMT == NUM_WP_ATM_MPMT ) ; // NOT YET
        assert( OFFSET_WP_WAL_PMT_END - OFFSET_WP_WAL_PMT == NUM_WP_WAL_PMT ) ;

        assert( NUM_LPMTIDX          == NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT );
        assert( NUM_CONTIGUOUSIDX    == NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT + NUM_SPMT ) ;
        assert( NUM_OLDCONTIGUOUSIDX == NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT + NUM_SPMT ) ;
        assert( NUM_ALL              == NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT + NUM_SPMT ) ;

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

    Transform non-contiguous pmtid::

          +------------------+      +-----------------------------+       +--------+--------------+       +-------------+          +-------------+
          |     CD_LPMT      |      |     SPMT                    |       |  WP    | WP_ATM_LPMT  |       | WP_ATM_MPMT |          | WP_WAL_PMT  |
          |     17612        |      |     25600                   |       |  2400  |   348        |       |      0      |          |     5       |
          +------------------+      +-----------------------------+       +--------+--------------+       +-------------+          +-------------+
          ^                  ^      ^                             ^       ^        ^              ^       ^             ^          ^             ^
          0                  17612  20000                         45600   50000    52400          52748   53000         53000      54000         54005
          |                  |      |                             |       |        |              |       |             |          |             |
          |                  |      |                             |       |        |              |       |             |          |             |
          |                  |      |                             |       |        |              |       |             |          |             OFFSET_WP_WAL_PMT_END
          |                  |      |                             |       |        |              |       |             |          OFFSET_WP_WAL_PMT
          |                  |      |                             |       |        |              |       |             |
          |                  |      |                             |       |        |              |       |             "OFFSET_WP_ATM_MPMT_END"
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
          |     17612        |     25600                   |  2400  |   348        |      0        |     5       |
          +------------------+-----------------------------+--------+--------------+---------------+-------------+
          ^                  ^                             ^        ^
          0                17612                         43212    45612

         17612 + 25600 = 43212


    **/


    // returns true when ix argument is within the oldcontinuousidx ranges
    SPMT_FUNCTION bool ix_CD_LPMT(     int ix ){ return in_range(ix, 0                                                                   , NUM_CD_LPMT           ) ; }
    SPMT_FUNCTION bool ix_CD_SPMT(     int ix ){ return in_range(ix, NUM_CD_LPMT                                                         , NUM_CD_LPMT + NUM_SPMT) ; }
    SPMT_FUNCTION bool ix_WP_PMT(      int ix ){ return in_range(ix, NUM_CD_LPMT + NUM_SPMT                                              , NUM_CD_LPMT + NUM_SPMT + NUM_WP)  ; }
    SPMT_FUNCTION bool ix_WP_ATM_LPMT( int ix ){ return in_range(ix, NUM_CD_LPMT + NUM_SPMT + NUM_WP                                     , NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT)  ; }
    SPMT_FUNCTION bool ix_WP_ATM_MPMT( int ix ){ return in_range(ix, NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT                   , NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT)  ; }
    SPMT_FUNCTION bool ix_WP_WAL_PMT(  int ix ){ return in_range(ix, NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT , NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT ) ; }



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
        else if( id_WP_WAL_PMT(id))  ix = id - OFFSET_WP_WAL_PMT  + NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT ;

        return ix ;
    }

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
        else if( ix_WP_WAL_PMT(ix) )  id = OFFSET_WP_WAL_PMT  + ix - ( NUM_CD_LPMT + NUM_SPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT ) ;

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
          |     17612        |   2400  |    348         |        0       |        5      |  25600        |
          +------------------+---------+----------------+------------------------------------------------+
          ^                  ^         ^
          0                17612     20012

         17612 + 2400 = 20012
         17612 + 2400 + 25600 = 45612


     The ordering CD_LPMT, WP, SPMT must match that used in::

         PMTSimParamSvc::init_all_pmtID_qe_scale

    **/


    SPMT_FUNCTION int contiguousidx_from_pmtid( int id )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        check_pmtid(id);
#endif

        int iy = -1 ;

        if(      id_CD_LPMT(id) )    iy = id ;
        else if( id_CD_SPMT(id) )    iy = id - OFFSET_CD_SPMT     + NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT + NUM_WP_WAL_PMT ;
        else if( id_WP_PMT(id)  )    iy = id - OFFSET_WP_PMT      + NUM_CD_LPMT ;
        else if( id_WP_ATM_LPMT(id)) iy = id - OFFSET_WP_ATM_LPMT + NUM_CD_LPMT + NUM_WP ;
        else if( id_WP_ATM_MPMT(id)) iy = id - OFFSET_WP_ATM_MPMT + NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT ;
        else if( id_WP_WAL_PMT(id))  iy = id - OFFSET_WP_WAL_PMT  + NUM_CD_LPMT + NUM_WP + NUM_WP_ATM_LPMT + NUM_WP_ATM_MPMT ;

        return iy ;
    }


    /**
    s_pmt::lpmtidx_from_pmtid (SPMT yields -1)
    -------------------------------------------

    Rearrange the non-contiguous pmtid::

          +------------------+      +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+       +--------+--------------+     +--NOT YET----+     +-------------+
          |     CD_LPMT      |      :     SPMT                    :       |  WP    | WP_ATM_LPMT  |     | WP_ATM_MPMT |     | WP_WAL_PMT  |
          |     17612        |      :     25600                   :       |  2400  |    348       |     |      0      |     |     5       |
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
          |     17612        | 2400   |       348      |        0      |     5       |
          +------------------+--------+----------------+---------------+-------------+
          ^                  ^        ^                ^               ^             ^
          0                 17612   20012           20360            20360         20365

          np.cumsum([17612, 2400, 348, 0, 5])
          array([17612, 20012, 20360, 20360, 20365])


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




    // returns true when iy argument is within the continuousidx ranges
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


    Convert a contiguous 0-based index lpmtidx with SPMT excluded (NB currently NUM_WP_ATM_MPMT=0 as not yet in geometry)::

          +------------------+--------+----------------+----------------+--------------+~~~~~~~~~~~~~~~EXCLUDED~~~~~~
          |     CD_LPMT      |  WP    |   WP_ATM_LPMT  |  WP_ATM_MPMT   |   WP_WAL_PMT |      SPMT                  :
          |     17612        | 2400   |      348       |      0         |      5       |       25600                :
          +------------------+--------+----------------+----------------+--------------+~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
          ^                  ^        ^                ^                ^              ^                            ^
          0                 17612   20012            20360             20360         20365                        45965

                            |        |                 |                |              |
                            |        |                 |                |              NUM_CD_LPMT+NUM_WP+NUM_WP_ATM_PMT+NUM_WP_ATM_MPMT+NUM_WP_WAL_PMT
                            |        |                 |                NUM_CD_LPMT+NUM_WP+NUM_WP_ATM_PMT+NUM_WP_ATM_MPMT
                            |        |                NUM_CD_LPMT+NUM_WP+NUM_WP_ATM_LPMT
                            |       NUM_CD_LPMT+NUM_WP
                            NUM_CD_LPMT

    In [14]: np.cumsum([17612,2400,348,0,5,25600])
    Out[14]: array([17612, 20012, 20360, 20360, 20365, 45965])


    Into the the non-contiguous lpmtid::

          +------------------+                                            +--------+--------------+     +--NOT YET----+     +-------------+
          |     CD_LPMT      |                                            |  WP    | WP_ATM_LPMT  |     | WP_ATM_MPMT |     | WP_WAL_PMT  |
          |     17612        |                                            |  2400  |    348       |     |      0      |     |     5       |
          +------------------+                                            +--------+--------------+     +-------------+     +-------------+
          ^                  ^      ^                             ^       ^        ^              ^     ^             ^     ^             ^
          0                17612  20000                         45600     50000    52400        52748  53000        53600  54000        54005
                                                                          |
                                                                          OFFSET_WP_PMT


   OR -1 for invalid lpmtidx arguments such as from SPMT.

          17612+25600+2400 = 45612


    This is used from SPMT::init_lcqs

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







    /*
    SPMT_FUNCTION bool is_WP_( int pmtid )
    {
        return pmtid >= OFFSET_WP_PMT && pmtid < OFFSET_WP_PMT_END ;
    }
     */

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

