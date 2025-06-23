#pragma once
/**
s_pmt.h
========


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
#endif


namespace s_pmt
{
    enum
    {
        NUM_CAT            = 3,
        NUM_LAYR           = 4,
        NUM_PROP           = 2,
        NUM_CD_LPMT        = 17612,
        NUM_SPMT           = 25600,
        NUM_WP             = 2400,
        NUM_CD_LPMT_AND_WP = 20012,
        NUM_ALL            = 45612,
        OFFSET_CD_LPMT     = 0,
        OFFSET_CD_LPMT_END = 17612,
        OFFSET_CD_SPMT     = 20000,
        OFFSET_CD_SPMT_END = 45600,
        OFFSET_WP_PMT      = 50000,
        OFFSET_WP_PMT_END  = 52400,
        OFFSET_WP_ATM_LPMT = 52400,
        OFFSET_WP_ATM_LPMT_END = 52748,
        OFFSET_WP_ATM_MPMT     = 53000,
        OFFSET_WP_ATM_MPMT_END = 53600,
        OFFSET_WP_WAL_PMT      = 54000,
        OFFSET_WP_WAL_PMT_END  = 54005
    };



#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SPMT_FUNCTION void check_pmtid( int pmtid )
    {
        assert(  pmtid >= 0 && pmtid < OFFSET_WP_PMT_END );
        assert(!(pmtid >= OFFSET_CD_LPMT_END && pmtid < OFFSET_CD_SPMT ));
        assert(!(pmtid >= OFFSET_CD_SPMT_END && pmtid < OFFSET_WP_PMT ));

        assert( OFFSET_CD_LPMT_END - OFFSET_CD_LPMT == NUM_CD_LPMT );
        assert( OFFSET_CD_SPMT_END - OFFSET_CD_SPMT == NUM_SPMT );
        assert( OFFSET_WP_PMT_END  - OFFSET_WP_PMT  == NUM_WP );
    }
#endif


    /**
    s_pmt::contiguousidx_from_lpmtid
    ----------------------------------

    Transform non-contiguous lpmtid::

          +------------------+      +-----------------------------+       +--------+
          |     CD_LPMT      |      |     SPMT                    |       |  WP    |
          |     17612        |      |     25600                   |       |  2400  |
          +------------------+      +-----------------------------+       +--------+
          ^                  ^      ^                             ^       ^        ^
          0                  17612  20000                         45600   50000    52400
          |                  |      |                             |       |        |
          |                  |      |                             |       |        OFFSET_WP_PMT_END
          |                  |      |                             |       OFFSET_WP_PMT
          |                  |      |                             |
          |                  |      |                             OFFSET_CD_SPMT_END
          |                  |      OFFSET_CD_SPMT
          |                  |
          |                  OFFSET_CD_LPMT_END
          OFFSET_CD_LPMT


          lpmtid >= OFFSET_CD_LPMT_END && lpmtid < OFFSET_CD_SPMT    # 1st gap
          lpmtid >= OFFSET_CD_SPMT_END && lpmtid < OFFSET_WP_PMT     # 2nd gap

          17612+25600+2400 = 45612

          20000 - 17612 = 2388   # first gap
          50000 - 45600 = 4400   # second gap
          2388 + 4400   = 6788   # total gap
          52400 - 45612 = 6788   # diff between max pmtid kOFFSET_WP_PMT_END and NUM_ALL


    Into contiguousidx::

          +------------------+-----------------------------+--------+
          |     CD_LPMT      |     SPMT                    |  WP    |
          |     17612        |     25600                   |  2400  |
          +------------------+-----------------------------+--------+
          ^                  ^                             ^        ^
          0                17612                         43212    45612

         17612 + 25600 = 43212


    **/
    SPMT_FUNCTION int contiguousidx_from_pmtid( int pmtid )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        check_pmtid(pmtid);
#endif
        int contiguousidx = pmtid < OFFSET_CD_LPMT_END ?
                                                              pmtid
                                                        :
                                                            ( pmtid < OFFSET_CD_SPMT_END ?
                                                                                             pmtid - OFFSET_CD_SPMT + OFFSET_CD_LPMT_END
                                                                                          :
                                                                                             pmtid - OFFSET_WP_PMT + NUM_CD_LPMT + NUM_SPMT
                                                            )
                                                        ;

        return contiguousidx ;
    }

    SPMT_FUNCTION int pmtid_from_contiguousidx( int contiguousidx )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( contiguousidx >= 0 && contiguousidx < NUM_ALL );
#endif
        int pmtid = contiguousidx < NUM_CD_LPMT ?
                                                    contiguousidx
                                                 :
                                                    ( contiguousidx < NUM_CD_LPMT + NUM_SPMT ?
                                                                                                 contiguousidx - NUM_CD_LPMT + OFFSET_CD_SPMT
                                                                                             :
                                                                                                 contiguousidx - NUM_CD_LPMT - NUM_SPMT + OFFSET_WP_PMT
                                                    )
                                                 ;

         return pmtid ;
    }



    /**
    s_pmt::lpmtidx_from_lpmtid
    ---------------------------

    Rearrange the non-contiguous lpmtid::

          +------------------+      +-----------------------------+       +--------+
          |     CD_LPMT      |      |     SPMT                    |       |  WP    |
          |     17612        |      |     25600                   |       |  2400  |
          +------------------+      +-----------------------------+       +--------+
          ^                  ^      ^                             ^       ^        ^
          0                17612  20000                         45600   50000    52400
          |                  |      |                             |       |        |
          |                  |      |                             |       |        OFFSET_WP_PMT_END
          |                  |      |                             |       OFFSET_WP_PMT
          |                  |      |                             |
          |                  |      |                             OFFSET_CD_SPMT_END
          |                  |      OFFSET_CD_SPMT
          |                  |
          |                  OFFSET_CD_LPMT_END
          OFFSET_CD_LPMT



          17612+25600+2400 = 45612


    Into a contiguous 0-based index lpmtidx with SPMT excluded::

          +------------------+--------+
          |     CD_LPMT      |  WP    |
          |     17612        | 2400   |
          +------------------+--------+
          ^                  ^        ^
          0                 17612   20012

          17612+2400=20012

    **/

    SPMT_FUNCTION int lpmtidx_from_pmtid( int pmtid )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        check_pmtid(pmtid);
#endif
        int lpmtidx = pmtid < OFFSET_CD_LPMT_END ?
                                            pmtid
                                      :
                                           ( pmtid >= OFFSET_WP_PMT ? pmtid - OFFSET_WP_PMT + NUM_CD_LPMT : -1 )
                                      ;

        return lpmtidx ;
    }

    /**
    s_pmt::pmtid_from_lpmtidx
    ----------------------------

    Convert a contiguous 0-based index lpmtidx with SPMT excluded::

          +------------------+--------+
          |     CD_LPMT      |  WP    |
          |     17612        | 2400   |
          +------------------+--------+
          ^                  ^        ^
          0                 17612   20012


    Into the the non-contiguous lpmtid::

          +------------------+                                            +--------+
          |     CD_LPMT      |                                            |  WP    |
          |     17612        |                                            |  2400  |
          +------------------+                                            +--------+
          ^                  ^      ^                             ^       ^        ^
          0                17612  20000                         45600     50000    52400
                                                                          |
                                                                          OFFSET_WP_PMT


          17612+25600+2400 = 45612


    **/

    SPMT_FUNCTION int pmtid_from_lpmtidx( int lpmtidx )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( lpmtidx >= 0 && lpmtidx < NUM_CD_LPMT_AND_WP );
#endif
        int pmtid = lpmtidx < NUM_CD_LPMT ?
                                          lpmtidx
                                     :
                                          lpmtidx - NUM_CD_LPMT + OFFSET_WP_PMT
                                     ;

        return pmtid ;
    }



    SPMT_FUNCTION bool is_spmtid( int pmtid )
    {
        return pmtid >= OFFSET_CD_SPMT && pmtid < OFFSET_CD_SPMT + NUM_SPMT ;
    }
    SPMT_FUNCTION bool is_spmtidx( int pmtidx )
    {
        return pmtidx >= 0 && pmtidx < NUM_SPMT ;
    }


    SPMT_FUNCTION int pmtid_from_spmtidx( int spmtidx )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( is_spmtidx(spmtidx) );
#endif
        int pmtid = spmtidx + OFFSET_CD_SPMT ;
        return pmtid ;
    }

    SPMT_FUNCTION int spmtidx_from_pmtid( int pmtid )
    {
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
        assert( is_spmtid(pmtid) );
#endif
        int spmtidx = pmtid - OFFSET_CD_SPMT ;
        return spmtidx ;
    }






}

