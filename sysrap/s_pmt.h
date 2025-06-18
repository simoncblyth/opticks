#pragma once


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SPMT_FUNCTION __host__ __device__ __forceinline__
#else
#    define SPMT_FUNCTION inline
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
        NUM_ALL            = 45612
    };


    /**
    s_pmt::contiguousidx_from_lpmtid
    ----------------------------------

    Transform non-contiguous lpmtid::

          +------------------+      +-----------------------------+       +--------+
          |     CD_LPMT      |      |     SPMT                    |       |  WP    |
          |     17612        |      |     25600                   |       |  2400  |
          +------------------+      +-----------------------------+       +--------+
          ^                  ^      ^                             ^       ^        ^
          0                17612  20000                         45600   50000    52400

          17612+25600+2400 = 45612


    Into contiguousidx::

          +------------------+-----------------------------+--------+
          |     CD_LPMT      |     SPMT                    |  WP    |
          |     17612        |     25600                   |  2400  |
          +------------------+-----------------------------+--------+
          ^                  ^                             ^        ^
          0                17612                         43212    45612

         17612 + 25600 = 43212


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

    // note no handling to notice invalid lpmtid within the gaps

    **/
    SPMT_FUNCTION int contiguousidx_from_lpmtid( int lpmtid )
    {
        int contiguousidx = lpmtid < 17612 ?
                                                lpmtid
                                            :
                                                ( lpmtid < 45600 ?
                                                                    lpmtid - 20000 + 17612
                                                                 :
                                                                    lpmtid - 50000 + 17612 + 25600
                                                )
                                            ;

        return contiguousidx ;
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

          17612+25600+2400 = 45612


    Into a contiguous 0-based index lpmtidx with SPMT excluded::

          +------------------+--------+
          |     CD_LPMT      |  WP    |
          |     17612        | 2400   |
          +------------------+--------+
          ^                  ^        ^
          0                 17612   20012

          17612+2400=20012

    // note no handling to notice invalid lpmtid within the gaps

    **/

    SPMT_FUNCTION int lpmtidx_from_lpmtid( int lpmtid )
    {
        int lpmtidx = lpmtid < 17612 ?
                                            lpmtid
                                      :
                                           ( lpmtid >= 50000 ? lpmtid - 50000 + 17612 : -1 )
                                      ;

        return lpmtidx ;
    }

    /**
    s_pmt::lpmtid_from_lpmtidx
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
          0                17612  20000                         45600   50000    52400

          17612+25600+2400 = 45612


    **/

    SPMT_FUNCTION int lpmtid_from_lpmtidx( int lpmtidx )
    {
        //assert( lpmtidx >= 0 && lpmtidx < NUM_CD_LPMT + NUM_WP );
        int lpmtid = lpmtidx < 17612 ?
                                          lpmtidx
                                     :
                                          lpmtidx - 17612 + 50000
                                     ;

        return lpmtid ;
    }


}

