/**
QPMTTest.cc
=============

The lpmtcat 0,1,2 correspond to NNVT,HAMA,NNVT_HiQE
----------------------------------------------------

::

    In [6]: np.where( t.qpmt.src_lcqs[:,0] == 0 )  ## NNVT
    Out[6]:
    (array([   55,    98,   137,   267,   368,   372,   374,   378,   388,   391,   393,   394,   396,   398,   404,   409, ..., 17198, 17199, 17201, 17205, 17206, 17209, 17219, 17224, 17231, 17234,
            17242, 17255, 17327, 17504, 17526, 17537]),)

    In [7]: np.where( t.qpmt.src_lcqs[:,0] == 1 )  ## HAMA
    Out[7]:
    (array([    0,     1,     3,     5,     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,    17,    18, ..., 17596, 17597, 17598, 17599, 17600, 17601, 17602, 17603, 17604, 17605,
            17606, 17607, 17608, 17609, 17610, 17611]),)

    In [8]: np.where( t.qpmt.src_lcqs[:,0] == 2 )  ## NNVT_HiQE
    Out[8]:
    (array([    2,     4,     6,    21,    22,    23,    24,    25,    26,    27,    28,    29,    30,    31,    32,    33, ..., 17575, 17576, 17577, 17578, 17579, 17580, 17581, 17582, 17583, 17584,
            17585, 17586, 17587, 17588, 17589, 17590]),)

    In [9]: np.unique( t.qpmt.src_lcqs[:,0], return_counts=True )
    Out[9]: (array([0, 1, 2], dtype=int32), array([2720, 4997, 9895]))

    In [10]: np.unique( t.qpmt.src_lcqs[:,0], return_counts=True )[1].sum()
    Out[10]: 17612


2025/06
----------

::

    In [3]: np.c_[np.unique( t.qpmt.src_lcqs[:,0], return_counts=True )]
    Out[3]:
    array([[   0, 2738],
           [   1, 4955],
           [   2, 9919]])

    In [4]: tab = np.c_[np.unique( t.qpmt.src_lcqs[:,0], return_counts=True )]

    In [5]: tab[:,1].sum()
    Out[5]: np.int64(17612)


**/

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"
#include "SPMT.h"
#include "QPMTTest.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    std::cout << "QPMTTest.main.Before:" << QPMT<float>::Desc() << std::endl ;

    const NPFold* jpmt = SPMT::Serialize();

    LOG_IF(fatal, jpmt==nullptr) << " jpmt==nullptr : probably GEOM,${GEOM}_CFBaseFromGEOM envvars not setup ?" ;
    if(jpmt==nullptr) return 0 ;

    QPMTTest<float> t(jpmt);
    NPFold* f = t.serialize();
    cudaDeviceSynchronize();
    f->save("$FOLD");

    std::cout << "QPMTTest.main.After:" << QPMT<float>::Desc() << std::endl ;

    return 0 ;
}


