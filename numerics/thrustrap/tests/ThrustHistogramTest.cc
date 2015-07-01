#include <thrust/device_vector.h>
#include "ThrustHistogram.hh"
#include "NPY.hpp"
#include "assert.h"

int main()
{
    // sling histories onto device and create histogram 

    typedef unsigned long long T ;
    typedef unsigned int S ;

    NPY<T>* history = NPY<T>::load("/tmp/thv.npy");

    std::vector<T>& his = history->data();
    unsigned int size = his.size();

    thrust::device_vector<T> dhis(his.begin(), his.end());
    T* dhis_ptr = dhis.data().get();       
    T* dhis_ptr_alt = thrust::raw_pointer_cast(&dhis[0]);
    assert( dhis_ptr == dhis_ptr_alt );    // two eqivalent ways to access raw device pointers

    //thrust::device_vector<S> dtgt(size); // <--- cannot do in .cpp, needs to be in .cu https://github.com/thrust/thrust/issues/526
    std::vector<S> tgt(size);  
    thrust::device_vector<S> dtgt(tgt.begin(), tgt.end()); 
    S* dtgt_ptr = dtgt.data().get();       

    // above create device vectors to mimic the actual situation 
    // of addressing OpenGL/OptiX buffers 

    ThrustHistogram<T,S> th(dhis_ptr, dtgt_ptr, size); // NB target must be same size as history 

    th.createHistogram();

    th.dumpHistogram();

    th.apply(); 

    //th.dumpTarget();

    th.dumpHistoryTarget();

    cudaDeviceSynchronize();

    return 0 ; 
}

/*

In [1]: t = np.load("/tmp/thv.npy")

In [2]: t.shape
Out[2]: (612841, 1, 1)

In [3]: t[:,0,0]
Out[3]: array([19649, 15297,    65, ...,     3,     3,     3], dtype=uint64)

In [4]: tt = t[:,0,0]

In [5]: map(hex_,tt[:1000])
Out[5]: 
['0x4cc1',
 '0x3bc1',
 '0x41',
 '0x3bc1',
 '0x3bc1',
 '0x4c1',
 '0x4c1',
 '0x4c1',
 '0x41',
 '0x41',
 '0x41',
 '0x4cbc1',
 '0x4c1',
 '0x4c1',
 '0x4c1',
  ...

n [8]: utt = np.unique(tt)

In [9]: len(utt)
Out[9]: 2307




*/

