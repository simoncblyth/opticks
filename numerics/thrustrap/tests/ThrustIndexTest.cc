#include "ThrustIndex.hh"
#include "ThrustHistogram.hh"
#include <thrust/device_vector.h>
#include "NPY.hpp"
#include "assert.h"

int main()
{
    unsigned int target_itemsize = 4 ; 
    typedef unsigned long long T ;
    typedef unsigned char S ;

    // source sequences
   
    NPY<T>* history = NPY<T>::load("/tmp/thv.npy");
    std::vector<T>& his = history->data();
    unsigned int num_elements = his.size();
    thrust::device_vector<T> dhis(his.begin(), his.end());
    T* dhis_ptr = dhis.data().get();       

    NPY<T>* material = NPY<T>::load("/tmp/thv.npy");  // use same path for now
    std::vector<T>& mat = material->data();
    assert(num_elements == mat.size());
    thrust::device_vector<T> dmat(mat.begin(), mat.end());
    T* dmat_ptr = dmat.data().get();       

    // target sequence index 

    std::vector<S> tgt(num_elements*target_itemsize);  
    thrust::device_vector<S> dtgt(tgt.begin(), tgt.end()); 
    S* dtgt_ptr = dtgt.data().get();       

    // above is mockup of app environment

    ThrustIndex<T,S> idx( dtgt_ptr, num_elements, target_itemsize);
    idx.indexHistory(  dhis_ptr, 0 );
    idx.indexMaterial( dmat_ptr, 1 );
    idx.dumpTarget("ThrustIndexTest", 100);

    NPY<S>* target = idx.makeTargetArray();
    target->setVerbose(); 
    target->save("/tmp/ThrustIndexTest.npy");

    cudaDeviceSynchronize();

    return 0 ; 
}

/*

In [1]: i = np.load("/tmp/ThrustIndexTest.npy")

In [2]: i.shape
Out[2]: (2451364, 1, 1)

In [3]: i.reshape(-1,4).shape
Out[3]: (612841, 4)

In [4]: ii = i.reshape(-1,4)

In [5]: ii
Out[5]: 
array([[ 15,  15,   0,   0],
       [255, 255,   0,   0],
       [  0,   0,   0,   0],
       ..., 
       [  1,   1,   0,   0],
       [  1,   1,   0,   0],
       [  1,   1,   0,   0]], dtype=uint8)

*/


