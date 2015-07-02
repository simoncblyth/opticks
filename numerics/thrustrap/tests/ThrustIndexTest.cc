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
 
    NPY<T>* hsia = idx.getHistory()->makeSequenceIndexArray();
    hsia->save("/tmp/ThrustSequenceIndexArray.npy");

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



Hmm comparing indices with SequenceNPY, get correspondence
until the counts are matching where run into unstable
sort order differences::


   s = np.load("/tmp/seqhis.npy")
   t = np.load("/tmp/ThrustSequenceIndexArray.npy")
   t[:40,0] == s[:40,0]


   In [21]: np.all( t[:40,0] == s[:40,0] )
   Out[21]: True

    In [23]: s[40:50,0]
    Out[23]: 
    array([[  1288490177,          837],
           [879608812881,          786],
           [879609298113,          762],
           [      283985,          711],
           [810621389905,          607],
           [879324089425,          600],
           [879592524993,          599],
           [     5031249,          597],
           [329853486417,          591],
           [879609273425,          591]], dtype=uint64)

    In [24]: t[40:50,0]
    Out[24]: 
    array([[  1288490177,          837],
           [879608812881,          786],
           [879609298113,          762],
           [      283985,          711],
           [810621389905,          607],
           [879324089425,          600],
           [879592524993,          599],
           [     5031249,          597],
           [879609273425,          591],
           [329853486417,          591]], dtype=uint64)


So compare at seqidx level..


*/


