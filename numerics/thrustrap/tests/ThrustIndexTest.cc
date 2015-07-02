#include "ThrustIndex.hh"
#include "ThrustHistogram.hh"
#include <thrust/device_vector.h>
#include "NPY.hpp"
#include "Index.hpp"
#include "assert.h"

int main()
{
    typedef unsigned long long T ;
    typedef unsigned char S ;
    unsigned int sequence_itemsize = 1 ; 
    unsigned int target_itemsize = 4 ; 

    // source sequences
   
    NPY<T>* sequence = NPY<T>::load("/tmp/thv.npy");
    std::vector<T>& seq = sequence->data();
    unsigned int num_elements = seq.size();
    thrust::device_vector<T> dseq(seq.begin(), seq.end());
    T* dseq_ptr = dseq.data().get();       

    // target sequence index 

    std::vector<S> tgt(num_elements*target_itemsize);  
    thrust::device_vector<S> dtgt(tgt.begin(), tgt.end()); 
    S* dtgt_ptr = dtgt.data().get();       

    // above is mockup of app environment

    ThrustIndex<T,S> idx( dseq_ptr, dtgt_ptr, num_elements, sequence_itemsize, target_itemsize);
    idx.indexHistory(0);
    //idx.indexMaterial(1);
    idx.dumpTarget("ThrustIndexTest", 100);

    NPY<S>* target = idx.makeTargetArray();
    target->setVerbose(); 
    target->save("/tmp/ThrustIndexTest_SeqIdx.npy");
 
    NPY<T>* hsia = idx.getHistory()->makeSequenceIndexArray();
    hsia->save("/tmp/ThrustSequenceIndexArray.npy");

    Index* hisidx = idx.getHistory()->getIndex();
    hisidx->dump();

    //Index* matidx = idx.getMaterial()->getIndex();
    //matidx->dump();



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


    In [1]: s = np.load("/tmp/SequenceNPYTest_SeqIdx.npy")

    In [2]: t = np.load("/tmp/ThrustIndexTest_SeqIdx.npy")

    In [3]: s.shape
    Out[3]: (6128410, 1, 4)

    In [4]: t.shape
    Out[4]: (2451364, 1, 1)

    In [5]: t.reshape(-1,1,4).shape
    Out[5]: (612841, 1, 4)

    In [6]: tt = t.reshape(-1,1,4)

    In [13]: s[:200:10,0,0]
    Out[13]: 
    array([16,  0,  1,  0,  0,  9,  9,  9,  1,  1,  1,  0,  9,  9,  9,  1,  9,
            0,  1,  9], dtype=uint8)

    In [15]: tt[:20,0,0]
    Out[15]: 
    array([ 15, 255,   0, 255, 255,   8,   8,   8,   0,   0,   0, 255,   8,
             8,   8,   0,   8, 255,   0,   8], dtype=uint8)

    In [16]: tt[:20,0,0] + 1
    Out[16]: 
    array([16,  0,  1,  0,  0,  9,  9,  9,  1,  1,  1,  0,  9,  9,  9,  1,  9,
            0,  1,  9], dtype=uint8)


    In [17]: ttt = tt[:,0,0] + 1

    In [20]: ss = s[::10,0,0] 

    In [27]: ss == ttt
    Out[27]: array([ True,  True,  True, ...,  True,  True,  True], dtype=bool)

    In [28]: np.count_nonzero( ss == ttt )
    Out[28]: 604629

    In [29]: np.count_nonzero( ss != ttt )
    Out[29]: 8212

    In [31]: 8212./604629.
    Out[31]: 0.013581882443614184

    In [32]: b = ss != ttt 

    In [33]: b
    Out[33]: array([False, False, False, ..., False, False, False], dtype=bool)

    In [34]: ss[b]
    Out[34]: array([39, 39, 39, ..., 39, 39, 39], dtype=uint8)

    In [35]: ttt[b]
    Out[35]: array([0, 0, 0, ..., 0, 0, 0], dtype=uint8)

    In [36]: ss[b].shape
    Out[36]: (8212,)

    In [37]: ttt[b].shape
    Out[37]: (8212,)

    ## ahha difference looks to be due to different truncation choice 

    In [39]: np.unique(ss[b])
    Out[39]: array([33, 34, 35, 36, 37, 38, 39], dtype=uint8)

    In [40]: np.unique(ttt[b])
    Out[40]: array([0], dtype=uint8)




After aligning truncation::

    In [1]: s = np.load("/tmp/SequenceNPYTest_SeqIdx.npy")

    In [2]: t = np.load("/tmp/ThrustIndexTest_SeqIdx.npy")

    In [3]: tt = t.reshape(-1,1,4)[:,0,0] 

    In [4]: ss = s[::10,0,0] 

    In [5]: tt
    Out[5]: array([ 15, 255,   0, ...,   1,   1,   1], dtype=uint8)

    In [6]: ss
    Out[6]: array([16,  0,  1, ...,  2,  2,  2], dtype=uint8)

    In [7]: np.all( tt + 1 == ss )
    Out[7]: True




*/


