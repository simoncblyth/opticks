#include "ThrustIdx.hh"
#include "ThrustHistogram.hh"
#include <thrust/device_vector.h>
#include "NPY.hpp"
#include "Index.hpp"
#include "assert.h"

int main()
{
    typedef unsigned long long T ;
    typedef unsigned char S ;
   
    NPY<T>* sequence = NPY<T>::load("/tmp/thv.npy");
    std::vector<T>& seq = sequence->data();
    unsigned int num_elements = seq.size();
    thrust::device_vector<T> dseq(seq.begin(), seq.end());
    T* dseq_ptr = dseq.data().get();       

    unsigned int maxrec = 10 ; 
    unsigned int target_itemsize = 4 ; 

    std::vector<S> tgt(num_elements*target_itemsize);  
    thrust::device_vector<S> dtgt(tgt.begin(), tgt.end()); 
    S* dtgt_ptr = dtgt.data().get();       

    std::vector<S> rep(num_elements*target_itemsize*maxrec);  
    thrust::device_vector<S> drep(rep.begin(), rep.end()); 
    S* drep_ptr = drep.data().get();       

    // above is mockup of app environment
    
    ThrustArray<T> source(dseq_ptr, num_elements, 1 ); 
    ThrustArray<S> target(dtgt_ptr, num_elements, target_itemsize ); 
    ThrustArray<S> repeat(drep_ptr, num_elements*maxrec, target_itemsize ); 

    ThrustIdx<T,S> idx(&target, &source);
    idx.makeHistogram(0, "ThrustIdxTest_FlagSequence");

    target.repeat_to( maxrec, repeat );

    Index* index = idx.getHistogramIndex(0);
    index->dump();

    target.save("/tmp/ThrustIdxTest_target.npy");    
    repeat.save("/tmp/ThrustIdxTest_repeat.npy");    


    cudaDeviceSynchronize();

    return 0 ; 
}


/*

In [1]: rc = recsel_cerenkov_(1)
INFO:env.g4dae.types:loading /usr/local/env/recsel_cerenkov/1.npy 
-rw-r--r--  1 blyth  staff  24513720 Jul  5 16:48 /usr/local/env/recsel_cerenkov/1.npy

In [3]: rc.shape
Out[3]: (6128410, 1, 4)

In [4]: s = np.load("/tmp/SequenceNPYTest_SeqIdx.npy")

In [6]: s[:,0,0]
Out[6]: array([16, 16, 16, ...,  2,  2,  2], dtype=uint8)

In [7]: rc[:,0,0]
Out[7]: array([16, 16, 16, ...,  2,  2,  2], dtype=uint8)

In [8]: b = s[:,0,0] != rc[:,0,0]

In [9]: b
Out[9]: array([False, False, False, ..., False, False, False], dtype=bool)

In [12]: np.unique(s[:,0,0][b])
Out[12]: array([0], dtype=uint8)

In [13]: np.unique(rc[:,0,0][b])
Out[13]: array([255], dtype=uint8)


*/
