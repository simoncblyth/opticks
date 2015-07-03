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
    idx.makeHistogram(0);

    target.repeat_to( maxrec, repeat );

    Index* index = idx.getHistogramIndex(0);
    index->dump();

    target.save("/tmp/ThrustIdxTest_target.npy");    
    repeat.save("/tmp/ThrustIdxTest_repeat.npy");    


    cudaDeviceSynchronize();

    return 0 ; 
}



