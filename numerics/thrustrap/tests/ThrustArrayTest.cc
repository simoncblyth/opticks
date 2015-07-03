#include <thrust/device_vector.h>
#include "ThrustArray.hh"
#include "NPY.hpp"
#include "Index.hpp"
#include "assert.h"

int main()
{
    typedef unsigned char S ;

    std::vector<S> ha ;

    ha.push_back(1);
    ha.push_back(2);
    ha.push_back(3);
    ha.push_back(4);

    ha.push_back(10);
    ha.push_back(20);
    ha.push_back(30);
    ha.push_back(40);


    thrust::device_vector<S> da(ha.begin(), ha.end());
    S* da_ptr = da.data().get();       

    ThrustArray<S> ta(da_ptr, ha.size()/4, 4 );   // numitems, itemsize 
    ta.dump();


    std::vector<S> hb(8) ;
    thrust::device_vector<S> db(hb.begin(), hb.end());
    S* db_ptr = db.data().get();       
    ThrustArray<S> tb(db_ptr, 2, 4 ); 


    //ta.copy_to(tb);
    //tb.dump();

    unsigned int dupe = 10 ;  

    std::vector<S> hc(ha.size()*dupe) ;
    thrust::device_vector<S> dc(hc.begin(), hc.end());
    S* dc_ptr = dc.data().get();       
    ThrustArray<S> tc(dc_ptr, hc.size()/4, 4 );   // numitems, itemsize
 

    ta.repeat_to(dupe, tc);
    tc.dump();




    NPY<S>* ntc = tc.makeNPY();
    ntc->setVerbose();
    ntc->save("/tmp/ThrustArrayTest_ntc.npy");




    cudaDeviceSynchronize();

    return 0 ; 
}

