#include <iostream>
#include <string>
#include <vector>

#include "NPY.hpp"
#include "NMeta.hpp"

/**
ABANDONED : npy_serialize_test.cc
------------------------------------

Decided better to do serialization/deserialization at lower standalone level 
instead of at NPY level. This is for generality, ease of use and deployment.

Aiming for header only NP server/client using boost::asio 
so Opticks C++ clients can just include a header to 
get GPU propagation.

See:

* https://github.com/simoncblyth/np
* ~/np 


**/

int main(int argc, char** argv)
{
    NPY<float>* arr0 = NPY<float>::make(10,4); 
    arr0->fillIndexFlat(); 
    arr0->dump("arr0"); 

    arr0->setMeta<unsigned>("unsigned", 1); 
    arr0->setMeta<int>("int", -5); 
    arr0->setMeta<float>("float", 42.5);
    arr0->setMeta<std::string>("string", "hello");
     
    NMeta* meta0 = arr0->getMeta();  
    meta0->dump(); 
    std::cout << "meta0 " << meta0 << std::endl ; 
    meta0->dumpLines(); 


    std::vector<unsigned char> vdst ; 
    arr0->saveToBuffer(vdst); 

    NPY<float>* arr1 = NPY<float>::loadFromBuffer(vdst); 
    arr1->dump("arr1"); 

    assert( NPY<float>::compare(arr0, arr1, true) == 0 ); 


    return 0 ; 
}

