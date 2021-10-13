#include <iostream>
#include <iomanip>
#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "NP.hh"
#include "SPath.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve("$TMP/NPY9SpawnTest", create_dirs ); 


    NPY<float>* f0 = NPY<float>::make_identity_transforms(5) ; 
    f0->fillIndexFlat(); 
    f0->save(fold, "f0.npy"); 

    NP* f1 = f0->spawn(); 
    f1->save(fold, "f1.npy"); 


    NPY<double>* d0 = NPY<double>::make_identity_transforms(5) ; 
    d0->fillIndexFlat(); 
    d0->save(fold, "d0.npy"); 

    NP* d1 = d0->spawn(); 
    d1->save(fold, "d1.npy"); 


    NPY<int>* i0 = NPY<int>::make(10,4,4) ;  
    i0->fillIndexFlat(); 
    i0->save(fold, "i0.npy"); 

    NP* i1 = i0->spawn(); 
    i1->save(fold, "i1.npy"); 



    return 0 ; 
}

// om-;TEST=NPY9SpawnTest om-t

