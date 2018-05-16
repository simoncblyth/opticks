#include <iostream>
#include "SSys.hh"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    SSys::Dump("SSys found");


    NPY<float>* a = NPY<float>::make_identity_transforms(10) ;  

    a->save("$TMP", "FindOpticks.npy") ; 



    return 0 ; 
}
