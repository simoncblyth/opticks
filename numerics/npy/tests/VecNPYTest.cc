#include "VecNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"

#include <iostream>
#include "assert.h"

void test_VecNPY()
{   
    NPY* npy = NPY::load("cerenkov","1");

    VecNPY v(npy,1,0);  // [:,1,0:3]
    v.dump("vecNPY"); 
    v.Summary("VecNPY::Summary");

    print( v.getModelToWorldPtr(), "v.getModelToWorldPtr()");

}


int main()
{
    test_VecNPY();

    return 0 ;
}
