#include "ViewNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"

#include <iostream>
#include "assert.h"

void test_ViewNPY()
{   
    NPY<float>* npy = NPY<float>::load("cerenkov","1","dayabay");

    ViewNPY v("test", npy,1,0);  // [:,1,0:3]
    v.dump("vecNPY"); 
    v.Summary("ViewNPY::Summary");

    print( v.getModelToWorldPtr(), "v.getModelToWorldPtr()");

}


int main()
{
    test_ViewNPY();

    return 0 ;
}
