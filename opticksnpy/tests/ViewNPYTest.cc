#include "ViewNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"

#include <iostream>
#include <cassert>

void test_ViewNPY()
{   
    //NPY<float>* npy = NPY<float>::load("cerenkov","1","dayabay");

    NPY<float>* npy = NPY<float>::make(1,6,4);
    npy->zero();


    ViewNPY v("test", npy,1,0,0);  // [:,1,0:3]
    v.dump("vecNPY"); 
    v.Summary("ViewNPY::Summary");

    print( v.getModelToWorldPtr(), "v.getModelToWorldPtr()");

}


int main()
{
    test_ViewNPY();

    return 0 ;
}
