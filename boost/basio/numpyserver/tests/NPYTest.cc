#include "NPY.hpp"
#include "G4StepNPY.hpp"
#include <iostream>


void test_ctor()
{
    std::vector<int> shape = {2,2} ;
    std::vector<float> data = {1.f,2.f,3.f,4.f}  ;
    std::string metadata = "{}";

    NPY npy(shape,data,metadata) ;
    std::cout << npy.description("npy") << std::endl ; 
}

void test_path()
{
    std::string path = NPY::path("cerenkov", "1");
    std::cout << path << std::endl ; 
}

void test_load()
{
    NPY* npy = NPY::load("cerenkov","1");
    std::cout << npy->description("npy") << std::endl ; 
}

void test_load_missing()
{
    NPY* npy = NPY::load("cerenkov","missing");
    if(npy) std::cout << npy->description("npy") << std::endl ; 
}

void test_g4stepnpy()
{
    NPY* npy = NPY::load("cerenkov","1");
    G4StepNPY* step = new G4StepNPY(npy);   
    step->dump("G4StepNPY");
}



int main()
{
    //test_ctor();
    //test_path();
    //test_load();
    //test_load_missing();

    test_g4stepnpy();

    return 0 ;
}
