#include "NPY.hpp"
#include "G4StepNPY.hpp"


#include <iostream>
#include "assert.h"

#include <glm/glm.hpp>


void test_setQuad()
{
   NPYBase* dom = NPY<float>::make_vec4(10,1,0.f) ;

   glm::vec4 q0(0.f,1.f,2.f,3.f);
   glm::vec4 q1(1.f,1.f,2.f,3.f);
   glm::vec4 q9(9.f,1.f,2.f,3.f);

   dom->setQuad(0,0, q0 );
   dom->setQuad(1,0, q1 );
   dom->setQuad(9,0, q9 );
    
   dom->save("/tmp/test_setQuad.npy");
}


void test_ctor()
{
    std::vector<int> shape = {2,2} ;
    std::vector<float> data = {1.f,2.f,3.f,4.f}  ;
    std::string metadata = "{}";

    NPY<float> npy(shape,data,metadata) ;
    std::cout << npy.description("npy") << std::endl ; 
}

void test_path()
{
    std::string path = NPY<float>::path("cerenkov", "1");
    std::cout << path << std::endl ; 
}

void test_load()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1");
    std::cout << npy->description("npy") << std::endl ; 
}

void test_save_path()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1");
    std::cout << npy->description("npy") << std::endl ; 
    npy->save("/tmp/test_save_path.npy");
}



void test_load_path()
{
    const char* path = "/tmp/slowcomponent.npy" ;
    //const char* path = "/usr/local/env/cerenkov/1.npy" ;
    NPY<float>* npy = NPY<float>::debugload(path);
    if(npy) npy->Summary(path);
}


void test_load_missing()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","missing");
    if(npy) std::cout << npy->description("npy") << std::endl ; 
}

void test_g4stepnpy()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1");
    G4StepNPY* step = new G4StepNPY(npy);   
    step->dump("G4StepNPY");
}

void test_getData()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1");
    float* data = npy->getValues();

    for(unsigned int i=0 ; i < 16 ; i++ )
    {
        uif_t uif ;
        uif.f = *(data+i) ;
        printf(" %3u : %15f f   %15d i  %15u u  \n", i, uif.f, uif.i, uif.u );
    }

    char* raw = (char*)data ;
    for(unsigned int i=0 ; i < 16 ; i++ )
    {
        char c = *(raw+i) ;
        printf(" %3u : %d i   %x x \n", i, c, c);
    }
    std::cout << npy->description("npy") << std::endl ; 
}

void test_getUSum()
{
    NPY<float>* c1 = NPY<float>::load("cerenkov","1");
    NPY<float>* s1 = NPY<float>::load("scintillation","1");
    
    unsigned int n_c1 = c1->getUSum(0, 3);
    unsigned int n_s1 = s1->getUSum(0, 3);
    printf("test_getUSum n_c1:%u n_c1:%u tot:%u \n", n_c1, n_s1, n_c1+n_s1);

}


int main()
{
    //test_ctor();
    //test_path();
    //test_load();
    //test_load_missing();
    //test_g4stepnpy();
    //test_getData();

    //test_getUSum();
    //test_load_path();
    //test_save_path();

    test_setQuad();

    return 0 ;
}
