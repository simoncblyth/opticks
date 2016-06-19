
#include "NPY.hpp"
#include "G4StepNPY.hpp"


void test_g4stepnpy()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
    if(!npy) return ; 

    G4StepNPY* step = new G4StepNPY(npy);   
    step->dump("G4StepNPY");
}

void test_make_modulo()
{
    NPY<float>* npy0 = NPY<float>::load("cerenkov","1", "dayabay");
    if(!npy0) return ; 
    G4StepNPY* step0 = new G4StepNPY(npy0);   
    step0->dump("G4StepNPY0");


    NPY<float>* npy1 = NPY<float>::make_modulo(npy0, 10) ;
    assert(npy1); 
    G4StepNPY* step1 = new G4StepNPY(npy1);   
    step1->dump("G4StepNPY1");

}

void test_g4stepnpy_materials()
{

    //const char* det = "dayabay" ; 
    const char* det = "juno" ; 

    NPY<float>* npy = NPY<float>::load("cerenkov","1", det);
    if(!npy) return ; 
    G4StepNPY* step = new G4StepNPY(npy);   
    step->dump("G4StepNPY");

    std::set<int> s = npy->uniquei(0,2);
    typedef std::set<int>::const_iterator SII ; 

    for(SII it=s.begin() ; it != s.end() ; it++)
    {
        printf(" %d \n", *it ); 
    }


    std::map<int,int> m = npy->count_uniquei(0,2);
    typedef std::map<int, int>::const_iterator MII ; 

    for(MII it=m.begin() ; it != m.end() ; it++)
    {
        printf(" %d : %d \n", it->first, it->second ); 
    }
}



int main(int , char** )
{
    test_g4stepnpy();
    test_g4stepnpy_materials();
    test_make_modulo();   

    return 0 ;
}
