// ./SPlace_test.sh 


#include <cstdlib>
#include "SPlace.h"

const char* TEST = getenv("TEST"); 
const char* FOLD = getenv("FOLD") ; 


const char* Name(unsigned idx)
{
    std::stringstream ss ; 
    ss << TEST << idx << ".npy" ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()) ; 
}


void test_AroundCylinder()
{
    double sc = 5. ; 
    double radius = sc ; 
    double halfheight = sc ; 

    for(unsigned i=0 ; i < 2 ; i++)
    {
        NP* tr = SPlace::AroundCylinder(radius, halfheight, bool(i) ); 
        tr->save(FOLD, Name(i)); 
    }
}

void test_AroundSphere()
{
    double sc = 5. ; 
    double radius = sc ; 
    double item_arc_length = 1. ; 
    unsigned num_ring = 8 ;


    for(unsigned i=0 ; i < 2 ; i++)
    {
        NP* tr = SPlace::AroundSphere(radius, item_arc_length, bool(i), num_ring ); 
        tr->save(FOLD, Name(i)); 
    }
}


int main(int argc, char** argv)
{
    if(strcmp(TEST,"AroundCylinder")==0) test_AroundCylinder(); 
    if(strcmp(TEST,"AroundSphere")==0)   test_AroundSphere(); 
    return 0 ; 
}
