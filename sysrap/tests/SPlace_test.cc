// ./SPlace_test.sh 


#include <cstdlib>
#include "SPlace.h"

const char* FOLD = getenv("FOLD") ; 
const char* TEST = getenv("TEST") ; 
const char* OPTS = getenv("OPTS") ; 


const char* Name()
{
    std::stringstream ss ; 
    ss << TEST << ".npy" ; 
    std::string s = ss.str(); 
    return strdup(s.c_str()) ; 
}


void test_AroundCylinder()
{
    double sc = 5. ; 
    double radius = sc ; 
    double halfheight = sc ; 

    NP* tr = SPlace::AroundCylinder(OPTS, radius, halfheight ); 
    tr->save(FOLD, Name()); 
}

void test_AroundSphere()
{
    double sc = 5. ; 
    double radius = sc ; 
    double item_arclen = 1. ; 
    unsigned num_ring = 8 ;

    NP* tr = SPlace::AroundSphere(OPTS, radius, item_arclen, num_ring); 
    tr->save(FOLD, Name()); 
}


int main(int argc, char** argv)
{
    if(strcmp(TEST,"AroundCylinder")==0) test_AroundCylinder(); 
    if(strcmp(TEST,"AroundSphere")==0)   test_AroundSphere(); 
    return 0 ; 
}
