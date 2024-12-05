/**
SGenstep_test.cc
==================

~/o/sysrap/tests/SGenstep_test.sh 

**/

#include "ssys.h"
#include "SGenstep.hh"

struct SGenstep_test
{
    static int Slices_0(); 
    static int Slices_1(); 
    static int Slices_2();
 
    static int Main();  
};


int SGenstep_test::Slices_0()
{
    int max_photon = 500 ;    
    std::vector<int> num_ph = {  100,100,100,100,100,   100,100,100,100,100 } ; 
    std::cout << SGenstep::DescNum(num_ph) ;  
    NP* gs = SGenstep::MakeTestArray(num_ph) ; 

    std::vector<sslice> sl ; 
    SGenstep::GetGenstepSlices(sl, gs, max_photon ); 

    assert( sl.size() == 2 ); 
    assert( sl[0].matches(0,  5,   0, 500) ); 
    assert( sl[1].matches(5, 10, 500, 500) ); 

    std::cout << sslice::Desc(sl) ; 
    return 0 ; 
}


int SGenstep_test::Slices_1()
{
    int max_photon = 599 ;    
    std::vector<int> num_ph = {  100,100,100,100,100,   100,100,100,100,100 } ; 
    std::cout << SGenstep::DescNum(num_ph) ;  
    NP* gs = SGenstep::MakeTestArray(num_ph) ; 

    std::vector<sslice> sl ; 
    SGenstep::GetGenstepSlices(sl, gs, max_photon ); 

    assert( sl.size() == 2 ); 
    assert( sl[0].matches(0,  5,   0, 500) ); 
    assert( sl[1].matches(5, 10, 500, 500) ); 

    std::cout << sslice::Desc(sl) ; 
    return 0 ; 
}


int SGenstep_test::Slices_2()
{
    int max_photon = 1000 ;    
    std::vector<int> num_ph = {  100,100,100,100,100,   100,100,100,100,100 } ; 
    std::cout << SGenstep::DescNum(num_ph) ;  
    NP* gs = SGenstep::MakeTestArray(num_ph) ; 

    std::vector<sslice> sl ; 
    SGenstep::GetGenstepSlices(sl, gs, max_photon ); 

    assert( sl.size() == 1 ); 
    assert( sl[0].matches(0,  10,   0, 1000) ); 

    std::cout << sslice::Desc(sl) ; 
    return 0 ; 
}



int SGenstep_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST","Slices_1"); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 
 
    int rc = 0 ; 
    if(ALL||strcmp(TEST,"Slices_0") == 0 ) rc += Slices_0(); 
    if(ALL||strcmp(TEST,"Slices_1") == 0 ) rc += Slices_1(); 
    if(ALL||strcmp(TEST,"Slices_2") == 0 ) rc += Slices_2();
 
    return rc ;
}

int main() {  return SGenstep_test::Main() ; }

