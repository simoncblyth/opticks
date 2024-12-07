/**
SGenstep_test.cc
==================

~/o/sysrap/tests/SGenstep_test.sh 

TEST=Slices_3 ~/o/sysrap/tests/SGenstep_test.sh 
TEST=Slices_4 ~/o/sysrap/tests/SGenstep_test.sh 


**/

#include "ssys.h"
#include "SGenstep.h"

struct SGenstep_test
{
    static int Slices_0(); 
    static int Slices_1(); 
    static int Slices_2();
    static int Slices_3();
    static int Slices_4();
 
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


int SGenstep_test::Slices_3()
{
    int max_photon = 1000 ;    
    //                           0      1    2      3  4    5    6    7    8 
    std::vector<int> num_ph =  { 300,   500, 200,   0, 100, 300, 400, 200, 400 } ; 
    //                           300  800   1000  1000 1100 1400 1800 2000 2400

    std::cout << SGenstep::DescNum(num_ph) ;  
    NP* gs = SGenstep::MakeTestArray(num_ph) ; 

    std::vector<sslice> sl ; 
    SGenstep::GetGenstepSlices(sl, gs, max_photon ); 
    std::cout << sslice::Desc(sl) ; 

    assert( sl.size() == 3 ); 
    assert( sl[0].matches(0,  4,    0, 1000) ); 
    assert( sl[1].matches(4,  8, 1000, 1000) ); 
    assert( sl[2].matches(8,  9, 2000,  400) ); 

    return 0 ; 
}

int SGenstep_test::Slices_4()
{
    int max_photon = 2400 ;    
    //                           0      1    2      3  4    5    6    7    8 
    std::vector<int> num_ph =  { 300,   500, 200,   0, 100, 300, 400, 200, 400 } ; 
    //                           300  800   1000  1000 1100 1400 1800 2000 2400

    std::cout << SGenstep::DescNum(num_ph) ;  
    NP* gs = SGenstep::MakeTestArray(num_ph) ; 

    std::vector<sslice> sl ; 
    SGenstep::GetGenstepSlices(sl, gs, max_photon ); 
    std::cout << sslice::Desc(sl) ; 

    assert( sl.size() == 1 ); 
    assert( sl[0].matches(0,  9,    0, 2400) ); 

    return 0 ; 
}






int SGenstep_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST","ALL"); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 
 
    int rc = 0 ; 
    if(ALL||strcmp(TEST,"Slices_0") == 0 ) rc += Slices_0(); 
    if(ALL||strcmp(TEST,"Slices_1") == 0 ) rc += Slices_1(); 
    if(ALL||strcmp(TEST,"Slices_2") == 0 ) rc += Slices_2();
    if(ALL||strcmp(TEST,"Slices_3") == 0 ) rc += Slices_3();
    if(ALL||strcmp(TEST,"Slices_4") == 0 ) rc += Slices_4();

    std::cout 
        << "SGenstep_test::Main"
        << " TEST " << TEST 
        << " rc " << rc
        << "\n"
        ;
   

    return rc ;
}

int main() {  return SGenstep_test::Main() ; }

