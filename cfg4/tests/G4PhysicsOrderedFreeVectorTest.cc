#include <iostream>
#include <fstream>
#include <sstream>

#include "CVec.hh"
#include "G4PhysicsOrderedFreeVector.hh"
#include "PLOG.hh"

#include "SDirect.hh"


void test_redirected( G4PhysicsOrderedFreeVector& vec, bool ascii )
{
    std::ofstream fp("/dev/null", std::ios::out); 
    std::stringstream ss ;     
    stream_redirect rdir(ss,fp); // stream_redirect such that writes to the file instead go to the stringstream 
    
    vec.Store(fp, ascii );

    std::cout <<  ss.str() << std::endl ; 
}



void test_caveman( G4PhysicsOrderedFreeVector& vec, bool ascii )
{
    std::vector<char> buf(512);
    for(unsigned j=0 ; j < buf.size() ; j++ ) buf[j] = '*' ; 

    std::ofstream fp("/dev/null", std::ios::out); 

    fp.rdbuf()->pubsetbuf(buf.data(),buf.size());

    vec.Store(fp, ascii );

    for(unsigned j=0 ; j < buf.size() ; j++ )
    {
        std::cout << " " << buf[j] ; 
        if( (j + 1) % 16 == 0 ) std::cout << std::endl ; 
    }
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    CVec* v = CVec::MakeDummy(5); 

    G4PhysicsOrderedFreeVector& vec  = *v->getVec() ; 

    std::cout << vec << std::endl ; 


    // Making an ofstream writing method write into a buffer 

    bool ascii = false ; 
    //test_caveman(   v, ascii ); 
    test_redirected( vec, ascii ); 



    return 0 ; 
}


