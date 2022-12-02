#include <iostream>
#include <cstdlib>

#include "SPhoton_Debug.h"


//const char A = 'A' ; 
//const char B = 'B' ; 
template<> std::vector<SPhoton_Debug<'A'>> SPhoton_Debug<'A'>::record = {} ;
template<> std::vector<SPhoton_Debug<'B'>> SPhoton_Debug<'B'>::record = {} ;

const char* FOLD = getenv("FOLD"); 

int main(int argc, char** argv)
{
    SPhoton_Debug<'A'> dbg0 ; 
    SPhoton_Debug<'B'> dbg1 ; 

    std::cout 
        << " sizeof(SPhoton_Debug<'A'>) " <<  sizeof(SPhoton_Debug<'A'>) 
        << std::endl  
        << " sizeof(SPhoton_Debug<'B'>) " <<  sizeof(SPhoton_Debug<'B'>) 
        << std::endl  
        << " sizeof(dbg0) " <<  sizeof(dbg0) 
        << std::endl  
        << " sizeof(dbg1) " <<  sizeof(dbg1) 
        << std::endl  
        << " sizeof(double)*16 " << sizeof(double)*16 
        << std::endl 
        ; 

    dbg0.fill(0.); 
    dbg0.add(); 

    dbg1.fill(0.); 
    dbg1.add(); 
    dbg1.add(); 


    dbg0.fill(1.); 
    dbg0.add(); 

    dbg1.fill(1.); 
    dbg1.add(); 
    dbg1.add(); 


    dbg0.fill(2.); 
    dbg0.add(); 

    dbg1.fill(2.); 
    dbg1.add(); 
    dbg1.add(); 


    SPhoton_Debug<'A'>::Save(FOLD) ; 
    SPhoton_Debug<'B'>::Save(FOLD) ; 

    return 0 ; 
}
