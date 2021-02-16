
#include <iostream>

#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SProc.hh"
#include "NGLM.hpp"
#include "NPX.hpp"

unsigned mock_numsteps( unsigned evt, unsigned scale=1 )
{
    unsigned ns = 0 ; 
    switch( evt % 10 )
    {
       case 0: ns = 10 ; break ;
       case 1: ns = 50 ; break ;
       case 2: ns = 60 ; break ;
       case 3: ns = 80 ; break ;
       case 4: ns = 10 ; break ;
       case 5: ns = 100 ; break ;
       case 6: ns = 30 ; break ;
       case 7: ns = 300 ; break ;
       case 8: ns = 20 ; break ;
       case 9: ns = 10 ; break ;
   }
   return ns*scale ;  
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned mock_numevt = 10 ; 
    NPX<float>* gs = NPX<float>::make(0, 6, 4); 

    unsigned scale = 1000 ; 
    unsigned itemsize = 6*4 ;  
    float gsi[itemsize];


    int reservation = SSys::getenvint("RESERVATION",0) ; 
    for(int i=0 ; i < itemsize ; i++) gsi[i] = float(i); 

    float v0 = SProc::VirtualMemoryUsageMB() ;
    for(unsigned evt=0 ; evt < mock_numevt ; evt++)
    {
        unsigned num_steps = mock_numsteps(evt, scale) ; 

        if(reservation > 0 ) 
        {
            std::cout << " fixed reservation " << reservation << std::endl ; 
            gs->reserve( reservation ); 
        } 
        else if( reservation < 0 )
        {
            std::cout << " cheat and use pre-knowledge of the number of items : " << num_steps ; 
            gs->reserve(num_steps);   
        }  


        for(unsigned i=0 ; i < num_steps ; i++) gs->add(gsi, itemsize);  
        std::cout 
            << " evt " << evt
            << " num_steps " << num_steps
            << " gs " << gs->getShapeString()
            << std::endl 
            ;
        gs->reset(); 
    }

    float v1 = SProc::VirtualMemoryUsageMB() ;
    float dv = v1 - v0 ; 
    float dvp = dv/float(mock_numevt) ;  



    if(reservation > 0 ) 
    {
        std::cout << " +ve reservation : fixed reservation " << reservation << std::endl ; 
    } 
    else if( reservation < 0 )
    {
        std::cout << " -ve reservation : cheat and use pre-knowledge of the number of items for each event  " ; 
    }  


    std::cout 
        << " reservation " << reservation
        << " mock_numevt " << mock_numevt
        << " v0 " << v0 
        << " v1 " << v1
        << " dv " << dv 
        << " dvp " << dvp 
        << std::endl 
        ; 

    return 0 ; 
}



