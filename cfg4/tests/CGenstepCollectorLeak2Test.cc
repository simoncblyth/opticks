
#include <iostream>

#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SProc.hh"
#include "NGLM.hpp"

//#define WITH_NPX 1 

#ifdef WITH_NPX
#include "NPX.hpp"
#else
#include "NPY.hpp"
#endif

unsigned mock_numsteps( unsigned evt )
{
    unsigned ns = 0 ; 
    switch( evt % 10 )
    {
       case 0: ns = 3100 ; break ;
       case 1: ns = 3200 ; break ;
       case 2: ns = 3300 ; break ;
       case 3: ns = 3100 ; break ;
       case 4: ns = 3500 ; break ;
       case 5: ns = 3000 ; break ;
       case 6: ns = 3300 ; break ;
       case 7: ns = 3100 ; break ;
       case 8: ns = 3200 ; break ;
       case 9: ns = 3100 ; break ;
   }
   return ns ;  
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned mock_numevt = 10 ; 

#ifdef WITH_NPX
    NPX<float>* gs = NPX<float>::make(0, 6, 4); 
#else
    NPY<float>* gs = NPY<float>::make(0, 6, 4); 
#endif

    unsigned itemsize = 6*4 ;  
    float gsi[itemsize];
    for(int i=0 ; i < int(itemsize) ; i++) gsi[i] = float(i); 

    int reservation = SSys::getenvint("RESERVATION",0) ; 
    gs->setReservation(reservation);  


    float v0 = SProc::VirtualMemoryUsageMB() ;
    float r0 = SProc::ResidentSetSizeMB() ;

    for(unsigned evt=0 ; evt < mock_numevt ; evt++)
    {
        unsigned num_steps = mock_numsteps(evt) ; 

#ifdef MANUAL_RESERVE
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
#endif

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
    float r1 = SProc::ResidentSetSizeMB() ;
    float dv = v1 - v0 ; 
    float dvp = dv/float(mock_numevt) ;  
    float dr = r1 - r0 ; 
    float drp = dr/float(mock_numevt) ;  


#ifdef MANUAL_RESERVE
    if(reservation > 0 ) 
    {
        std::cout << " +ve reservation : fixed reservation " << reservation << std::endl ; 
    } 
    else if( reservation < 0 )
    {
        std::cout << " -ve reservation : cheat and use pre-knowledge of the number of items for each event  " ; 
    }  
#endif

    std::cout 
#ifdef WITH_NPX
        << " NPX "
#else
        << " NPY "
#endif
        << " reservation " << reservation
        << " mock_numevt " << mock_numevt
        << " v0 " << v0 
        << " v1 " << v1
        << " dv " << dv 
        << " dvp " << dvp 
        << "    "
        << " r0 " << r0 
        << " r1 " << r1
        << " dr " << dr 
        << " drp " << drp 
        << std::endl 
        ; 

    return 0 ; 
}



