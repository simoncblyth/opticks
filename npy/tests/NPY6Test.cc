#include <iostream>
#include <iomanip>
#include "OPTICKS_LOG.hh"
#include "SProc.hh"
#include "NPY.hpp"

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

struct test_grow_leak
{
    NPY<float>* a ;
    unsigned itemsize ; 
    float* gs ; 


    void stamp(unsigned i)
    {
        float vm = SProc::VirtualMemoryUsageMB() ;
        std::cout 
            << std::setw(5) << i  
            << " : "
            << vm 
            << " : "
            << a->getShapeString()
            << std::endl ; 
            ;
    }

    test_grow_leak()
        :
        a(NPY<float>::make(0,6,4)),
        itemsize(6*4),
        gs(new float[itemsize])  
    {
        for(unsigned i=0 ; i < itemsize ; i++) gs[i] = float(i) ; 
    }

    void one(unsigned i)
    {
        unsigned numsteps = mock_numsteps(i, 1000);  
        for(unsigned j=0 ; j < numsteps ; j++) a->add(gs, itemsize) ;  // mimic collecting gensteps        
        stamp(i); 
        a->reset();  
    }

    void many()
    {
        for(unsigned i=0 ; i < 100 ; i++) one(i); 
    }
};


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_grow_leak tgl ; 
    tgl.many(); 

    return 0 ; 
}

// om-;TEST=NPY6Test om-t

