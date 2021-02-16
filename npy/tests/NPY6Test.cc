#include <iostream>
#include <iomanip>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SProc.hh"

#include "NPX.hpp"
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


unsigned mock_numsteps2( unsigned evt )
{
    unsigned ns = 0 ; 
    switch( evt % 10 )
    {   
       case 0: ns = 3271 ; break ;
       case 1: ns = 3270 ; break ;
       case 2: ns = 3057 ; break ;
       case 3: ns = 3453 ; break ;
       case 4: ns = 3459 ; break ;
       case 5: ns = 3362 ; break ;
       case 6: ns = 3111 ; break ;
       case 7: ns = 3702 ; break ;
       case 8: ns = 3479 ; break ;
       case 9: ns = 3500 ; break ;
   }   
   return ns ;   
}





struct test_grow_leak
{
#ifdef WITH_NPX
    NPX<float>* a ;
#else
    NPY<float>* a ;
#endif
    unsigned itemsize ; 
    float* gs ; 
    int nevt ; 
    int reservation ; 
    unsigned scale ; 

    test_grow_leak()
        :
#ifdef WITH_NPX
        a(NPX<float>::make(0,6,4)),
#else
        a(NPY<float>::make(0,6,4)),
#endif
        itemsize(6*4),
        gs(new float[itemsize]),
        nevt(SSys::getenvint("NEVT",10)),
        reservation(SSys::getenvint("RESERVATION",0)),
        scale(1000)
    {

        std::cout 
            << " nevt " << nevt 
            << " reservation " << reservation
            << std::endl
            ;

        a->setReservation( reservation );
 
        for(unsigned i=0 ; i < itemsize ; i++) gs[i] = float(i) ; 
    }


    float vm(){  return SProc::VirtualMemoryUsageMB() ; }
    float rss(){ return SProc::ResidentSetSizeMB() ; }

    void stamp(unsigned i)
    {
        std::cout 
            << std::setw(5) << i  
            << " : "
            << vm()
            << " : "
            << rss()
            << " : "
            << a->getShapeString()
            << std::endl ; 
            ;
    }

    unsigned get_numsteps(unsigned  i)
    {
        //unsigned numsteps = mock_numsteps(i, scale);  
        unsigned numsteps = mock_numsteps2(i);  
        return numsteps ;
    }

    void one(int i)
    {
        unsigned numsteps = get_numsteps(i) ;
        for(unsigned j=0 ; j < numsteps ; j++) a->add(gs, itemsize) ;  // mimic collecting gensteps        
        stamp(i); 
        a->reset();  
    }

    void dump(int i)
    {
        unsigned numsteps = get_numsteps(i) ;  
        std::cout 
            << " i " << std::setw(3) << i 
            << " numsteps " << numsteps 
            << std::endl 
            ; 
    }

    void many()
    {  
        for(int i=0 ; i < nevt ; i++) dump(i); 
 
        float vm0 = vm();  
        float rss0 = rss();  
        for(int i=0 ; i < nevt ; i++) one(i); 
        float vm1 = vm();
        float rss1 = rss();  
    
        float dvm = vm1 - vm0 ; 
        float dvm_nevt = dvm/float(nevt); 

        float drss = rss1 - rss0 ; 
        float drss_nevt = drss/float(nevt); 


        for(int i=0 ; i < nevt ; i++) dump(i); 

        std::cout 
            << " reservation " << reservation 
            << " nevt " << nevt 
            << " vm0 " << vm0  
            << " vm1 " << vm1
            << " dvm " << dvm 
            << " dvm_nevt " << dvm_nevt
            << " rss0 " << rss0  
            << " rss1 " << rss1
            << " drss " << drss 
            << " drss_nevt " << drss_nevt
            << std::endl
            ; 
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

