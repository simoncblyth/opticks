#include "SSys.hh"
#include "SProc.hh"
#include "NLookup.hpp"
#include "NGLM.hpp"
#include "NPY.hpp"
#include "OpticksGenstep.h"
#include "OpticksProfile.hh"
#include "CGenstepCollector.hh"
#include "OPTICKS_LOG.hh"

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


//#define WITH_STAMP 1 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned mock_numevt = 10 ; 

#ifdef WITH_COLLECTOR
    NLookup* lookup = NULL ; 
    CGenstepCollector cc(lookup);
    NPY<float>* gs = cc.getGensteps(); 
#else
    NPY<float>* gs = NPY<float>::make(0, 6, 4); 
#endif

    unsigned scale = 1000 ; 
    unsigned itemsize = 6*4 ;  
    float gsi[itemsize];
    for(int i=0 ; i < int(itemsize) ; i++) gsi[i] = float(i); 


    float v0 = SProc::VirtualMemoryUsageMB() ;
#ifdef WITH_STAMP
    std::vector<float> stamps ; 
#endif

    for(unsigned evt=0 ; evt < mock_numevt ; evt++)
    {
        unsigned num_steps = mock_numsteps(evt, scale); 
        for(unsigned i=0 ; i < num_steps ; i++) 
        {
#ifdef WITH_COLLECTOR
            cc.collectMachineryStep(gentype);
#else
            gs->add(gsi, itemsize);  
#endif
        }

        std::cout 
            << " evt " << evt
            << " num_steps " << num_steps
            << " gs " << gs->getShapeString()
            << std::endl 
            ;

#ifdef WITH_COLLECTOR
        cc.reset(); 
#else
        gs->reset(); 
#endif

#ifdef WITH_STAMP
        glm::vec4 stamp = OpticksProfile::Stamp() ; 
        stamps.push_back(stamp.x); 
        stamps.push_back(stamp.y); 
        stamps.push_back(stamp.z); 
        stamps.push_back(stamp.w); 
#endif
    }
  
#ifdef WITH_STAMP
    NPY<float>* a = NPY<float>::make_from_vec(stamps); 
    a->reshape(-1,4); 
    a->dump(); 
    OpticksProfile::Report(a);
#endif

    float v1 = SProc::VirtualMemoryUsageMB() ;
    float dv = v1 - v0 ; 
    float dvp = dv/float(mock_numevt) ;  

    std::cout 
        << " mock_numevt " << mock_numevt
        << " v0 " << v0 
        << " v1 " << v1
        << " dv " << dv 
        << " dvp " << dvp 
        << std::endl 
        ; 

    return 0 ; 
}



