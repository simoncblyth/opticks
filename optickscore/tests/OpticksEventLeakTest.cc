#include <cassert>
#include <iostream>

#include "OPTICKS_LOG.hh"
#include "SProc.hh"

#include "NPY.hpp"
#include "GLMPrint.hpp"

#include "Opticks.hh"
#include "OpticksProfile.hh"
#include "OpticksEventSpec.hh"
#include "OpticksEvent.hh"
#include "OpticksGenstep.hh"


NPY<float>* test_resetEvent(Opticks* ok, unsigned nevt, char ctrl)
{
    unsigned num_photons = 10000 ; 
    unsigned tagoffset = 0 ; 

    NPY<float>* gs = OpticksGenstep::MakeCandle(num_photons, tagoffset ) ;


#ifdef USE_VEC4
    std::vector<glm::vec4> stamps ; 
#else
    std::vector<float> stamps ; 
#endif

    for(unsigned i=0 ; i < nevt ; i++)
    {
        ok->createEvent(gs, ctrl); 
        ok->resetEvent(ctrl); 
        glm::vec4 stamp = OpticksProfile::Stamp() ; 
#ifdef USE_VEC4
        stamps.push_back(stamp);  
#else
        stamps.push_back(stamp.x);  
        stamps.push_back(stamp.y);  
        stamps.push_back(stamp.z);  
        stamps.push_back(stamp.w);  
#endif
    }

#ifdef USE_VEC4
    NPY<float>* a = NPY<float>::make(stamps) ;  
#else
    NPY<float>* a = NPY<float>::make_from_vec(stamps) ;  
    a->reshape(-1,4); 
#endif

    delete gs ; 

    return a ; 
}


int main(int argc, char** argv)
{
    int nevt = argc > 1 ? atoi(argv[1]) : 1000 ; 

    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 
    ok.configure(); 

    glm::vec4 space_domain(0.f,0.f,0.f,1000.f); 
    ok.setSpaceDomain(space_domain); 

    char ctrl = '+' ; 
    NPY<float>* a = test_resetEvent(&ok, nevt, ctrl); 
    const char* path = "$TMP/optickscore/tests/OpticksEventLeakTest.npy" ; 
    LOG(info) << " save to path " << path ; 
    LOG(info) << " make plot with: ipython -i ~/opticks/optickscore/tests/OpticksEventLeakTest.py "  ;
    a->save(path); 

    unsigned num_stamp = a->getNumItems(); 
    float t0 = a->getValue( 0, 0, 0) ; 
    float t1 = a->getValue(-1, 0, 0) ; 
    float v0 = a->getValue( 0, 1, 0) ; 
    float v1 = a->getValue(-1, 1, 0) ; 

    float dt = t1 - t0 ; 
    float dv = v1 - v0 ; 
    float dt_per_stamp = dt/float(num_stamp); 
    float dv_per_stamp = dv/float(num_stamp); 

    std::cout 
        << " num_stamp " << num_stamp
        << "    "
        << " t0 " << t0 
        << " t1 " << t1 
        << " dt " << dt 
        << " dt/num_stamp " << dt_per_stamp
        << "    "
        << " v0 " << v0 
        << " v1 " << v1 
        << " dv " << dv 
        << " dv/num_stamp " << dv_per_stamp
        << std::endl
        ;


    return 0 ;
}
