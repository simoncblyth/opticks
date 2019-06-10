#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "SSys.hh"
#include "BOpticksResource.hh"

#include "NPY.hpp"
#include "HitsNPY.hpp"
#include "NSensorList.hpp"
#include "NSensor.hpp"

#include "OPTICKS_LOG.hh"



struct HitsNPYTest
{
    HitsNPYTest( const char* idpath )
        :
        _testgeo(false), 
        _res(_testgeo)
    {
        
        _res.setupViaID(idpath); 

        const char* idmpath = _res.getIdMapPath(); 
        assert( idmpath ); 

        _sens.load(idmpath);

    }

    void loadPhotons(const char* tag)
    {
        NPY<float>* photons = NPY<float>::load("oxtorch", tag,"dayabay");

        if(!photons)
        {
            LOG(error) << "failed to load photons " ;
            return  ;
        }    

        HitsNPY hits(photons, &_sens );
        hits.debugdump() ; 
    }

    bool             _testgeo ; 
    BOpticksResource _res ; 
    NSensorList      _sens ; 
};



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* idpath = SSys::getenvvar("IDPATH");
    if(!idpath) return 0 ; 

    HitsNPYTest hnt(idpath);
    hnt.loadPhotons("1");


    return 0 ;
}


