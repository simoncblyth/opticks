#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "BOpticksResource.hh"

#include "NPY.hpp"
#include "HitsNPY.hpp"
#include "NSensorList.hpp"
#include "NSensor.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 


    const char* idmpath = BOpticksResource::IdMapSrcPath(); 
    if(!idmpath) 
    {
        printf("%s : requires OPTICKS_SRCPATH  envvar \n", argv[0]);
        return 0 ;
    }


   

    NSensorList sens;
    sens.load(idmpath);

    const char* tag = "1" ; 
    NPY<float>* photons = NPY<float>::load("oxtorch", tag,"dayabay");

    if(!photons)
    {
        printf("%s : failed to load photons \n", argv[0]);
        return 0 ;
    }    

    HitsNPY hits(photons, &sens );
    hits.debugdump();


    return 0 ;
}


