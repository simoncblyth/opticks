#include <cassert>
#include <iostream>

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "RecordsNPY.hpp"
#include "PLOG.hh"

// okc-
#include "OpticksEventSpec.hh"
#include "OpticksEvent.hh"


void test_genstep_derivative()
{
    OpticksEventSpec sp("cerenkov", "1", "dayabay", "") ;
    OpticksEvent evt(&sp) ;

    NPY<float>* trk = evt.loadGenstepDerivativeFromFile("track");
    assert(trk);

    LOG(info) << trk->getShapeString();

    glm::vec4 origin    = trk->getQuad(0,0) ;
    glm::vec4 direction = trk->getQuad(0,1) ;
    glm::vec4 range     = trk->getQuad(0,2) ;

    print(origin,"origin");
    print(direction,"direction");
    print(range,"range");

}


void test_genstep()
{   
    OpticksEventSpec sp("cerenkov", "1", "dayabay", "") ;
    OpticksEvent evt(&sp) ;

    evt.setGenstepData(evt.loadGenstepFromFile());

    evt.dumpPhotonData();
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_genstep_derivative();
    //test_genstep();
    return 0 ;
}
