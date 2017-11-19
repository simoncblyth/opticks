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

/*
   not compiling anymore
    evt.setGenstepData(evt.loadGenstepFromFile());
    evt.dumpPhotonData();
*/
}

void test_appendNote()
{   
    OpticksEventSpec sp("cerenkov", "1", "dayabay", "") ;
    OpticksEvent evt(&sp) ;

    evt.appendNote("hello");
    evt.appendNote("world");

    std::string note = evt.getNote();

    bool match = note.compare("hello world") == 0 ;
    if(!match) LOG(fatal) << "got unexpected " << note ; 
    assert(match);

}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_genstep_derivative();
    //test_genstep();
    test_appendNote();
    return 0 ;
}
