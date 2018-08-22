#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CLHEP/Random/MixMaxRng.h"
#include "CRandomListener.hh"

struct CFG4_API CMixMaxRng :  public CRandomListener, public CLHEP::MixMaxRng
{
    CMixMaxRng();  

    double flat();

    // CRandomListener 
    void preTrack() ; 
    void postTrack(); 
    void postStep() ; 
    void postpropagate(); 
    double flat_instrumented(const char* file, int line);

    unsigned count ; 

};




