#include "OPTICKS_LOG.hh"

#include "U4Scintillation_Debug.hh"
#include "U4Cerenkov_Debug.hh"
#include "U4Hit_Debug.hh"
#include "U4Debug.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    U4Scintillation_Debug s ; 
    U4Cerenkov_Debug c ; 
    U4Hit_Debug h ; 

    int eventID = 0 ; 

    s.posx = 1. ; 
    s.posy = 1. ; 
    s.posz = 1. ; 
    s.time = 1. ; 

    c.posx = 1. ; 
    c.posy = 1. ; 
    c.posz = 1. ; 
    c.time = 1. ; 

    s.ScintillationYield = 1. ; 
    s.MeanNumberOfTracks = 2. ; 
    s.NumTracks = 3. ; 
    s.Spare = 4. ; 


    h.label = {1,2,3,4} ; 
    h.add(); 
    h.add(); 


    c.BetaInverse = 2. ; 
    c.step_length = 2. ; 
    c.MeanNumberOfPhotons = 2. ; 
    c.fNumPhotons = 2. ; 


    s.add(); 
    s.add(); 
    s.add(); 
    s.add(); 

    c.add(); 
    c.add(); 

    U4Debug::Save(eventID); 


    s.add(); 
    s.add(); 

    c.add(); 

    h.add(); 
  
    eventID += 1 ; 
    U4Debug::Save(eventID); 

    return 0 ; 
}
