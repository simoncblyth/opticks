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

    s.fill(1.); 
    c.fill(1.) ; 


    h.label = {1,2,3,4} ; 
    h.add(); 
    h.add(); 


    c.fill(2.); 

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
