#include "OPTICKS_LOG.hh"
#include "U4Scintillation_Debug.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    U4Scintillation_Debug dbg ; 
    dbg.ScintillationYield = 1. ; 
    dbg.MeanNumberOfTracks = 2. ; 
    dbg.NumTracks = 3. ; 
    dbg.Spare = 4. ; 

    dbg.add(); 
    dbg.add(); 
    dbg.add(); 
    dbg.add(); 

    U4Scintillation_Debug::EndOfEvent(0); 

    dbg.add(); 
    dbg.add(); 
    U4Scintillation_Debug::EndOfEvent(1); 


    return 0 ; 
}
