#include "U4Hit_Debug.hh"

int main(int argc, char** argv)
{
    U4Hit_Debug dbg ; 
    for(int i=0 ; i < 10 ; i++)
    {
        dbg.label = { 0, i, 0, 0 } ; 
        dbg.add(); 
    }
    U4Hit_Debug::EndOfEvent(0); 

    return 0 ; 
}
