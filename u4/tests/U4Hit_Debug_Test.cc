#include "U4Hit_Debug.hh"

int main(int argc, char** argv)
{
    U4Hit_Debug dbg ; 
    for(int i=0 ; i < 10 ; i++)
    {
        dbg.label = { 0, i, 0, {0,0,0,0} } ; 
        dbg.add(); 
    }
    U4Hit_Debug::Save("/tmp/U4Hit_Debug/000"); 

    return 0 ; 
}
