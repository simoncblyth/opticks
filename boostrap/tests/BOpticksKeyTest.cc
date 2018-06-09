#include "BOpticksKey.hh"
#include "OPTICKS_LOG.hh"

int main( int argc, char** argv )
{
    OPTICKS_LOG_COLOR__(argc, argv ); 

    const char* spec = "X4PhysicalVolumeTest.X4PhysicalVolume.World.3ad454e0990085f20c4689fce16c0819" ; 

    BOpticksKey::SetKey(spec); 

    BOpticksKey* key = BOpticksKey::GetKey(); 
    LOG(info) << key->desc() ; 


    const char* exename = PLOG::instance->args.exename() ; 
    std::cout << " exename " << exename << std::endl ; 


    return 0 ; 
}
