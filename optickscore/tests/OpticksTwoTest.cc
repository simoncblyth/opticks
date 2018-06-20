//  while true; do OpticksTwoTest ; done
//  while OpticksTwoTest ; do echo -n ; done 

#include "OPTICKS_LOG.hh"

#include "BOpticksKey.hh"
#include "Opticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    Opticks ok(argc, argv);
    ok.configure();

    const char* key = "CX4GDMLTest.X4PhysicalVolume.World0xc15cfc0_PV.27c39be4e46a36ea28a3c4da52522c9e" ; 

    BOpticksKey::SetKey(key);

    Opticks ok2(0,0);
    ok2.configure();

    const char* idpath = ok2.getIdPath();

    LOG(info)  << "idpath " << idpath ;


    return 0 ;   
}
