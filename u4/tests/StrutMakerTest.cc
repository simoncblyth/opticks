#include "StrutMaker.h"
#include "U4Mesh.h"

int main()
{
    const char* name = getenv("SOLID");
    const G4VSolid* solid = StrutMaker::Make(name);

    NPFold* fold = U4Mesh::Serialize(solid) ;
    fold->set_meta<std::string>("name",name);
    fold->set_meta<std::string>("desc",name);
    fold->save("$FOLD", name );

    return 0 ;
}

