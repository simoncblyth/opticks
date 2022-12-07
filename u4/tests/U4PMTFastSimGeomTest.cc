#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"

#include "OPTICKS_LOG.hh"
#include "stree.h"
#include "SPath.hh"
#include "SSimtrace.h"

#include "U4VolumeMaker.hh"
#include "U4Tree.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* base = SPath::Resolve("$FOLD", DIRPATH ) ;

    const G4VPhysicalVolume* pv = U4VolumeMaker::PV();  // sensitive to GEOM envvar 

    stree st ; 
    U4Tree ut(&st, pv ) ;
    st.save_trs(base); 

    //LOG(info) << st.desc() ; 
    //LOG(info) << st.desc_solids() ; 

    assert( st.soname.size() == ut.solids.size() ); 
    for(unsigned i=0 ; i < st.soname.size() ; i++)  // over unique solid names
    {
        const char* soname = st.soname[i].c_str(); 
        const G4VSolid* solid = ut.solids[i] ; 
        G4String name = solid->GetName(); 
        assert( strcmp( name.c_str(), soname ) == 0 ); 
        SSimtrace::Scan(solid, base) ;   
    }


    return 0 ; 
}
