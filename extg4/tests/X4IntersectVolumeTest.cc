/**
X4IntersectVolumeTest
========================

Used from script extg4/xxv.sh 

**/

#include <cstdlib>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "X4Intersect.hh"

#include "G4VSolid.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#include "P4Volume.hh"
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* geom_default = "nnvtBodyPhys" ; 
    const char* geom = SSys::getenvvar("GEOM", geom_default );  
    int rc = 0 ; 

#ifdef WITH_PMTSIM

    typedef std::vector<double> VD ; 
    typedef std::vector<G4VSolid*> VS ; 

    VD* tr = new VD ; 
    VS* so = new VS ; 

    G4VPhysicalVolume* pv = PMTSim::GetPV(geom, tr, so );
    assert(pv); 
    assert( tr->size() % 16 == 0 ); 
    assert( tr->size() == 16*so->size() );  // expect 16 doubles of the transform matrix for every solid

    const char* base = SPath::Resolve("$TMP/extg4/X4IntersectVolumeTest", geom, DIRPATH ) ; 
    P4Volume::DumpTransforms(tr, so, "X4IntersectVolumeTest.DumpTransforms"); 
    P4Volume::SaveTransforms(tr, so, base, "transforms.npy" ); 
    unsigned num = so->size(); 

    for(unsigned i=0 ; i < num ; i++)
    {
        G4VSolid* solid = (*so)[i] ; 
        G4String soname = solid->GetName(); 
        X4Intersect::Scan(solid, soname.c_str(), base ); 
    }
#else
    LOG(fatal) << " not-WITH_PMTSIM : Not implemented for geom " << geom ; 
    rc=1 ;   
#endif
    return rc ; 
}

