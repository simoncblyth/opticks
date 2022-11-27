/**
X4IntersectVolumeTest
========================

Used from script extg4/xxv.sh 

The call to PMTSim::GetPV populates vectors via pointer to vector arguments 
containing the solids and their transforms, relative to the top level PV presumably. 
(Could be just simple one level). These transforms are saved.  

Subsequently the scanning is applied to each solid individually.

The python level plotting then brings together the solid frame intersects
into the volume frame by applying the saved transforms for each solid.  
 

**/

#include <cstdlib>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "X4Intersect.hh"

#include "G4VSolid.hh"

#ifdef WITH_PMTFASTSIM
#include "PMTFastSim.hh"
#include "N4Volume.hh"
#elif WITH_PMTSIM
#include "PMTSim.hh"
#include "P4Volume.hh"
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //const char* geom_default = "nnvtBodyPhys" ; 
    const char* geom_default = "hamaBodyPhys" ; 
    const char* geom = SSys::getenvvar("X4IntersectVolumeTest_GEOM", geom_default );  
    int rc = 0 ; 

#if defined(WITH_PMTFASTSIM) || defined(WITH_PMTSIM)

    typedef std::vector<double> VD ;  // curious why not use NP ?
    typedef std::vector<G4VSolid*> VS ; 
    VD* tr = new VD ; 
    VS* so = new VS ; 

#ifdef WITH_PMTFASTSIM
    LOG(info) << "[ PMTFastSim::GetPV geom [" << geom << "]" ; 
    G4VPhysicalVolume* pv = PMTFastSim::GetPV(geom, tr, so );
    LOG(info) << "] PMTFastSim::GetPV geom [" << geom << "]" ; 
#elif WITH_PMTSIM
    LOG(info) << "[ PMTSim::GetPV geom [" << geom << "]" ; 
    G4VPhysicalVolume* pv = PMTSim::GetPV(geom, tr, so );
    LOG(info) << "] PMTSim::GetPV geom [" << geom << "]" ; 
#endif
    assert(pv); 
    unsigned num = so->size(); 
    assert( tr->size() % 16 == 0 ); 
    assert( tr->size() == 16*num );  // expect 16 doubles of the transform matrix for every solid

    const char* base = SPath::Resolve("$TMP/extg4/X4IntersectVolumeTest", geom, DIRPATH ) ; 

#ifdef WITH_PMTFASTSIM
    N4Volume::DumpTransforms(tr, so, "X4IntersectVolumeTest.DumpTransforms"); 
    N4Volume::SaveTransforms(tr, so, base, "transforms.npy" ); 
#elif WITH_PMTSIM
    P4Volume::DumpTransforms(tr, so, "X4IntersectVolumeTest.DumpTransforms"); 
    P4Volume::SaveTransforms(tr, so, base, "transforms.npy" ); 
#endif

    LOG(info) << " num " << num ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        G4VSolid* solid = (*so)[i] ; 
        G4String soname_ = solid->GetName(); 
        const char* soname = soname_.c_str() ; 

        LOG(info) << "[ X4Intersect::Scan soname [" << soname << "]" ; 
        X4Intersect::Scan(solid, soname, base ); 
        LOG(info) << "] X4Intersect::Scan soname [" << soname << "]" ; 
    }
#else
    LOG(fatal) << " not-WITH_PMTSIM OR WITH_PMTFASTSIM : cannot do anything without the volume " << geom ; 
    rc=1 ;   
#endif
    return rc ; 
}

