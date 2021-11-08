#include "G4VSolid.hh"

#include "GeoChain.h"

#include <cstring>

#include "SSys.hh"
#include "SPath.hh"

#include "GMesh.hh"
#include "GGeo.hh"
#include "X4PhysicalVolume.hh"

#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"

#include "PLOG.hh"


const plog::Severity GeoChain::LEVEL = PLOG::EnvLevel("GeoChain", "DEBUG") ; 

GeoChain::GeoChain(Opticks* ok_)
    :
    ok(ok_), 
    ggeo(new GGeo(ok, true)),  // live=true to initLibs and not load from cache
    mesh(nullptr),
    volume(nullptr),
    fd(new CSGFoundry) 
{
}

void GeoChain::convert(const G4VSolid* const solid, const std::string& meta_)
{
    const char* meta = meta_.empty() ? nullptr : meta_.c_str() ; 
    LOG(info) << "[ meta " << meta ; 

    int lvIdx = 0 ; 
    int soIdx = 0 ; 

    std::string lvname = solid->GetName(); 

    mesh = X4PhysicalVolume::ConvertSolid(ok, lvIdx, soIdx, solid, lvname ) ; 
    LOG(info) << " mesh " << mesh ; 

    ggeo->add(mesh); 

    // standin for X4PhysicalVolume::convertStructure
    volume = X4PhysicalVolume::MakePlaceholderNode(); 
    ggeo->setRootVolume(volume);

    ggeo->prepareVolumes();   // creates GMergedMesh 

    ggeo->deferredCreateGParts(); 

    CSG_GGeo_Convert conv(fd, ggeo, meta ) ; 
    conv.convert();

    LOG(info) << "]" ;  
}

void GeoChain::save(const char* name) const 
{
    int create_dirs = 2 ; // 2: dirpath
    const char* fold = SPath::Resolve("$TMP/GeoChain", name, create_dirs );   
    const char* cfbase = SSys::getenvvar("CFBASE", fold  );
    const char* rel = "CSGFoundry" ; 

    fd->write(cfbase, rel );    // expects existing directory $CFBASE/CSGFoundry 

    CSGFoundry* lfd = CSGFoundry::Load(cfbase, rel);  // load foundary and check identical bytes
    assert( 0 == CSGFoundry::Compare(fd, lfd ) );  
}



