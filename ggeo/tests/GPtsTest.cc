// TEST=GPtsTest om-t

#include "OPTICKS_LOG.hh"

#include "Opticks.hh"
#include "GPt.hh"
#include "GPts.hh"
#include "GParts.hh"

#include "GBndLib.hh"
#include "GMeshLib.hh"
#include "GMergedMesh.hh"
#include "GGeoLib.hh"

/**
GPtsTest
============

This is checking the postcache creation of a merged GParts instance
using the persisted higher level GPts. 

See notes/issues/x016.rst 

**/


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv, "--envkey" );
    ok.configure();
    if(!ok.isDirect())
    {
        LOG(fatal) << "this is a direct only test : that means use --envkey option and have a valid OPTICKS_KEY envvar "  ; 
        return 0 ; 
    }

    GMeshLib* meshlib = GMeshLib::Load(&ok);
    meshlib->dump(); 
    const std::vector<const NCSG*>& solids = meshlib->getSolids(); 
    LOG(info) << " meshlib.solids " << solids.size(); 

    bool constituents = true ; 
    GBndLib* bndlib = GBndLib::load(&ok, constituents);
    bndlib->closeConstituents();   // required otherwise GParts::close asserts
    
    //GMaterialLib* mlib = bndlib->getMaterialLib(); 
    //GSurfaceLib* slib = bndlib->getSurfaceLib(); 

    bool analytic = false ;   // <-- funny need to say false to get smth, TODO: eliminate this, all libs now analytic 
    GGeoLib* geolib = GGeoLib::Load(&ok, analytic, bndlib); 
    geolib->dump("geolib");


    unsigned nmm = geolib->getNumMergedMesh(); 
    LOG(info) << " geolib.nmm " << nmm ; 

    GMergedMesh* mm = geolib->getMergedMesh(nmm-1);  // last one is sFastener
    assert(mm); 

    GParts* parts = mm->getParts();  
    parts->dump("parts"); 
    parts->save("$TMP/GGeo/GPtsTest/parts"); 

    GPts* pts = mm->getPts(); 
    pts->dump("pts"); 

    unsigned verbosity = 1 ; 
    GParts* parts2 = GParts::Create( pts, solids, verbosity  ); 
    assert( parts2 ); 
    parts2->setBndLib(bndlib); 
    parts2->close(); 
    parts2->dump("parts2"); 
    parts2->save("$TMP/GGeo/GPtsTest/parts2"); 

    int rc = GParts::Compare( parts, parts2 ); 

    return rc ;
}

