#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GGeo.hh"
#include "GGeoLib.hh"
#include "GGeoTest.hh"

const char* USAGE = R"TOKEN(
GGeoTestTest is intended for non-GPU test geometry preparation and 
machinery testing. It requires many commandline options including
"--test" and "--testconfig" and python preparation of 
a serialized geometry directory.  

Do all this with the tboolean- functions, eg::

    tboolean-;GGeoTest=INFO GGeoLib=INFO tboolean-box --ggeotesttest
 
)TOKEN";


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure(); 

    bool is_test = ok.isTest(); 
    if(!is_test)
    {
       LOG(info) << USAGE ; 
       return 0 ; 
    }

    GGeo gg(&ok);
    gg.loadFromCache();
    gg.dumpStats();

    GGeoBase* basis = dynamic_cast<GGeoBase*>(&gg);
    GGeoTest tgeo(&ok, basis);

    unsigned nmm = tgeo.getNumMergedMesh(); 
    assert( nmm == 1 ); 

    GMergedMesh* mm = tgeo.getMergedMesh(0); 
    assert( mm ); 
    LOG(info) << " mm " << mm ; 

    GGeoLib* glib = tgeo.getGeoLib(); 
    glib->dryrun_convert(); 

    return 0 ;
}

/**
tboolean-;GGeoTest=INFO GGeoLib=INFO tboolean-box --ggeotesttest
**/
