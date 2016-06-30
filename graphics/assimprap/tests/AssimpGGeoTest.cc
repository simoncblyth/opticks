#include <cassert>

#include "AssimpGGeo.hh"
#include "AssimpTree.hh"
#include "AssimpSelection.hh"
#include "AssimpImporter.hh"

#include "GMesh.hh"
#include "GGeo.hh"

#include "OpticksQuery.hh"
#include "OpticksResource.hh"
#include "Opticks.hh"

#include "GGEO_LOG.hh"
#include "ASIRAP_LOG.hh"
#include "PLOG.hh"

// cf canonical int AssimpGGeo::load(GGeo*) 

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    GGEO_LOG__ ;
    ASIRAP_LOG__ ;

    Opticks ok(argc, argv);

    ok.configure();

    const char* path = ok.getDAEPath();

    OpticksResource* resource = ok.getResource();

    OpticksQuery* query = ok.getQuery() ;

    const char* ctrl = resource->getCtrl() ;

    LOG(info)<< "AssimpGGeoTest "  
             << " path " << ( path ? path : "NULL" ) 
             << " query " << ( query ? query->getQueryString() : "NULL" )
             << " ctrl " << ( ctrl ? ctrl : "NULL" )
             ;    

    assert(path);
    assert(query);
    assert(ctrl);


    GGeo* ggeo = new GGeo(&ok);

    LOG(info) << " import " << path ; 

    AssimpImporter assimp(path);

    assimp.import();

    LOG(info) << " select " ; 

    AssimpSelection* selection = assimp.select(query);

    AssimpTree* tree = assimp.getTree();

    AssimpGGeo agg(ggeo, tree, selection); 

    LOG(info) << " convert " ; 

    int rc = agg.convert(ctrl);

    LOG(info) << " convert DONE " ; 
     
    assert(rc == 0);

    unsigned nmesh = agg.getNumMeshes();

    for(unsigned i=0 ; i < nmesh ; i++)
    {
        GMesh* gm = agg.convertMesh(i); 
        gm->Summary();
    }

    GMesh* iav = agg.convertMesh("iav");
    GMesh* oav = agg.convertMesh("oav");

    if(iav) iav->Summary("iav");
    if(oav) oav->Summary("oav");

    LOG(info) << " DONE " << argv[0] ; 

    return 0 ;
}

