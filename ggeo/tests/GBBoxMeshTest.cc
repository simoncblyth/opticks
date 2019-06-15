#include "Opticks.hh"
#include "OpticksResource.hh"

#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();

    unsigned int ridx = 1 ;  

    OpticksResource* resource = ok.getResource();

    std::string mmpath = resource->getMergedMeshPath(ridx);
    GMergedMesh* mm = GMergedMesh::Load(mmpath.c_str(), ridx);
    if(!mm)
    {
        LOG(error) << "NULL mm" ;
        return 0 ; 
    } 
    

    mm->Summary("mm loading");
    mm->dump("mm dump", 10);
    mm->dumpVolumes("dumpVolumes");

    unsigned int numVolumes = mm->getNumVolumes();

    LOG(info) << "mm numVolumes " << numVolumes  ;

    //GBBoxMesh* bb = GBBoxMesh::create(mm);


    return 0 ;
}
