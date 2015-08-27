#include "GCache.hh"
#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"


int main(int argc, char** argv)
{
    GCache gc("GGEOVIEW_");
    unsigned int ridx = 1 ;  
    std::string mmpath = gc.getMergedMeshPath(ridx);
    GMergedMesh* mm = GMergedMesh::load(mmpath.c_str(), ridx);

    mm->Summary("mm loading");
    mm->Dump("mm dump", 10);
    mm->dumpSolids("dumpSolids");

    unsigned int numSolids = mm->getNumSolids();

    LOG(info) << "mm numSolids " << numSolids  ;

    GBBoxMesh* bb = GBBoxMesh::create(mm);


    return 0 ;
}
