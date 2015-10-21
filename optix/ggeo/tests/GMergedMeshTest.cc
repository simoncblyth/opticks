#include "GCache.hh"
#include "GMergedMesh.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


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
    unsigned int numSolidsSelected = mm->getNumSolidsSelected();

    LOG(info) 
                  << " numSolids " << numSolids       
                  << " numSolidsSelected " << numSolidsSelected ;      



    return 0 ;
}
