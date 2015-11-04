#include "GCache.hh"
#include "GMergedMesh.hh"
#include "NLog.hpp"

int main(int argc, char** argv)
{
    GCache gc("GGEOVIEW_");
    GMergedMesh* mm = GMergedMesh::load(&gc, 1);

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
