//  ggv --mm

#include "GCache.hh"
#include "GMergedMesh.hh"
#include "NLog.hpp"

int main(int argc, char** argv)
{
    GCache gc("GGEOVIEW_", "mm.log", "info");
    gc.configure(argc, argv);

    GMergedMesh* mm = GMergedMesh::load(&gc, 1);

    mm->Summary("mm loading");
    mm->Dump("mm dump", 10);
    mm->dumpSolids("dumpSolids");

    unsigned int numSolids = mm->getNumSolids();
    unsigned int numSolidsSelected = mm->getNumSolidsSelected();

    LOG(info) 
                  << " numSolids " << numSolids       
                  << " numSolidsSelected " << numSolidsSelected ;      


    for(unsigned int i=0 ; i < numSolids ; i++)
    {
        gbbox bb = mm->getBBox(i);
        bb.Summary("bb"); 
    }

    //mm->getLow()->Summary("low");
    //mm->getHigh()->Summary("high");



    return 0 ;
}
