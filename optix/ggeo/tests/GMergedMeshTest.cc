//  ggv --mm

#include "Opticks.hh"

#include "GVector.hh"
#include "GMergedMesh.hh"
#include "NLog.hpp"

int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "mm.log");

    GMergedMesh* mm = GMergedMesh::load(&ok, 1);

    mm->Summary("mm loading");
    mm->dump("mm dump", 10);
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


    GBuffer* idbuf = mm->getIdentityBuffer();
    idbuf->dump<unsigned int>("idbuf");

    for(unsigned int i=0 ; i < mm->getNumSolids() ; i++)
    {
        guint4 id = mm->getIdentity(i);
        LOG(info) << id.description() ; 
    }

    //mm->getLow()->Summary("low");
    //mm->getHigh()->Summary("high");


    return 0 ;
}
