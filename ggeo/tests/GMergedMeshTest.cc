//  ggv --mm

#include "Opticks.hh"

#include "GVector.hh"
#include "GMergedMesh.hh"
#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    Opticks ok(argc, argv);

    GMergedMesh* mm = GMergedMesh::load(&ok, 1);

    if(!mm)
    {
        LOG(error) << "NULL mm" ;
        return 0 ; 
    } 


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
