//  op --mm
//  op --j1707 --mm
//  op --j1707 --mm --debugger
//
//
#include "Opticks.hh"

#include "GVector.hh"
#include "GMergedMesh.hh"
#include "PLOG.hh"
#include "GGEO_LOG.hh"



void test_GMergedMesh_Dump(GMergedMesh* mm)
{
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
}



void test_GMergedMesh_MakeComposite(GMergedMesh* mm)
{
    std::vector<GMergedMesh*> mms ; 
    mms.push_back(mm);
    mms.push_back(mm);

    GMergedMesh* comp = GMergedMesh::MakeComposite(mms);

    comp->dumpSolids("test_GMergedMesh_MakeComposite.dumpSolids");
    comp->dumpComponents("test_GMergedMesh_MakeComposite.dumpComponents");

    const char* dir = "$TMP/test_GMergedMesh_MakeComposite" ; 
    comp->save(dir);    

    GMergedMesh* comp2 = GMergedMesh::load(dir);

    comp2->dumpSolids("test_GMergedMesh_MakeComposite.dumpSolids.comp2");
    comp2->dumpComponents("test_GMergedMesh_MakeComposite.dumpComponents.comp2");

}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG__ ;

    Opticks ok(argc, argv);

    GMergedMesh* mm = GMergedMesh::load(&ok, 1);

    if(!mm)
    {
        LOG(error) << "NULL mm" ;
        return 0 ; 
    } 

    //test_GMergedMesh_Dump(mm); 
    test_GMergedMesh_MakeComposite(mm); 


    return 0 ;
}
