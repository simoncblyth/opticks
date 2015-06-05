#include "GLoader.hh"
#include "GMergedMesh.hh"
#include "GGeo.hh"


int main(int argc, char** argv)
{
    bool nogeocache = true ; 

    GLoader loader ; 
    loader.load("GGEOVIEW_", nogeocache);

    GMergedMesh* mm = loader.getMergedMesh();
    mm->dumpSolids(); 

    GGeo*  gg = loader.getGGeo();
    if(!gg) return 1 ; 

    gg->Summary();
    //gg->dumpRaw();

    GPropertyMap<float>* scint = gg->findRawMaterial("LiquidScintillator");
    //scint->Summary();

    GProperty<float>* slow = scint->getProperty("SLOWCOMPONENT");
    slow->Summary();

    // hmm should have permanent slot in idpath 
    slow->save("/tmp/slowcomponent.npy");


    return 0 ;
}

