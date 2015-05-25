#include "Geometry.hh"
#include "GMergedMesh.hh"
#include "GGeo.hh"

// TODO: move into GGeo

int main(int argc, char** argv)
{
    bool nogeocache = true ; 

    Geometry geometry ; 
    geometry.load("GGEOVIEW_", nogeocache);

    GMergedMesh* mm = geometry.getMergedMesh();
    mm->dumpSolids(); 

    GGeo*  gg = geometry.getGGeo();
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

