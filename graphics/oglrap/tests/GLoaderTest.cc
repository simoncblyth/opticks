#include "GLoader.hh"
#include "GMergedMesh.hh"
#include "GGeo.hh"

// this test is here in oglrap- rather than ggeo- due to dependency on AssimpWrap 
#include "AssimpWrap/AssimpGGeo.hh"


int main(int argc, char** argv)
{
    bool nogeocache = true ; 

    GLoader loader ; 

    loader.setImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    loader.load("GGEOVIEW_", nogeocache);

    GMergedMesh* mm = loader.getMergedMesh();

    mm->dumpSolids(); 

    GGeo*  gg = loader.getGGeo();
    if(!gg) return 1 ; 

    gg->Summary();
    //gg->dumpRaw();

    GProperty<float>* slow = gg->findRawMaterialProperty("LiquidScintillator", "SLOWCOMPONENT");
    slow->save("/tmp/slowcomponent.npy");   // hmm should have permanent slot in idpath 


    gg->dumpRawSkinSurface();

    gg->dumpRawBorderSurface();



    return 0 ;
}

