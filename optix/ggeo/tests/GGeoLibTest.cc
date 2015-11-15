// ggv --geolib

#include "GGeoLib.hh"
#include "GMergedMesh.hh"
#include "GBuffer.hh"
#include "GCache.hh"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"


int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "geolib.log", "info");

    cache->configure(argc, argv);

    GGeoLib* geolib = GGeoLib::load(cache); 

    GMergedMesh* mm = geolib->getMergedMesh(1);


    //GBuffer* itransforms = mm->getITransformsBuffer();
    NPY<float>* itransforms = mm->getITransformsBuffer();

    unsigned int numITransforms = itransforms ? itransforms->getNumItems() : 0  ;

    printf("numITransforms %u \n", numITransforms  );


    NPY<unsigned int>* aii = mm->getAnalyticInstancedIdentityBuffer();

    aii->dump();

    unsigned int ni = aii->getShape(0);
    unsigned int nj = aii->getShape(1);
    unsigned int nk = aii->getShape(2);
    assert(nj == 1 && nk == 4);

    for(unsigned int i=0 ; i < ni ; i++)
    {
        printf("%d\n", i);
        glm::uvec4 q = aii->getQuadU(i, 0) ;
        print_u(q, "aii");
    }



}
