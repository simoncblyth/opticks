// ggv --geolib

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"

#include "Opticks.hh"

#include "GBuffer.hh"
#include "GMergedMesh.hh"
#include "GGeoLib.hh"

#include "GGEO_CC.hh"
#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    Opticks* ok = new Opticks(argc, argv);

    GGeoLib* geolib = GGeoLib::load(ok); 

    GMergedMesh* mm = geolib->getMergedMesh(1);

    if(!mm)
    {
        LOG(error) << "NULL mm" ;
        return 0 ; 
    }


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
