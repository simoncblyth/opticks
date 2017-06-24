// ggv --geolib

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"

#include "Opticks.hh"

#include "GBuffer.hh"
#include "GMergedMesh.hh"
#include "GGeoLib.hh"

#include "GGEO_BODY.hh"
#include "PLOG.hh"
#include "GGEO_LOG.hh"


void test_InstancedMergedMesh(GMergedMesh* mm)
{
    assert(mm->getIndex() > 0);

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


void test_GlobalMergedMesh(GMergedMesh* mm)
{
    assert(mm->getIndex() == 0);
    //mm->dumpSolids("test_GlobalMergedMesh");
    unsigned numSolids = mm->getNumSolids();
    for(unsigned i=0 ; i < numSolids ; i++)
    {
        guint4 nodeinfo = mm->getNodeInfo(i);
        unsigned nface = nodeinfo.x ;
        unsigned nvert = nodeinfo.y ;
        unsigned node = nodeinfo.z ;
        unsigned parent = nodeinfo.w ;
        assert( node == i );

        guint4 id = mm->getIdentity(i);
        unsigned node2 = id.x ;
        unsigned mesh = id.y ;
        unsigned boundary = id.z ;
        unsigned sensor = id.w ;
        assert( node2 == i );
        
        guint4 iid = mm->getInstancedIdentity(i);  // nothing new for GlobalMergedMesh 
        assert( iid.x == id.x );
        assert( iid.y == id.y );
        assert( iid.z == id.z );
        assert( iid.w == id.w );

        std::cout 
             << " " << std::setw(8) << i 
             << " ni[" 
             << " " << std::setw(6) << nface
             << " " << std::setw(6) << nvert 
             << " " << std::setw(6) << node
             << " " << std::setw(6) << parent
             << " ]"
             << " id[" 
             << " " << std::setw(6) << node2
             << " " << std::setw(6) << mesh
             << " " << std::setw(6) << boundary
             << " " << std::setw(6) << sensor
             << " ]"
             << std::endl 
             ;
    }
}






int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    Opticks ok(argc, argv);

    GGeoLib* geolib = GGeoLib::load(&ok); 

    GMergedMesh* mm0 = geolib->getMergedMesh(0);
    GMergedMesh* mm1 = geolib->getMergedMesh(1);

    test_InstancedMergedMesh(mm1);
    test_GlobalMergedMesh(mm0);


    return 0 ; 
}
