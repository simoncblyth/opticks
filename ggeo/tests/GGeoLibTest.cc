// ggv --geolib

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"

#include "Opticks.hh"

#include "GBuffer.hh"
#include "GMergedMesh.hh"
#include "GBndLib.hh"
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
        print(q, "aii"); // _u
    }
}


void test_GlobalMergedMesh(GMergedMesh* mm)
{
    assert(mm->getIndex() == 0);
    //mm->dumpVolumes("test_GlobalMergedMesh");
    unsigned numVolumes = mm->getNumVolumes();
    for(unsigned i=0 ; i < numVolumes ; i++)
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




void test_getFaceRepeatedIdentityBuffer(GMergedMesh* mm)
{
    GBuffer* buf = mm->getFaceRepeatedIdentityBuffer();
    assert(buf);
}
void test_getFaceRepeatedInstancedIdentityBuffer(GMergedMesh* mm)
{
    GBuffer* buf = mm->getFaceRepeatedInstancedIdentityBuffer();
    assert(buf);
}


void test_GGeoLib(GGeoLib* geolib)
{
    unsigned nmm = geolib->getNumMergedMesh();
    for(unsigned i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = geolib->getMergedMesh(i);
        if(!mm) return ;  

        if(i == 0)
        {
            //test_GlobalMergedMesh(mm);
            test_getFaceRepeatedIdentityBuffer(mm);
        }
        else
        {
            //test_InstancedMergedMesh(mm);
            test_getFaceRepeatedInstancedIdentityBuffer(mm);
        }
    }
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG__ ;

    Opticks ok(argc, argv);


    bool constituents = true ; 
    GBndLib* bndlib = GBndLib::load(&ok, constituents);

    bool analytic ; 
    analytic = false ; 
    GGeoLib* geolib = GGeoLib::Load(&ok, analytic, bndlib); 
    geolib->dump("geolib");
    test_GGeoLib(geolib);


    analytic = true ; 
    GGeoLib* geolib_analytic = GGeoLib::Load(&ok, analytic, bndlib ); 
    geolib_analytic->dump("geolib_analytic");   
    test_GGeoLib(geolib_analytic);

    return 0 ; 
}


