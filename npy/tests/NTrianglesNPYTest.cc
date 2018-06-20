
#include "NBBox.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "NTrianglesNPY.hpp"

#include "OPTICKS_LOG.hh"

void test_prism()
{
    LOG(info) << "test_prism" ; 

    glm::vec4 param(90,100,100,200);
    NTrianglesNPY* m = NTrianglesNPY::prism(param);
    m->getTris()->dump("prism");
    m->getTris()->save("$TMP/prism.npy");
/*
In [1]: fig = plt.figure()
In [2]: xyz3d(fig, "$TMP/prism.npy")
In [3]: plt.show()
*/
}

void test_transform()
{
    LOG(info) << "test_transform" ; 

    NTrianglesNPY* c = NTrianglesNPY::cube();
    c->getTris()->dump("cube");

    glm::vec3 tr(100.,0.,0);
    glm::vec3 sc(10.,10.,10.);

   // scale and then translate   (translation not scaled)
    glm::mat4 m = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);

    print(m, "m");

    NTrianglesNPY* tc = c->transform(m);
    tc->getTris()->dump("tcube");
}

void test_latlon()
{
    LOG(info) << "test_latlon" ; 

    unsigned n_polar = 24 ; 
    unsigned n_azimuthal = 2 ; 

    NTrianglesNPY* s = NTrianglesNPY::sphere(n_polar, n_azimuthal);
    s->getTris()->dump("s");

    glm::vec4 param(0.,1.,0,0) ;
    NTrianglesNPY* hp = NTrianglesNPY::sphere(param, n_polar, n_azimuthal);
    hp->getTris()->dump("hp");
}


void test_icosahedron()
{
    LOG(info) << "test_icosahedron" ;
 
    NTrianglesNPY* icos = NTrianglesNPY::icosahedron();
    icos->getTris()->save("$TMP/icos.npy"); 
}


void test_box()
{
    nbbox bb = make_bbox( -100,-100,-100, 100, 100, 100 ); 
    NTrianglesNPY* tris = NTrianglesNPY::box(bb);
    tris->dump();
}

void test_to_from_vtxidx()
{
    nbbox bb = make_bbox( -100,-100,-100, 100, 100, 100 ); 
    NTrianglesNPY* tris = NTrianglesNPY::box(bb);
    tris->dump("tris");

    NVtxIdx vtxidx ;
    tris->to_vtxidx(vtxidx);

    vtxidx.vtx->dump();
    vtxidx.idx->dump();

    NTrianglesNPY* tris2 = NTrianglesNPY::from_indexed(vtxidx.vtx, vtxidx.idx);
    tris2->dump("tris2");

    float md = tris2->maxdiff(tris, true );

    LOG(info) << " maxdiff " << md ; 

    assert( md == 0.f );
}


int main(int argc, char**argv)
{
    OPTICKS_LOG(argc, argv);

/*
    test_prism();
    test_transform();
    test_latlon();
    test_icosahedron();
*/
    //test_box();
    test_to_from_vtxidx();
}


