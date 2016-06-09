#include "NTrianglesNPY.hpp"
#include "BLog.hh"
#include "NPY.hpp"
#include "GLMPrint.hpp"

void test_prism()
{
    glm::vec4 param(90,100,100,200);
    NTrianglesNPY* m = NTrianglesNPY::prism(param);
    m->getBuffer()->dump("prism");
    m->getBuffer()->save("/tmp/prism.npy");
/*
In [1]: fig = plt.figure()
In [2]: xyz3d(fig, "/tmp/prism.npy")
In [3]: plt.show()
*/
}

void test_transform()
{
    NTrianglesNPY* c = NTrianglesNPY::cube();
    c->getBuffer()->dump("cube");

    glm::vec3 tr(100.,0.,0);
    glm::vec3 sc(10.,10.,10.);

   // scale and then translate   (translation not scaled)
    glm::mat4 m = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);

    print(m, "m");

    NTrianglesNPY* tc = c->transform(m);
    tc->getBuffer()->dump("tcube");
}

void test_latlon()
{
     unsigned int n_polar = 24 ; 
     unsigned int n_azimuthal = 2 ; 

     NTrianglesNPY* s = NTrianglesNPY::sphere(n_polar, n_azimuthal);
     s->getBuffer()->dump("s");

     glm::vec4 param(0.,1.,0,0) ;
     NTrianglesNPY* hp = NTrianglesNPY::sphere(param, n_polar, n_azimuthal);
     hp->getBuffer()->dump("hp");
}


void test_icosahedron()
{
    NTrianglesNPY* icos = NTrianglesNPY::icosahedron();
    icos->getBuffer()->save("/tmp/icos.npy"); 
}



int main(int argc, char**argv)
{
    BLog nl(argc, argv);

    //test_icosahedron();
    //test_transform();
    //test_latlon();
    test_prism();

}



/*
    In [3]: np.linalg.norm(i.reshape(-1,3), 2, 1 )
    Out[3]: 
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)

*/

