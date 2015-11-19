#include "NTesselate.hpp"
#include "NLog.hpp"
#include "NSphere.hpp"
#include "NPY.hpp"

void test_icosahedron_subdiv(unsigned int nsd)
{
    NPY<float>* icos_0 = NSphere::icosahedron(0);
    NPY<float>* icos = NSphere::icosahedron(nsd);

    NTesselate* tess = new NTesselate(icos_0);
    tess->subdivide(nsd);         
    NPY<float>* tris = tess->getBuffer();
   
    unsigned int ntr_i = icos->getNumItems();
    unsigned int ntr = tris->getNumItems();
    assert(ntr_i == ntr);

    float mxd = tris->maxdiff(icos);

    LOG(info) << "test_icosahedron_subdiv" 
              << " nsd " << std::setw(4) << nsd
              << " ntr " << std::setw(6) << ntr 
              << " mxdiff " << mxd  
              ; 
}


void test_icosahedron_subdiv()
{
    for(int i=0 ; i < 6 ; i++) test_icosahedron_subdiv(i) ;
}


void test_octahedron_subdiv(unsigned int nsd)
{
    NPY<float>* oct = NSphere::octahedron(nsd);
    unsigned int ntr = oct->getNumItems();
    LOG(info) << "test_octahedron_subdiv" 
              << " nsd " << std::setw(4) << nsd
              << " ntr " << std::setw(6) << ntr 
              ;
}

void test_octahedron_subdiv()
{
    for(int i=0 ; i < 6 ; i++) test_octahedron_subdiv(i) ;
}




int main(int argc, char** argv)
{
    NLog nl("tess.log","info");
    nl.configure(argc, argv, "/tmp");

    test_icosahedron_subdiv();
    test_octahedron_subdiv();

    return 0 ; 
}



