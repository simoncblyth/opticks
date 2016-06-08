#include "NTrianglesNPY.hpp"
#include "BLog.hh"
#include "NPY.hpp"

void test_icosahedron_subdiv(unsigned int nsd)
{
    NTrianglesNPY* icos = NTrianglesNPY::icosahedron();

    NTrianglesNPY* tris = icos->subdivide(nsd);         
   
    unsigned int ntr = tris->getNumTriangles();

    LOG(info) << "test_icosahedron_subdiv" 
              << " nsd " << std::setw(4) << nsd
              << " ntr " << std::setw(6) << ntr 
              ; 
}


void test_icosahedron_subdiv()
{
    for(int i=0 ; i < 6 ; i++) test_icosahedron_subdiv(i) ;
}


void test_octahedron_subdiv(unsigned int nsd)
{
    NTrianglesNPY* oct = NTrianglesNPY::octahedron();

    NTrianglesNPY* tris = oct->subdivide(nsd);

    unsigned int ntr = tris->getNumTriangles();

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
    BLog nl("tess.log","info");
    nl.configure(argc, argv, "/tmp");

    test_icosahedron_subdiv();
    test_octahedron_subdiv();

    return 0 ; 
}



