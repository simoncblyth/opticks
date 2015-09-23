#include "Types.hpp"
#include "GCache.hh"
#include "GLoader.hh"
#include "GGeo.hh"


//#include "AssimpGGeo.hh"


int main(int argc, char* argv[])
{
    assert(0 && "CANNOT TEST GLoader from ggeo- due to assimpwrap- dependency, see ggeoview- tests/GLoaderTest.cc ");
 
    Types* m_types = new Types ;
    m_types->readFlags("$ENV_HOME/graphics/optixrap/cu/photon.h");

    GCache* m_cache = new GCache("GGEOVIEW_");

  /*
    GLoader* m_loader = new GLoader ;

    m_loader->setTypes(m_types);
    m_loader->setCache(m_cache);
    m_loader->setImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    bool nogeocache = false ; 
    m_loader->load(nogeocache);

    GGeo* m_ggeo = m_loader->getGGeo();
    m_ggeo->dumpTree();

  */

    return 0 ;
}




