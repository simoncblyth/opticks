/*

    ggv --loader 
    ggv --loader --idyb
    ggv --dbg --loader --idyb
    ggv --loader --jpmt

*/

#include "Types.hpp"
#include "GCache.hh"
#include "GLoader.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "AssimpGGeo.hh"

int main(int argc, char* argv[])
{
    Types* m_types = new Types ;
    m_types->readFlags("$ENV_HOME/graphics/optixrap/cu/photon.h");

    GCache* m_cache = new GCache("GGEOVIEW_");

    GGeo* m_ggeo = new GGeo(m_cache);

    GLoader* m_loader = new GLoader(m_ggeo) ;

    m_loader->setTypes(m_types);
    m_loader->setCache(m_cache);
    m_loader->setLoaderImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    m_loader->load();

    m_ggeo->dumpTree();

    m_ggeo->dumpVolume(3158);    
    m_ggeo->dumpVolume(3159);    


    GMergedMesh* m_mesh0 = m_ggeo->getMergedMesh(0);
    m_mesh0->explodeZVertices(1000.f, -(5564.950f + 5565.000f)/2.f );


    return 0 ;
}


/*


*/

