#include "Geometry.hh"

// assimpwrap
#include "AssimpWrap/AssimpGGeo.hh"
#include "GMergedMesh.hh"
#include "GGeo.hh"

// npy-
#include "stringutil.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



Geometry::Geometry() 
   :
   m_ggeo(NULL),
   m_mergedmesh(NULL)
{
}

GGeo* Geometry::getGGeo()
{
    return m_ggeo ; 
}

GMergedMesh* Geometry::getMergedMesh()
{
    return m_mergedmesh ; 
}
GDrawable* Geometry::getDrawable()
{
    return m_mergedmesh ; 
}

const char* Geometry::identityPath( const char* envprefix)
{
    const char* geokey = getenvvar(envprefix, "GEOKEY" );
    const char* path = getenv(geokey);
    const char* query = getenvvar(envprefix, "QUERY");
    const char* ctrl = getenvvar(envprefix, "CTRL");
 
    //  
    // #. real path is converted into "fake" path 
    //    incorporating the digest of geometry selecting envvar 
    //
    //    Kludge done to reuse OptiX sample code accelcache, allowing
    //    to benefit from caching as vary the envvar while 
    //    still only having a single geometry file.
    //

    std::string digest = md5digest( query, strlen(query));
    std::string kfn = insertField( path, '.', -1 , digest.c_str());
    const char* idpath = strdup(kfn.c_str());

    LOG(info)<< "Geometry::identityPath geokey " << geokey 
                   << " path " << path 
                   << " query " << query 
                   << " ctrl " << ctrl 
                   << " idpath " << idpath ; 

    return idpath ; 
}


void Geometry::load(const char* envprefix)
{
    const char* idpath = identityPath(envprefix);
    LOG(info) << "Geometry::load start idpath " << idpath  ;

    m_ggeo = AssimpGGeo::load(envprefix);
    m_mergedmesh = m_ggeo->getMergedMesh(); 
    assert(m_mergedmesh);
    m_mergedmesh->setColor(0.5,0.5,1.0);
    LOG(info) << "Geometry::load done " ;
}

void Geometry::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_mergedmesh->Summary("Geometry::Summary");
    m_mergedmesh->Dump("Geometry::Summary Dump",10);
}



