#include "Geometry.hh"

// assimpwrap
#include "AssimpWrap/AssimpGGeo.hh"
#include "GMergedMesh.hh"
#include "GSubstanceLib.hh"
#include "GSubstanceLibMetadata.hh"
#include "GGeo.hh"

// npy-
#include "stringutil.hpp"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



Geometry::Geometry() 
   :
   m_ggeo(NULL),
   m_mergedmesh(NULL),
   m_metadata(NULL)
{
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


const char* Geometry::load(const char* envprefix, bool nogeocache)
{
    const char* idpath = identityPath(envprefix);
    LOG(info) << "Geometry::load start idpath " << idpath << " nogeocache " << nogeocache  ;

    fs::path geocache(idpath);

    if(fs::exists(geocache) && fs::is_directory(geocache) && !nogeocache ) 
    {
        LOG(info) << "Geometry::load loading from cache directory " << idpath ;
        m_ggeo = NULL ; 
        m_mergedmesh = GMergedMesh::load(idpath);
        m_metadata   = GSubstanceLibMetadata::load(idpath);
    } 
    else
    {
        LOG(info) << "Geometry::load slow loading using AssimpGGeo " << envprefix ;
        m_ggeo = AssimpGGeo::load(envprefix);

        m_ggeo->Details("Geometry::load"); 

        m_mergedmesh = m_ggeo->getMergedMesh(); 
        m_mergedmesh->setColor(0.5,0.5,1.0);
        LOG(info) << "Geometry::load saving to cache directory " << idpath ;
        m_mergedmesh->save(idpath); 

        GSubstanceLib* lib = m_ggeo->getSubstanceLib();
        m_metadata = lib->getMetadata();
        m_metadata->save(idpath);

    } 

    LOG(info) << "Geometry::load done " << idpath ;
    assert(m_mergedmesh);
    return idpath ;
}

void Geometry::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_mergedmesh->Summary("Geometry::Summary");
    m_mergedmesh->Dump("Geometry::Summary Dump",10);
}



