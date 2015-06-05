#include "GLoader.hh"

// assimpwrap
//#include "AssimpWrap/AssimpGGeo.hh"

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



GLoader::GLoader() 
   :
   m_ggeo(NULL),
   m_mergedmesh(NULL),
   m_metadata(NULL)
{
}

const char* GLoader::identityPath( const char* envprefix)
{
    const char* geokey = getenvvar(envprefix, "GEOKEY" );
    const char* path = getenv(geokey);
    const char* query = getenvvar(envprefix, "QUERY");
    const char* ctrl = getenvvar(envprefix, "CTRL");


    if(query == NULL || path == NULL || geokey == NULL )
    {
        printf("GLoader::identityPath geokey %s path %s query %s ctrl %s \n", geokey, path, query, ctrl );
        LOG(fatal) << "GLoader::identityPath envprefix[" << envprefix << "] missing required envvars " ; 
        assert(0);
    }
 
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

    LOG(info)<< "GLoader::identityPath geokey " << geokey 
                   << " path " << path 
                   << " query " << query 
                   << " ctrl " << ctrl 
                   << " idpath " << idpath ; 

    return idpath ; 
}


const char* GLoader::load(const char* envprefix, bool nogeocache)
{
    const char* idpath = identityPath(envprefix);
    LOG(info) << "GLoader::load start idpath " << idpath << " nogeocache " << nogeocache  ;

    fs::path geocache(idpath);

    if(fs::exists(geocache) && fs::is_directory(geocache) && !nogeocache ) 
    {
        LOG(info) << "GLoader::load loading from cache directory " << idpath ;
        m_ggeo = NULL ; 
        m_mergedmesh = GMergedMesh::load(idpath);
        m_metadata   = GSubstanceLibMetadata::load(idpath);
    } 
    else
    {
        LOG(info) << "GLoader::load slow loading using m_imp (disguised AssimpGGeo) " << envprefix ;
        //m_ggeo = AssimpGGeo::load(envprefix);
        m_ggeo = (*m_imp)(envprefix);    

        //m_ggeo->Details("GLoader::load"); 

        m_mergedmesh = m_ggeo->getMergedMesh(); 
        m_mergedmesh->setColor(0.5,0.5,1.0);
        LOG(info) << "GLoader::load saving to cache directory " << idpath ;
        m_mergedmesh->save(idpath); 

        GSubstanceLib* lib = m_ggeo->getSubstanceLib();
        m_metadata = lib->getMetadata();
        m_metadata->save(idpath);

    } 

    LOG(info) << "GLoader::load done " << idpath ;
    assert(m_mergedmesh);
    return idpath ;
}

void GLoader::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_mergedmesh->Summary("GLoader::Summary");
    m_mergedmesh->Dump("GLoader::Summary Dump",10);
}



