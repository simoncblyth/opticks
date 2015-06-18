#include "GLoader.hh"

// look no hands : no dependency on AssimpWrap 

#include "GMergedMesh.hh"
#include "GBoundaryLib.hh"
#include "GBoundaryLibMetadata.hh"
#include "GGeo.hh"

// npy-
#include "stringutil.hpp"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* GLoader::identityPath( const char* envprefix)
{
/*

:param envprefix: of the required envvars, eg with "GGEOVIEW_" need:

   GGEOVIEW_GEOKEY
   GGEOVIEW_QUERY
   GGEOVIEW_CTRL

Example values::

    simon:~ blyth$ ggeoview-
    simon:~ blyth$ ggeoview-export
    simon:~ blyth$ env | grep GGEOVIEW_
    GGEOVIEW_CTRL=
    GGEOVIEW_QUERY=range:3153:12221
    GGEOVIEW_GEOKEY=DAE_NAME_DYB
    simon:~ blyth$ echo $DAE_NAME_DYB
    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae


GGeoview application reports the idpath via::

    simon:~ blyth$ idpath=$(ggv --idpath)
    [2015-Jun-18 11:35:52.881375]: GLoader::identityPath 
     envprefix GGEOVIEW_
     geokey    DAE_NAME_DYB
     path      /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae
     query     range:3153:12221
     ctrl      
     idpath    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae

    simon:~ blyth$ echo $idpath
    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae


*path* 
     identifies the source geometry G4DAE exported file

*query*
     string used to select volumes from the geometry 

*idpath* 
     directory name based on *path* incorporating a hexdigest of the *query* 
     this directory is used to house:

     * geocache of NPY persisted buffers of the geometry
     * json metadata files
     * bookmarks

*ctrl*
     not currently used?


*/

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

    LOG(info)<< "GLoader::identityPath "
             << "\n envprefix " << envprefix 
             << "\n geokey    " << geokey 
             << "\n path      " << path 
             << "\n query     " << query 
             << "\n ctrl      " << ctrl 
             << "\n idpath    " << idpath ; 

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
        m_metadata   = GBoundaryLibMetadata::load(idpath);
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

        GBoundaryLib* lib = m_ggeo->getBoundaryLib();
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



