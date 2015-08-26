#include "GCache.hh"

#include "assert.h"
#include "stdio.h"

// npy-
#include "stringutil.hpp"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


const char* GCache::JUNO    = "juno" ; 
const char* GCache::DAYABAY = "dayabay" ; 

GCache* GCache::g_instance = NULL ; 


void GCache::readEnvironment()
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

    m_geokey = getenvvar(m_envprefix, "GEOKEY" );
    m_path = getenv(m_geokey);
    m_query = getenvvar(m_envprefix, "QUERY");
    m_ctrl = getenvvar(m_envprefix, "CTRL");

    if(m_query == NULL || m_path == NULL || m_geokey == NULL )
    {
        printf("GCache::readEnvironment geokey %s path %s query %s ctrl %s \n", m_geokey, m_path, m_query, m_ctrl );
        printf("GCache::readEnvironment  m_envprefix[%s] missing required envvars ", m_envprefix );
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

    std::string digest = md5digest( m_query, strlen(m_query));
    m_digest = strdup(digest.c_str());
    std::string kfn = insertField( m_path, '.', -1 , m_digest );

    m_idpath = strdup(kfn.c_str());

    int overwrite = 1; 
    assert(setenv("IDPATH", m_idpath, overwrite)==0);

    printf("GCache::readEnvironment setting IDPATH internally to %s \n", getenv("IDPATH")) ; 
}


void GCache::Summary(const char* msg)
{
    printf("%s \n", msg );
    printf("envprefix: %s \n", m_envprefix ); 
    printf("geokey   : %s \n", m_geokey ); 
    printf("path     : %s \n", m_path ); 
    printf("query    : %s \n", m_query ); 
    printf("ctrl     : %s \n", m_ctrl ); 
    printf("digest   : %s \n", m_digest ); 
    printf("idpath   : %s \n", m_idpath ); 
}


void GCache::init()
{
    readEnvironment();
    m_juno     = idPathContains("env/geant4/geometry/export/juno") ;
    m_dayabay  = idPathContains("env/geant4/geometry/export/DayaBay") ;
    assert( m_juno ^ m_dayabay ); // exclusive-or

    if(m_juno)    m_detector = JUNO ; 
    if(m_dayabay) m_detector = DAYABAY ; 
}


