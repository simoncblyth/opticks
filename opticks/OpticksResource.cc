#include "OpticksResource.hh"
#include <cassert>

// npy-
#include "stringutil.hpp"
#include "GLMFormat.hpp"
#include "NLog.hpp"
#include "Map.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


const char* OpticksResource::JUNO    = "juno" ; 
const char* OpticksResource::DAYABAY = "dayabay" ; 
const char* OpticksResource::PREFERENCE_BASE = "$HOME/.opticks" ; 

void OpticksResource::init()
{
   readEnvironment();
   m_juno     = idPathContains("env/geant4/geometry/export/juno") ;
   m_dayabay  = idPathContains("env/geant4/geometry/export/DayaBay") ;
   assert( m_juno ^ m_dayabay ); // exclusive-or

   if(m_juno)    m_detector = JUNO ; 
   if(m_dayabay) m_detector = DAYABAY ; 
}


void OpticksResource::readEnvironment()
{
/*

:param envprefix: of the required envvars, eg with "GGEOVIEW_" need:

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
        printf("OpticksResource::readEnvironment geokey %s path %s query %s ctrl %s \n", m_geokey, m_path, m_query, m_ctrl );
        printf("OpticksResource::readEnvironment  m_envprefix[%s] missing required envvars ", m_envprefix );
        assert(0);
    }

    m_meshfix = getenvvar(m_envprefix, "MESHFIX");
    m_meshfixcfg = getenvvar(m_envprefix, "MESHFIX_CFG");
 
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

    m_idfold = strdup(m_idpath);

    char* p = (char*)strrchr(m_idfold, '/');  // point to last slash 
    *p = '\0' ;                               // chop to give parent fold
 

    //Summary("GCache::readEnvironment");

    int overwrite = 1; 
    assert(setenv("IDPATH", m_idpath, overwrite)==0);

    // DO NOT PRINT ANYTHING FROM HERE TO AVOID IDP CAPTURE PROBLEMS
}


void OpticksResource::Summary(const char* msg)
{
    printf("%s \n", msg );
    printf("envprefix: %s \n", m_envprefix ); 
    printf("geokey   : %s \n", m_geokey ); 
    printf("path     : %s \n", m_path ); 
    printf("query    : %s \n", m_query ); 
    printf("ctrl     : %s \n", m_ctrl ); 
    printf("digest   : %s \n", m_digest ); 
    printf("idpath   : %s \n", m_idpath ); 
    printf("meshfix  : %s \n", m_meshfix ); 
}



glm::vec4 OpticksResource::getMeshfixFacePairingCriteria()
{
   //
   // 4 comma delimited floats specifying criteria for faces to be deleted from the mesh
   //
   //   xyz : face barycenter alignment 
   //     w : dot face normal cuts 
   //

    assert(m_meshfixcfg) ; 
    std::string meshfixcfg = m_meshfixcfg ;
    return gvec4(meshfixcfg);
}

std::string OpticksResource::getMergedMeshPath(unsigned int index)
{
    return getObjectPath("GMergedMesh", index);
}

std::string OpticksResource::getPmtPath(unsigned int index, bool relative)
{
    return getObjectPath("GPmt", index, relative);
}

std::string OpticksResource::getObjectPath(const char* name, unsigned int index, bool relative)
{
    fs::path dir ; 
    if(!relative)
    {
        fs::path cachedir(m_idpath);
        dir = cachedir/name/boost::lexical_cast<std::string>(index) ;
    }
    else
    {
        fs::path reldir(name);
        dir = reldir/boost::lexical_cast<std::string>(index) ;
    }
    return dir.string() ;
}

std::string OpticksResource::getPropertyLibDir(const char* name)
{
    fs::path cachedir(m_idpath);
    fs::path pld(cachedir/name );
    return pld.string() ;
}

std::string OpticksResource::getPreferenceDir(const char* type, const char* udet )
{
    std::stringstream ss ; 
    ss << PREFERENCE_BASE ; 
    if(udet) ss << "/" << udet ; 
    ss << "/" << type ; 
    return ss.str();
}

bool OpticksResource::loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    typedef Map<std::string, std::string> MSS ;  
    MSS* pref = MSS::load(prefdir.c_str(), name ) ; 
    if(pref)
        mss = pref->getMap(); 
    return pref != NULL ; 
}

bool OpticksResource::loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    typedef Map<std::string, unsigned int> MSU ;  
    MSU* pref = MSU::load(prefdir.c_str(), name ) ; 
    if(pref)
        msu = pref->getMap(); 
    return pref != NULL ; 
}






