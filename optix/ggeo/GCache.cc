#include "GCache.hh"
#include "GFlags.hh"
#include "GColors.hh"

#include <sstream>
#include "assert.h"
#include "stdio.h"

// npy-
#include "NLog.hpp"
#include "stringutil.hpp"
#include "GLMFormat.hpp"
#include "Types.hpp"
#include "Typ.hpp"
#include "Map.hpp"


#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


const char* GCache::COMPUTE = "--compute" ; 
const char* GCache::JUNO    = "juno" ; 
const char* GCache::DAYABAY = "dayabay" ; 
const char* GCache::PREFERENCE_BASE = "$HOME/.opticks" ; 

GCache* GCache::g_instance = NULL ; 


void GCache::init()
{
    m_log = new NLog(m_logname, m_loglevel);
    readEnvironment();
    m_juno     = idPathContains("env/geant4/geometry/export/juno") ;
    m_dayabay  = idPathContains("env/geant4/geometry/export/DayaBay") ;
    assert( m_juno ^ m_dayabay ); // exclusive-or

    if(m_juno)    m_detector = JUNO ; 
    if(m_dayabay) m_detector = DAYABAY ; 
}


void GCache::configure(int argc, char** argv)
{
    for(unsigned int i=1 ; i < argc ; i++ )
    {
        //printf("GCache::configure  %2d : %s \n", i, argv[i] );
        if(strcmp(argv[i], COMPUTE) == 0) 
        {
            //printf("GCache::configure setting compute \n");
            m_compute = true ; 
        }
    }

    m_log->configure(argc, argv);
    m_log->init(m_idpath);

    m_lastarg = argc > 1 ? strdup(argv[argc-1]) : NULL ;
}


const char* GCache::getLastArg()
{
   return m_lastarg ; 
}
int GCache::getLastArgInt()
{
    int index(-1);
    if(!m_lastarg) return index ;
 
    try{ 
        index = boost::lexical_cast<int>(m_lastarg) ;
    }
    catch (const boost::bad_lexical_cast& e ) {
        LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
    }
    catch( ... ){
        LOG(warning) << "Unknown exception caught!" ;
    }
    return index;
}

// lazy constituent construction : as want to avoid any output until after logging is configured

GColors* GCache::getColors()
{
    if(m_colors == NULL)
    {
        std::string prefdir = getPreferenceDir("GCache");
        m_colors = GColors::load(prefdir.c_str(),"GColors.json");  // colorname => hexcode 
    }
    return m_colors ;
}

Typ* GCache::getTyp()
{
    if(m_typ == NULL)
    {
       m_typ = new Typ ; 
    }
    return m_typ ; 
}

Types* GCache::getTypes()
{
    if(m_types == NULL)
    {
        m_types = new Types ;  
        m_types->saveFlags(m_idpath, ".ini");
    }
    return m_types ;
}


GFlags* GCache::getFlags()
{
    if(m_flags == NULL)
    {
        m_flags = new GFlags(this);  // parses the flags enum source, from $ENV_HOME/opticks/OpticksPhoton.h
        m_flags->save(m_idpath);

    }
    return m_flags ;
}





std::string GCache::getPreferenceDir(const char* type)
{
    std::stringstream ss ; 
    ss << PREFERENCE_BASE << "/" << type ; 
    return ss.str();
}


bool GCache::loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    typedef Map<std::string, std::string> MSS ;  
    MSS* pref = MSS::load(prefdir.c_str(), name ) ; 
    if(pref)
        mss = pref->getMap(); 
    return pref != NULL ; 
}


bool GCache::loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    typedef Map<std::string, unsigned int> MSU ;  
    MSU* pref = MSU::load(prefdir.c_str(), name ) ; 
    if(pref)
        msu = pref->getMap(); 
    return pref != NULL ; 
}








void GCache::readEnvironment()
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
        printf("GCache::readEnvironment geokey %s path %s query %s ctrl %s \n", m_geokey, m_path, m_query, m_ctrl );
        printf("GCache::readEnvironment  m_envprefix[%s] missing required envvars ", m_envprefix );
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



glm::vec4 GCache::getMeshfixFacePairingCriteria()
{
    assert(m_meshfixcfg) ; 
    std::string meshfixcfg = m_meshfixcfg ;
    return gvec4(meshfixcfg);
}




std::string GCache::getMergedMeshPath(unsigned int index)
{
    return getObjectPath("GMergedMesh", index);
}

std::string GCache::getPmtPath(unsigned int index, bool relative)
{
    return getObjectPath("GPmt", index, relative);
}

std::string GCache::getObjectPath(const char* name, unsigned int index, bool relative)
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





std::string GCache::getPropertyLibDir(const char* name)
{
    fs::path cachedir(m_idpath);
    fs::path pld(cachedir/name );
    return pld.string() ;
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
    printf("meshfix  : %s \n", m_meshfix ); 
}


