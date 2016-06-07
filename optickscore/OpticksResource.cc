#include "OpticksResource.hh"
#include "OpticksQuery.hh"
#include "OpticksColors.hh"
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"

#include <cassert>

// npy-
#include "stringutil.hpp"
#include "GLMFormat.hpp"
#include "NLog.hpp"
#include "Map.hpp"

#include "Typ.hpp"
#include "Types.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


const char* OpticksResource::JUNO    = "juno" ; 
const char* OpticksResource::DAYABAY = "dayabay" ; 
const char* OpticksResource::DPIB    = "PmtInBox" ; 
const char* OpticksResource::OTHER   = "other" ; 

const char* OpticksResource::PREFERENCE_BASE = "$HOME/.opticks" ; 

const char* OpticksResource::DEFAULT_GEOKEY = "DAE_NAME_DYB" ; 
const char* OpticksResource::DEFAULT_QUERY = "range:3153:12221" ; 
const char* OpticksResource::DEFAULT_CTRL = "volnames" ; 

void OpticksResource::init()
{
   std::cerr << "OpticksResource::init"<< std::endl ; 

   readEnvironment();
   readMetadata();
   identifyGeometry();

   std::cerr << "OpticksResource::init DONE"<< std::endl ; 
}

void OpticksResource::identifyGeometry()
{
   // TODO: somehow extract detector name from the exported file metadata or sidecar

   m_juno     = idPathContains("env/geant4/geometry/export/juno") ;
   m_dayabay  = idPathContains("env/geant4/geometry/export/DayaBay") ;
   m_dpib     = idPathContains("env/geant4/geometry/export/dpib") ;

   if(m_juno == false && m_dayabay == false && m_dpib == false )
   {
       const char* detector = getMetaValue("detector") ;
       if(detector)
       {
           if(     strcmp(detector, DAYABAY) == 0) m_dayabay = true ; 
           else if(strcmp(detector, JUNO)    == 0) m_juno = true ; 
           else if(strcmp(detector, DPIB)    == 0) m_dpib = true ; 
           else 
                 m_other = true ;

           printf("OpticksResource::identifyGeometry from metadata : %s \n", detector ); 
       }
       else
           m_other = true ;
   }


   assert( m_juno ^ m_dayabay ^ m_dpib ^ m_other ); // exclusive-or
   
   if(m_juno)    m_detector = JUNO ; 
   if(m_dayabay) m_detector = DAYABAY ; 
   if(m_dpib)    m_detector = DPIB ; 
   if(m_other)   m_detector = OTHER ; 
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

    if(m_geokey == NULL)
    {
        m_geokey = DEFAULT_GEOKEY ;  
        printf("OpticksResource::readEnvironment USING DEFAULT geokey %s \n", m_geokey );
    }

    m_daepath = getenv(m_geokey);

    if(m_daepath == NULL)
    {
        if(m_lastarg && existsFile(m_lastarg))
        {
            m_daepath = m_lastarg ; 
            printf("OpticksResource::readEnvironment MISSING ENVVAR pointing to geometry for geokey %s but lastarg is a path %s \n", m_geokey, m_daepath );
        }
    }

    if(m_daepath == NULL)
    {
        printf("OpticksResource::readEnvironment MISSING ENVVAR pointing to geometry for geokey %s path %s \n", m_geokey, m_daepath );
        //assert(0);
        setValid(false);
    } 
    else
    {
         std::string metapath = makeSidecarPath(m_daepath, ".dae", ".ini");
         m_metapath = strdup(metapath.c_str());
         std::string gdmlpath = makeSidecarPath(m_daepath, ".dae", ".gdml");
         m_gdmlpath = strdup(gdmlpath.c_str());
    }


    m_query_string = getenvvar(m_envprefix, "QUERY");

    if(m_query_string == NULL)
    {
        m_query_string = DEFAULT_QUERY ;  
        printf("OpticksResource::readEnvironment USING DEFAULT geo query %s \n", m_query_string );
    }
    m_query = new OpticksQuery(m_query_string);


    m_ctrl = getenvvar(m_envprefix, "CTRL");
    if(m_ctrl == NULL)
    {
        m_ctrl = DEFAULT_CTRL ;  
        printf("OpticksResource::readEnvironment USING DEFAULT geo ctrl %s \n", m_ctrl );
    }

    m_meshfix = getenvvar(m_envprefix, "MESHFIX");
    m_meshfixcfg = getenvvar(m_envprefix, "MESHFIX_CFG");


    std::string query_digest = md5digest( m_query_string, strlen(m_query_string));
    m_digest = strdup(query_digest.c_str());
 
    // idpath incorporates digest of geometry selection envvar 
    // allowing to benefit from caching as vary geometry selection 
    // while still only having a single source geometry file.

    if(m_daepath)
    {
        std::string kfn = insertField( m_daepath, '.', -1 , m_digest );

        m_idpath = strdup(kfn.c_str());
    
        //int overwrite = 1; 
        //assert(setenv("IDPATH", m_idpath, overwrite)==0);  // TODO: avoid this, windoze lacks setenv 

        assert(setenvvar("","IDPATH", m_idpath )==0);  // uses putenv for windows mingw compat 
        


    }
    else
    {
         // IDPATH envvar is last resort, but handy for testing
         m_idpath = getenv("IDPATH");
    }

    if(m_idpath)
    {
        m_idfold = strdup(m_idpath);
        char* p = (char*)strrchr(m_idfold, '/');  // point to last slash 
        *p = '\0' ;                               // chop to give parent fold
        // TODO: avoid this dirty way, do with boost fs
    } 

    // DO NOT PRINT ANYTHING FROM HERE TO AVOID IDP CAPTURE PROBLEMS
}


void OpticksResource::readMetadata()
{
    if(m_metapath)
    {
         loadMetadata(m_metadata, m_metapath);
         //dumpMetadata(m_metadata);
    }
}

void OpticksResource::Dump(const char* msg)
{
    Summary(msg);

    std::string mmsp = getMergedMeshPath(0);
    std::string pmtp = getPmtPath(0);

    std::cerr << "mmsp(0) :" << mmsp << std::endl ;  
    std::cerr << "pmtp(0) :" << pmtp << std::endl ;  
}

void OpticksResource::Summary(const char* msg)
{
    std::cerr << msg << std::endl ; 
    std::cerr << "valid    :" <<   (m_valid ? "valid" : "NOT VALID" ) << std::endl ; 
    std::cerr << "envprefix: " <<  (m_envprefix?m_envprefix:"NULL") << std::endl; 
    std::cerr << "geokey   : " <<  (m_geokey?m_geokey:"NULL") << std::endl; 
    std::cerr << "daepath  : " <<  (m_daepath?m_daepath:"NULL") << std::endl; 
    std::cerr << "gdmlpath : " <<  (m_gdmlpath?m_gdmlpath:"NULL") << std::endl; 
    std::cerr << "metapath : " <<  (m_metapath?m_metapath:"NULL") << std::endl; 
    std::cerr << "query    : " <<  (m_query_string?m_query_string:"NULL") << std::endl; 
    std::cerr << "ctrl     : " <<  (m_ctrl?m_ctrl:"NULL") << std::endl; 
    std::cerr << "digest   : " <<  (m_digest?m_digest:"NULL") << std::endl; 
    std::cerr << "idpath   : " <<  (m_idpath?m_idpath:"NULL") << std::endl; 
    std::cerr << "meshfix  : " <<  (m_meshfix ? m_meshfix : "NULL" ) << std::endl; 

    typedef std::map<std::string, std::string> SS ;
    for(SS::const_iterator it=m_metadata.begin() ; it != m_metadata.end() ; it++)
        std::cerr <<  std::setw(10) << it->first.c_str() << ":" <<  it->second.c_str() << std::endl  ;

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
        assert(m_idpath && "OpticksResource::getObjectPath idpath not set");
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
    assert(m_idpath && "OpticksResource::getPropertyLibDir idpath not set");
    fs::path cachedir(m_idpath);
    fs::path pld(cachedir/name );
    return pld.string() ;
}

std::string OpticksResource::getPreferenceDir(const char* type, const char* udet, const char* subtype )
{
    fs::path prefdir(PREFERENCE_BASE) ;
    if(udet) prefdir /= udet ;
    prefdir /= type ; 
    if(subtype) prefdir /= subtype ; 
    return prefdir.string() ;
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

bool OpticksResource::existsFile(const char* path)
{
    fs::path fpath(path);
    return fs::exists(fpath ) && fs::is_regular_file(fpath) ;
}

bool OpticksResource::existsFile(const char* dir, const char* name)
{
    fs::path fpath(dir);
    fpath /= name ; 
    return fs::exists(fpath ) && fs::is_regular_file(fpath) ;
}

bool OpticksResource::existsDir(const char* path)
{
    fs::path fpath(path);
    return fs::exists(fpath ) && fs::is_directory(fpath) ;
}




std::string OpticksResource::makeSidecarPath(const char* path, const char* styp, const char* dtyp)
{
   std::string empty ; 

   fs::path src(path);
   fs::path ext = src.extension();
   bool is_styp = ext.string().compare(styp) == 0  ;

   assert(is_styp && "OpticksResource::makeSidecarPath source file type doesnt match the path file type");

   fs::path dst(path);
   dst.replace_extension(dtyp) ;

   /*
   LOG(debug) << "OpticksResource::makeSidecarPath"
             << " styp " << styp
             << " dtyp " << dtyp
             << " ext "  << ext 
             << " src " << src.string()
             << " dst " << dst.string()
             << " is_styp "  << is_styp 
             ;
   */

   return dst.string() ;
}

bool OpticksResource::loadMetadata(std::map<std::string, std::string>& mdd, const char* path)
{
    typedef Map<std::string, std::string> MSS ;  
    MSS* meta = MSS::load(path) ; 
    if(meta)
        mdd = meta->getMap(); 
    return meta != NULL ; 
}

void OpticksResource::dumpMetadata(std::map<std::string, std::string>& mdd)
{
    typedef std::map<std::string, std::string> SS ;
    for(SS::const_iterator it=mdd.begin() ; it != mdd.end() ; it++)
    {
       std::cout
             << std::setw(20) << it->first 
             << std::setw(20) << it->second
             << std::endl ; 
    }
}


bool OpticksResource::hasMetaKey(const char* key)
{
    return m_metadata.count(key) == 1 ; 
}
const char* OpticksResource::getMetaValue(const char* key)
{
    return m_metadata.count(key) == 1 ? m_metadata[key].c_str() : NULL ;
}


OpticksColors* OpticksResource::getColors()
{
    if(!m_colors)
    {
        // deferred to avoid output prior to logging setup
        std::string prefdir = getPreferenceDir("GCache");
        m_colors = OpticksColors::load(prefdir.c_str(),"GColors.json");   // colorname => hexcode
    }
    return m_colors ;
}

OpticksFlags* OpticksResource::getFlags()
{
    if(!m_flags)
    {
        m_flags = new OpticksFlags(m_opticks); 
        m_flags->save(getIdPath());
    }
    return m_flags ;
}


OpticksAttrSeq* OpticksResource::getFlagNames()
{
    OpticksFlags* flags = getFlags();
    OpticksAttrSeq* qflg = flags->getAttrIndex();
    qflg->setCtrl(OpticksAttrSeq::SEQUENCE_DEFAULTS);    
    return qflg ; 
}


Typ* OpticksResource::getTyp()
{
    if(m_typ == NULL)
    {   
       m_typ = new Typ ; 
    }   
    return m_typ ; 
}


Types* OpticksResource::getTypes()
{
    if(!m_types)
    {   
        // deferred because idpath not known at init ?
        m_types = new Types ;   
        m_types->saveFlags(getIdPath(), ".ini");
    }   
    return m_types ;
}


