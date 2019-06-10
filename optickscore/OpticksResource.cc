
#include <cstring>
#include <cassert>
#include <algorithm>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


// sysrap
#include "SLog.hh"
#include "SDigest.hh"
#include "SSys.hh"

// brap-
#include "BFile.hh"
#include "BStr.hh"
#include "PLOG.hh"
#include "Map.hh"
#include "BResource.hh"
#include "BOpticksKey.hh"
#include "BEnv.hh"

// npy-
#include "NSensorList.hpp"
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "Typ.hpp"
#include "Types.hpp"
#include "Index.hpp"


#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksQuery.hh"
#include "OpticksColors.hh"
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"


const plog::Severity OpticksResource::LEVEL = debug ; 

/*
TODO:
   migrate everything that does not need the Opticks instance (ie the commandline arguments) 
   down into the base class  BOpticksResource


   Ideally want to slim this class... to almost nothing 
    


*/

OpticksResource::OpticksResource(Opticks* ok) 
    :
    BOpticksResource(ok->isTest()),
    m_log(new SLog("OpticksResource::OpticksResource","",debug)),
    m_ok(ok),
    m_query(new OpticksQuery(SSys::getenvvar(
                      m_key ? "OPTICKS_QUERY_LIVE" : "OPTICKS_QUERY" ,
                      m_key ? DEFAULT_QUERY_LIVE : DEFAULT_QUERY 
              ))), 
    m_geokey(NULL),
    m_ctrl(NULL),
    m_meshfix(NULL),
    m_meshfixcfg(NULL),
    m_valid(true),
    m_colors(NULL),
    m_flags(new OpticksFlags),
    m_flagnames(NULL),
    m_types(NULL),
    m_typ(NULL),
    m_g4env(NULL),
    m_okenv(NULL),
    m_dayabay(false),
    m_juno(false),
    m_dpib(false),
    m_other(false),
    m_detector(NULL),
    m_detector_name(NULL),
    m_detector_base(NULL),
    m_resource_base(NULL),
    m_material_map(NULL),
    m_default_material(NULL),
    m_default_medium(NULL),
    m_example_matnames(NULL),
    m_sensor_surface(NULL),
    m_default_frame(DEFAULT_FRAME_OTHER),
    m_sensor_list(NULL),
    m_runresultsdir(NULL)
{
    init();
    (*m_log)("DONE"); 
}

 

void OpticksResource::setValid(bool valid)
{
    m_valid = valid ; 
}
bool OpticksResource::isValid()
{
   return m_valid ; 
}
const char* OpticksResource::getDetectorBase()
{
    return m_detector_base ;
}
const char* OpticksResource::getMaterialMap()
{
    return m_material_map ;
}
OpticksQuery* OpticksResource::getQuery()
{
    return m_query ;
}
const char* OpticksResource::getCtrl()
{
    return m_ctrl ;
}

bool OpticksResource::hasCtrlKey(const char* key) const 
{
    return BStr::listHasKey(m_ctrl, key, ",");
}


const char* OpticksResource::getMeshfix()
{
    return m_meshfix ;
}
const char* OpticksResource::getMeshfixCfg()
{
    return m_meshfixcfg ;
}


const char* OpticksResource::getDetector()
{
    return m_detector ;
}
const char* OpticksResource::getDetectorName()
{
    return m_detector_name ;
}



bool OpticksResource::isG4Live()
{
   return m_g4live ; 
}
bool OpticksResource::isJuno()
{
   return m_juno ; 
}
bool OpticksResource::isDayabay()
{
   return m_dayabay ; 
}
bool OpticksResource::isPmtInBox()
{
   return m_dpib ; 
}
bool OpticksResource::isOther()
{
   return m_other ; 
}



bool OpticksResource::idNameContains(const char* s)
{
    bool ret = false ; 
    if(m_idname)
    {
        std::string idn(m_idname);
        std::string ss(s);
        ret = idn.find(ss) != std::string::npos ;
    }
    else
    {
        LOG(error) << " idname NULL " ; 
    }

    return ret ; 
}




std::string OpticksResource::getRelativePath(const char* path)
{
    const char* idpath = getIdPath();

    if(strncmp(idpath, path, strlen(idpath)) == 0)
    {
        return path + strlen(idpath) + 1 ; 
    }
    else
    {
        return path ;  
    }
}


/**
OpticksResource::init
-----------------------

For booting via key to come into effect (necessary when trying to run without opticksdata) 
still need to provide "--envkey" argument to all Opticks executables::

    OPTICKS_INSTALL_PREFIX=/tmp/tt OpticksResourceTest --envkey


**/

void OpticksResource::init()
{
   LOG(LEVEL) << "OpticksResource::init" ; 

   BStr::split(m_detector_types, "GScintillatorLib,GMaterialLib,GSurfaceLib,GBndLib,GSourceLib", ',' ); 
   BStr::split(m_resource_types, "GFlags,OpticksColors", ',' ); 

   readG4Environment();
   readOpticksEnvironment();

   if( hasKey() )  
   {
       setupViaKey();    // from BOpticksResource base
   } 
   else
   {
       readEnvironment();  // invokes BOpticksResource::setupViaSrc after getting daepath from envvar
   }

   readMetadata();
   identifyGeometry();
   assignDetectorName(); 
   assignDefaultMaterial(); 

   initRunResultsDir(); 

   LOG(LEVEL) << "OpticksResource::init DONE" ; 
}

void OpticksResource::initRunResultsDir()
{
    const char* runfolder = m_ok->getRunFolder();  // eg geocache-bench 
    const char* runlabel = m_ok->getRunLabel();    // eg OFF_TITAN_V_AND_TITAN_RTX
    const char* rundate = m_ok->getRunDate() ;     // eg 20190422_162401 

    std::string runresultsdir = getResultsPath( runfolder, runlabel, rundate ) ;  // eg /usr/local/opticks/results/geocache-bench/OFF_TITAN_V_AND_TITAN_RTX/20190422_162401

    m_runresultsdir = strdup(runresultsdir.c_str());
    LOG(error) << runresultsdir ; 
}

/**
OpticksResource::getRunResultsDir
-----------------------------------

Used from OTracer::report 

**/
const char* OpticksResource::getRunResultsDir() const 
{
    return m_runresultsdir ;
}

bool OpticksResource::isDetectorType(const char* type_)
{
    return std::find(m_detector_types.begin(),m_detector_types.end(), type_) != m_detector_types.end()  ; 
}
bool OpticksResource::isResourceType(const char* type_)
{
    return std::find(m_resource_types.begin(),m_resource_types.end(), type_) != m_resource_types.end()  ; 
}








void OpticksResource::identifyGeometry()
{
   // TODO: somehow extract detector name from the exported file metadata or sidecar

   m_g4live   = idNameContains(G4LIVE) ;
   m_juno     = idNameContains("juno") ;
   m_dayabay  = idNameContains("DayaBay") ;
   m_dpib     = idNameContains("dpib") ;

   if(m_g4live == false && m_juno == false && m_dayabay == false && m_dpib == false )
   {
       const char* detector = getMetaValue("detector") ;
       if(detector)
       {
           if(     strcmp(detector, G4LIVE)  == 0) m_g4live = true ; 
           else if(strcmp(detector, DAYABAY) == 0) m_dayabay = true ; 
           else if(strcmp(detector, JUNO)    == 0) m_juno = true ; 
           else if(strcmp(detector, DPIB)    == 0) m_dpib = true ; 
           else 
                 m_other = true ;

           LOG(verbose) << "OpticksResource::identifyGeometry" 
                      << " metavalue detector " <<  detector 
                      ; 
       }
       else
       {
           m_other = true ;
       }
   }


   assert( m_g4live ^ m_juno ^ m_dayabay ^ m_dpib ^ m_other ); // exclusive-or
   
   if(m_g4live)  m_detector = G4LIVE ; 
   if(m_juno)    m_detector = JUNO ; 
   if(m_dayabay) m_detector = DAYABAY ; 
   if(m_dpib)    m_detector = DPIB ; 
   if(m_other)   m_detector = OTHER ; 

   if(m_detector == NULL)
       LOG(fatal) << "FAILED TO ASSIGN m_detector " ; 

   assert(m_detector);

   LOG(verbose) << "OpticksResource::identifyGeometry"
              << " m_detector " << m_detector
              ;

}

void OpticksResource::assignDefaultMaterial()
{
    m_default_material =  DEFAULT_MATERIAL_OTHER ; 
    if(m_juno)    m_default_material = DEFAULT_MATERIAL_JUNO ;
    if(m_dayabay) m_default_material = DEFAULT_MATERIAL_DYB ;

    m_default_medium =  DEFAULT_MEDIUM_OTHER ; 
    if(m_juno)    m_default_medium = DEFAULT_MEDIUM_JUNO ;
    if(m_dayabay) m_default_medium = DEFAULT_MEDIUM_DYB ;

    m_example_matnames =  EXAMPLE_MATNAMES_OTHER ; 
    if(m_juno)    m_example_matnames = EXAMPLE_MATNAMES_JUNO ;
    if(m_dayabay) m_example_matnames = EXAMPLE_MATNAMES_DYB ;

    m_sensor_surface = SENSOR_SURFACE_OTHER ; 
    if(m_juno)   m_sensor_surface =  SENSOR_SURFACE_JUNO  ; 
    if(m_dayabay) m_sensor_surface = SENSOR_SURFACE_DYB ; 

    m_default_frame = DEFAULT_FRAME_OTHER ; 
    if(m_juno)    m_default_frame = DEFAULT_FRAME_JUNO ;
    if(m_dayabay) m_default_frame = DEFAULT_FRAME_DYB ;

}

const char* OpticksResource::getDefaultMaterial()
{
    return m_default_material ; 
}
const char* OpticksResource::getDefaultMedium()
{
    return m_default_medium ; 
}
const char* OpticksResource::getExampleMaterialNames()
{
    return m_example_matnames ; 
}

const char* OpticksResource::getSensorSurface()
{
    return m_sensor_surface ; 
}
int OpticksResource::getDefaultFrame() const 
{
    return m_default_frame ; 
}





void OpticksResource::assignDetectorName()
{
   /**
       m_detector 
    -> m_detector_name    
          from lookup of a short list, for name standardization

       m_srcbase, m_detector_name  (eg /usr/local/opticks/opticksdata/export ,  DayaBay ) 
    -> m_detector_base             (eg /usr/local/opticks/opticksdata/export/DayaBay )

       m_detector_base 
    -> m_material_map              (eg /usr/local/opticks/opticksdata/export/DayaBay/ChromaMaterialMap.json )

   **/ 

   std::map<std::string, std::string> detname ; 
   detname[JUNO]    = "juno1707" ;
   detname[DAYABAY] = "DayaBay" ;
   detname[DPIB]    = "dpib" ;
   detname[OTHER]   = "other" ;

   assert(m_detector);

   LOG(LEVEL)
             << " m_detector " << m_detector
             ; 

   bool g4live = strcmp(m_detector, G4LIVE) == 0 ; 
   if(g4live)
   {
       const char* exename = m_key->getExename() ;
       assert(exename); 
       m_detector_name = strdup(exename) ; 
   }
   else
   {
       if(detname.count(m_detector) == 1) m_detector_name =  strdup(detname[m_detector].c_str()) ; 
   }

   if(m_detector_name == NULL)
   {
       Summary("FAILED TO ASSIGN m_detector_name");
       LOG(fatal) << "FAILED TO ASSIGN m_detector_name " ; 
   }
   assert(m_detector_name);


   // move detector_base relative to topdown export_dir 
   // rather than bottom-up (and src dependant) srcbase

   //if(m_srcbase )  // eg /usr/local/opticks/opticksdata/export
   if(m_export_dir )  // eg /usr/local/opticks/opticksdata/export
   {
        std::string detbase = BFile::FormPath(m_export_dir, m_detector_name);
        m_detector_base = strdup(detbase.c_str());
        m_res->addDir("detector_base",   m_detector_base  );
   }

   if(m_detector_base == NULL)
   {
       Summary("FAILED TO ASSIGN m_detector_base"); 
       LOG(fatal) 
           << "FAILED TO ASSIGN m_detector_base " ; 
   }
   assert(m_detector_base);


   std::string cmm = BFile::FormPath(m_detector_base, "ChromaMaterialMap.json" );
   m_material_map = strdup(cmm.c_str());

}






void OpticksResource::readG4Environment()
{
    // NB this relpath needs to match that in g4-;g4-export-ini
    //    it is relative to the install_prefix which 
    //    is canonically /usr/local/opticks
    //
    const char* inipath = InstallPathG4ENV();

    m_g4env = readIniEnvironment(inipath);
    if(m_g4env)
    {
        m_g4env->setEnvironment();
    }
    else
    {
        LOG(error)
                     << " MISSING inipath " << inipath
                     << " (create it with bash functions: g4-;g4-export-ini ) " 
                     ;
    }
}


void OpticksResource::readOpticksEnvironment()
{
    // NB this relpath needs to match that in opticksdata-;opticksdata-export-ini
    //    it is relative to the install_prefix which 
    //    is canonically /usr/local/opticks
    //
    const char* inipath = InstallPathOKDATA();
    LOG(LEVEL) << " inipath " << inipath ; 

    m_okenv = readIniEnvironment(inipath);
    if(m_okenv)
    {
        m_okenv->setEnvironment();
    }
    else
    {
        LOG(error)
                     << " MISSING inipath " << inipath 
                     << " (create it with bash functions: opticksdata-;opticksdata-export-ini ) " 
                     ;
    }
}

BEnv* OpticksResource::readIniEnvironment(const std::string& inipath)
{
    BEnv* env = NULL ; 
    if(BFile::ExistsFile(inipath.c_str()))
    {
        LOG(verbose) << "OpticksResource::readIniEnvironment" 
                  << " from " << inipath
                  ;

         env = BEnv::load(inipath.c_str()); 
    }
    return env ;  
}



void OpticksResource::readEnvironment()
{
/*


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



OPTICKS_GEOKEY 
    envvar naming another envvar (eg OPTICKSDATA_DAEPATH_DYB) that points to geometry file

*/
    if(m_ok->isDumpEnv())  SSys::DumpEnv("OPTICKS") ; 

    m_geokey = SSys::getenvvar("OPTICKS_GEOKEY", DEFAULT_GEOKEY);
    LOG(LEVEL) << " initial m_geokey " << m_geokey ;  

    const char* daepath = SSys::getenvvar(m_geokey);

    if(daepath == NULL)
    {
        LOG(error)
                     << " NO DAEPATH "
                     << " geokey " << m_geokey 
                     << " daepath " << ( daepath ? daepath : "NULL" )
                     ;
 
        //assert(0);
        setValid(false);
    } 

    m_ctrl         = SSys::getenvvar("OPTICKS_CTRL", DEFAULT_CTRL);
    m_meshfix      = SSys::getenvvar("OPTICKS_MESHFIX", DEFAULT_MESHFIX);
    m_meshfixcfg   = SSys::getenvvar("OPTICKS_MESHFIX_CFG", DEFAULT_MESHFIX_CFG);

    // idpath incorporates digest of geometry selection envvar 
    // allowing to benefit from caching as vary geometry selection 
    // while still only having a single source geometry file.

    if(daepath)
    {
        setupViaSrc(daepath, m_query->getQueryDigest() );  // this sets m_idbase, m_idfold, m_idname done in base BOpticksResource
        assert(m_idpath) ; 
        assert(m_idname) ; 
        assert(m_idfold) ; 
    }
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
    std::cout 
           << std::setw(40) << " getMergedMeshPath(0) " 
           << " : " 
           << mmsp
           << std::endl ; 

    std::string pmtp = getPmtPath(0);
    std::cout 
           << std::setw(40) << " getPmtPath(0) " 
           << " : " 
           << pmtp
           << std::endl ; 


    std::string bndlib = getPropertyLibDir("GBndLib");
    std::cout 
           << std::setw(40) << " getPropertyLibDir(\"GBndLib\") " 
           << " : " 
           << bndlib 
           << std::endl ; 

    const char* testcsgpath = getTestCSGPath() ; 
    std::cout 
           << std::setw(40) << " getTestCSGPath() " 
           << " : " 
           <<  ( testcsgpath ? testcsgpath : "NULL" )
           << std::endl ; 
    
}



void OpticksResource::Summary(const char* msg)
{
    std::cerr << msg << std::endl ; 

    std::cerr << "install_prefix    : " <<  (m_install_prefix ? m_install_prefix : "NULL" ) << std::endl ; 
    std::cerr << "opticksdata_dir   : " <<  (m_opticksdata_dir ? m_opticksdata_dir : "NULL" ) << std::endl ; 
    std::cerr << "geocache_dir      : " <<  (m_geocache_dir ? m_geocache_dir : "NULL" ) << std::endl ; 
    std::cerr << "resource_dir      : " <<  (m_resource_dir ? m_resource_dir : "NULL" ) << std::endl ; 
    std::cerr << "valid    : " <<  (m_valid ? "valid" : "NOT VALID" ) << std::endl ; 
    std::cerr << "geokey   : " <<  (m_geokey?m_geokey:"NULL") << std::endl; 
    std::cerr << "daepath  : " <<  (m_daepath?m_daepath:"NULL") << std::endl; 
    std::cerr << "gdmlpath : " <<  (m_gdmlpath?m_gdmlpath:"NULL") << std::endl; 
    std::cerr << "gltfpath : " <<  (m_gltfpath?m_gltfpath:"NULL") << std::endl; 
    std::cerr << "metapath : " <<  (m_metapath?m_metapath:"NULL") << std::endl; 
    std::cerr << "query    : " <<  (m_query?m_query->getQueryString():"NULL") << std::endl; 
    std::cerr << "ctrl     : " <<  (m_ctrl?m_ctrl:"NULL") << std::endl; 
    std::cerr << "digest   : " <<  (m_srcdigest?m_srcdigest:"NULL") << std::endl; 
    std::cerr << "idpath   : " <<  (m_idpath?m_idpath:"NULL") << std::endl; 
    std::cerr << "idpath_tmp " <<  (m_idpath_tmp?m_idpath_tmp:"NULL") << std::endl; 
    std::cerr << "idfold   : " <<  (m_idfold?m_idfold:"NULL") << std::endl; 
    std::cerr << "idname   : " <<  (m_idname?m_idname:"NULL") << std::endl; 

    std::cerr << "detector : " <<  (m_detector?m_detector:"NULL") << std::endl; 
    std::cerr << "detector_name : " <<  (m_detector_name?m_detector_name:"NULL") << std::endl; 
    std::cerr << "detector_base : " <<  (m_detector_base?m_detector_base:"NULL") << std::endl; 
    std::cerr << "material_map  : " <<  (m_material_map?m_material_map:"NULL") << std::endl; 
    std::cerr << "getPmtPath(0) : " <<  (m_detector_base?getPmtPath(0):"-") << std::endl; 
    std::cerr << "getMMPath(0)  : " <<  (m_detector_base?getMergedMeshPath(0):"-") << std::endl; 
    std::cerr << "getBasePath(?): " <<  (m_srcbase?getBasePath("?"):"-") << std::endl; 
    std::cerr << "meshfix  : " <<  (m_meshfix ? m_meshfix : "NULL" ) << std::endl; 
    std::cerr << "example_matnames  : " <<  (m_example_matnames ? m_example_matnames : "NULL" ) << std::endl; 
    std::cerr << "sensor_surface    : " <<  (m_sensor_surface ? m_sensor_surface : "NULL" ) << std::endl; 
    std::cerr << "default_medium    : " <<  (m_default_medium ? m_default_medium : "NULL" ) << std::endl; 
    std::cerr << "------ from " << ( m_metapath ? m_metapath : "NULL" ) << " -------- " << std::endl ;  

    typedef std::map<std::string, std::string> SS ;
    for(SS::const_iterator it=m_metadata.begin() ; it != m_metadata.end() ; it++)
        std::cerr <<  std::setw(10) << it->first.c_str() << ":" <<  it->second.c_str() << std::endl  ;


    m_res->dumpPaths("dumpPaths");
    m_res->dumpDirs("dumpDirs");

}


std::string OpticksResource::desc() const 
{
    std::stringstream ss ; 

    std::time_t* slwt = BFile::SinceLastWriteTime(m_idpath); 
    long seconds = slwt ? *slwt : -1 ;

    float minutes = float(seconds)/float(60) ; 
    float hours = float(seconds)/float(60*60) ; 
    float days = float(seconds)/float(60*60*24) ; 

    ss << "OpticksResource::desc"
       << " digest " << ( m_srcdigest ? m_srcdigest : "NULL" )
       << " age.tot_seconds " << std::setw(6) << seconds
       << std::fixed << std::setprecision(3)  
       << " age.tot_minutes " << std::setw(6) << minutes
       << " age.tot_hours " << std::setw(6) << hours
       << " age.tot_days " << std::setw(10) << days
       ;

    return ss.str();
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

std::string OpticksResource::getBasePath(const char* rel)
{
    assert(m_srcbase);
    fs::path dir(m_srcbase);
    if(rel) dir /= rel ;
    return dir.string() ;
}



std::string OpticksResource::getPmtPath(unsigned int index, bool relative)
{
    return relative ?
                        getRelativePath("GPmt", index)
                    :
                        getDetectorPath("GPmt", index)
                    ;

    // relocate from inside the "digested" idpath up to eg export/Dayabay/GPmt/0
    // as analytic PMT definitions dont change with the geometry selection parameters

}

std::string OpticksResource::getObjectPath(const char* name, unsigned int index)
{
    const char* idpath = getIdPath();
    assert(idpath && "OpticksResource::getObjectPath idpath not set");
    fs::path dir(idpath);
    dir /= name ;
    dir /= boost::lexical_cast<std::string>(index) ;
    return dir.string() ;
}


std::string OpticksResource::getRelativePath(const char* name, unsigned int index)
{
    // used eg by GPmt::loadFromCache returning "GPmt/0"
    fs::path reldir(name);
    reldir /= boost::lexical_cast<std::string>(index) ;
    return reldir.string() ;
}


std::string OpticksResource::getDetectorPath(const char* name, unsigned int index)
{
    assert(m_detector_base && "OpticksResource::getDetectorPath detector_path not set");
    fs::path dir(m_detector_base);
    dir /= name ;
    dir /= boost::lexical_cast<std::string>(index) ;
    return dir.string() ;
}


/**
OpticksResource::getPreferenceDir
-------------------------------------

::

    find /usr/local/opticks/opticksdata/export/DayaBay -name order.json
    /usr/local/opticks/opticksdata/export/DayaBay/GMaterialLib/order.json
    /usr/local/opticks/opticksdata/export/DayaBay/GSurfaceLib/order.json

    ## important materials arranged to have low indices, 
    ## for 4-bit per step GPU recording of indices

    epsilon:optickscore blyth$ cat /usr/local/opticks/opticksdata/export/DayaBay/GMaterialLib/order.json
    {
        "GdDopedLS": "1",
        "LiquidScintillator": "2",
        "Acrylic": "3",
        "MineralOil": "4",
        "Bialkali": "5",
        "IwsWater": "6",
        "Water": "7",
        ...

**/

std::string OpticksResource::getPreferenceDir(const char* type, const char* udet, const char* subtype )
{
    //if(isG4Live()) return EMPTY ;   // g4live running needs o standardize material order too 

    bool detector_type = isDetectorType(type) ; // GScintillatorLib,GMaterialLib,GSurfaceLib,GBndLib,GSourceLib
    bool resource_type = isResourceType(type) ; // GFlags,OpticksColors 

    const char* prefbase = PREFERENCE_BASE ;
    if(detector_type) prefbase = m_detector_base ; 
    if(resource_type) prefbase = m_resource_dir ;   // one of the top down dirs, set in base BOpticksResource, eg /usr/local/opticks/opticksdata/resource/ containing GFlags, OpticksColors

    if(detector_type)
    {
        assert(udet == NULL); // detector types dont need another detector subdir
    }


    fs::path prefdir(prefbase) ;
    if(udet) prefdir /= udet ;
    prefdir /= type ; 
    if(subtype) prefdir /= subtype ; 
    std::string pdir = prefdir.string() ;

    LOG(verbose) << "OpticksResource::getPreferenceDir"
              << " type " << type 
              << " detector_type " << detector_type
              << " resource_type " << resource_type
              << " udet " << udet
              << " subtype " << subtype 
              << " pdir " << pdir 
              ;

    return pdir ; 
}


/*

GPropLib/GAttrSeq prefs such as materials, surfaces, boundaries and flags 
come in threes (abbrev.json, color.json and order.json)
these provided attributes for named items in sequences. 

::

    delta:GMaterialLib blyth$ cat ~/.opticks/GMaterialLib/abbrev.json 
    {
        "ADTableStainlessSteel": "AS",
        "Acrylic": "Ac",
        "Air": "Ai",
        "Aluminium": "Al",


Some such triplets like GMaterialLib, GSurfaceLib belong within "Detector scope" 
as they depend on names used within a particular detector. 

* not within IDPATH as changing geo selection doesnt change names
* not within user scope, as makes sense to standardize 

Moved from ~/.opticks into opticksdata/export/<detname>  m_detector_base
by env-;export-;export-copy-detector-prefs-

::

    delta:GMaterialLib blyth$ l /usr/local/opticks/opticksdata/export/DayaBay/
    drwxr-xr-x  3 blyth  staff  102 Jul  5 10:58 GPmt



*/



bool OpticksResource::loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    bool empty = prefdir.empty() ; 

    LOG(verbose) << "OpticksResource::loadPreference(MSS)" 
              << " prefdir " << prefdir
              << " name " << name
              << " empty " << ( empty ? "YES" : "NO" )
              ; 


    typedef Map<std::string, std::string> MSS ;  
    MSS* pref = empty ? NULL :  MSS::load(prefdir.c_str(), name ) ; 
    if(pref)
    {
        mss = pref->getMap(); 
    }

    return pref != NULL ; 
}

bool OpticksResource::loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    bool empty = prefdir.empty() ; 

    LOG(verbose) << "OpticksResource::loadPreference(MSU)" 
              << " prefdir " << prefdir
              << " name " << name
              << " empty " << ( empty ? "YES" : "NO" )
              ; 

    typedef Map<std::string, unsigned int> MSU ;  
    MSU* pref = empty ? NULL : MSU::load(prefdir.c_str(), name ) ; 
    if(pref)
    {
        msu = pref->getMap(); 
    }
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
    {
        mdd = meta->getMap(); 
    }
    else
    {
        LOG(debug) << "OpticksResource::loadMetadata"
                  << " no path " << path 
                  ;
    } 
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
        //
        // The formerly named GCache is exceptionally is not an attribution triplet,
        // to reflect this manually changed name to OpticksColors
        //
        //std::string prefdir = getPreferenceDir("GCache"); 
        std::string prefdir = getPreferenceDir("OpticksColors"); 
        bool empty = prefdir.empty(); 
        m_colors = empty ? NULL :  OpticksColors::load(prefdir.c_str(),"OpticksColors.json");   // colorname => hexcode

        if(empty)
        {
            LOG(debug) << "OpticksResource::getColors"
                       << " empty PreferenceDir for OpticksColors " 
                       ;
        }
    }
    return m_colors ;
}

OpticksFlags* OpticksResource::getFlags() const 
{
    return m_flags ;
}

void OpticksResource::saveFlags(const char* dir)
{
    OpticksFlags* flags = getFlags();
    LOG(info) << " dir " << dir ;
    flags->save(dir);
}


OpticksAttrSeq* OpticksResource::getFlagNames()
{
    if(!m_flagnames)
    {
        OpticksFlags* flags = getFlags();
        Index* index = flags->getIndex();

        m_flagnames = new OpticksAttrSeq(m_ok, "GFlags");
        m_flagnames->loadPrefs(); // color, abbrev and order 
        m_flagnames->setSequence(index);
        m_flagnames->setCtrl(OpticksAttrSeq::SEQUENCE_DEFAULTS);    
    }
    return m_flagnames ; 
}


std::map<unsigned int, std::string> OpticksResource::getFlagNamesMap()
{
    OpticksAttrSeq* flagnames = getFlagNames();
    return flagnames->getNamesMap() ; 
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
    }   
    return m_types ;
}


void OpticksResource::saveTypes(const char* dir)
{
    LOG(info) << "OpticksResource::saveTypes " << dir ; 

    Types* types = getTypes(); 
    types->saveFlags(dir, ".ini");
}


NSensorList* OpticksResource::getSensorList()
{
    if(!m_sensor_list)
    {
        m_sensor_list = new NSensorList();
        m_sensor_list->load( m_idmappath );
        LOG(info) << "OpticksResource::getSensorList " << m_sensor_list->description() ; 
    }
    return m_sensor_list ; 
}
