#include <cassert>
#include <cstring>
#include <iostream>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "OKConf_Config.hh"

#include "SLog.hh"
#include "SSys.hh"
#include "SAr.hh"

#include "BFile.hh"
#include "BStr.hh"
#include "BPath.hh"
#include "BResource.hh"
#include "BOpticksResource.hh"
#include "BOpticksKey.hh"

#include "PLOG.hh"

const char* BOpticksResource::G4ENV_RELPATH = "externals/config/geant4.ini" ;
const char* BOpticksResource::OKDATA_RELPATH = "opticksdata/config/opticksdata.ini" ; // TODO: relocate into geocache



const char* BOpticksResource::EMPTY  = "" ; 

const char* BOpticksResource::G4LIVE  = "g4live" ; 
const char* BOpticksResource::JUNO    = "juno1707" ; 
const char* BOpticksResource::DAYABAY = "dayabay" ; 
const char* BOpticksResource::DPIB    = "PmtInBox" ; 
const char* BOpticksResource::OTHER   = "other" ; 

const char* BOpticksResource::PREFERENCE_BASE = "$HOME/.opticks" ; 


// TODO: having these defaults compiled in is problematic, better to read in from
//       json/ini so they are available at python level

const char* BOpticksResource::DEFAULT_GEOKEY = "OPTICKSDATA_DAEPATH_DYB" ; 
const char* BOpticksResource::DEFAULT_QUERY = "range:3153:12221" ; 
const char* BOpticksResource::DEFAULT_QUERY_LIVE = "all" ;
//const char* BOpticksResource::DEFAULT_QUERY_LIVE = "range:3153:12221" ;  // <-- EXPEDIENT ASSUMPTION THAT G4LIVE GEOMETRY IS DYB  
const char* BOpticksResource::DEFAULT_CTRL = "" ; 
const char* BOpticksResource::DEFAULT_MESHFIX = "iav,oav" ; 
const char* BOpticksResource::DEFAULT_MESHFIX_CFG = "100,100,10,-0.999" ; 

const char* BOpticksResource::DEFAULT_MATERIAL_DYB  = "GdDopedLS" ; 
const char* BOpticksResource::DEFAULT_MATERIAL_JUNO = "LS" ; 
const char* BOpticksResource::DEFAULT_MATERIAL_OTHER = "Water" ; 

const char* BOpticksResource::DEFAULT_MEDIUM_DYB  = "MineralOil" ; 
const char* BOpticksResource::DEFAULT_MEDIUM_JUNO = "Water" ; 
const char* BOpticksResource::DEFAULT_MEDIUM_OTHER = "Water" ; 

const char* BOpticksResource::EXAMPLE_MATNAMES_DYB = "GdDopedLS,Acrylic,LiquidScintillator,MineralOil,Bialkali" ;
const char* BOpticksResource::EXAMPLE_MATNAMES_JUNO = "LS,Acrylic" ; 
const char* BOpticksResource::EXAMPLE_MATNAMES_OTHER = "LS,Acrylic" ; 

const char* BOpticksResource::SENSOR_SURFACE_DYB = "lvPmtHemiCathodeSensorSurface" ;
const char* BOpticksResource::SENSOR_SURFACE_JUNO = "SS-JUNO-UNKNOWN" ; 
const char* BOpticksResource::SENSOR_SURFACE_OTHER = "SS-OTHER-UNKNOWN" ; 

const int BOpticksResource::DEFAULT_FRAME_OTHER = 0 ; 
const int BOpticksResource::DEFAULT_FRAME_DYB = 3153 ; 
const int BOpticksResource::DEFAULT_FRAME_JUNO = 62593 ; 






const plog::Severity BOpticksResource::LEVEL = debug ; 

BOpticksResource::BOpticksResource()
    :
    m_log(new SLog("BOpticksResource::BOpticksResource","",debug)),
    m_setup(false),
    m_key(BOpticksKey::GetKey()),   // will be NULL unless BOpticksKey::SetKey has been called 
    m_id(NULL),
    m_res(new BResource),
    m_layout(SSys::getenvint("OPTICKS_RESOURCE_LAYOUT", 0)),
    m_install_prefix(NULL),
    m_opticksdata_dir(NULL),
    m_geocache_dir(NULL),
    m_resource_dir(NULL),
    m_gensteps_dir(NULL),
    m_export_dir(NULL),
    m_installcache_dir(NULL),
    m_rng_installcache_dir(NULL),
    m_okc_installcache_dir(NULL),
    m_ptx_installcache_dir(NULL),
    m_tmpuser_dir(NULL),
    m_srcpath(NULL),
    m_srcfold(NULL),
    m_srcbase(NULL),
    m_srcdigest(NULL),
    m_idfold(NULL),
    m_idfile(NULL),
    m_idgdml(NULL),
    m_idsubd(NULL),
    m_idname(NULL),
    m_idpath(NULL),
    m_idpath_tmp(NULL),
    m_srcevtbase(NULL),
    m_evtbase(NULL),
    m_debugging_idpath(NULL),
    m_debugging_idfold(NULL),
    m_daepath(NULL),
    m_gdmlpath(NULL),
    m_srcgdmlpath(NULL),
    m_srcgltfpath(NULL),
    m_metapath(NULL),
    m_idmappath(NULL),
    m_g4codegendir(NULL),
    m_cachemetapath(NULL),
    m_primariespath(NULL),
    m_directgensteppath(NULL),
    m_directphotonspath(NULL),
    m_gltfpath(NULL)
{
    init();
    (*m_log)("DONE"); 
}


BOpticksResource::~BOpticksResource()
{
}

void BOpticksResource::init()
{
    LOG(LEVEL) << "layout : " << m_layout ; 

    initInstallPrefix() ;
    initTopDownDirs();
    initDebuggingIDPATH();
}

const char* BOpticksResource::getInstallPrefix() // canonically /usr/local/opticks
{
    return m_install_prefix ; 
}

const char* BOpticksResource::InstallPath(const char* relpath) 
{
    std::string path = BFile::FormPath(ResolveInstallPrefix(), relpath) ;
    return strdup(path.c_str()) ;
}

const char* BOpticksResource::InstallPathG4ENV() 
{
    return InstallPath(G4ENV_RELPATH);
}
const char* BOpticksResource::InstallPathOKDATA() 
{
    return InstallPath(OKDATA_RELPATH);
}


std::string BOpticksResource::getInstallPath(const char* relpath) const 
{
    std::string path = BFile::FormPath(m_install_prefix, relpath) ;
    return path ;
}


const char* BOpticksResource::ResolveInstallPrefix()  // static
{
    const char* evalue = SSys::getenvvar(INSTALL_PREFIX_KEY);  
    return evalue == NULL ?  strdup(OKCONF_OPTICKS_INSTALL_PREFIX) : evalue ; 
}


const char* BOpticksResource::INSTALL_PREFIX_KEY = "OPTICKS_INSTALL_PREFIX" ; 
const char* BOpticksResource::INSTALL_PREFIX_KEY2 = "OPTICKSINSTALLPREFIX" ; 

void BOpticksResource::initInstallPrefix()
{
    m_install_prefix = ResolveInstallPrefix();
    m_res->addDir("install_prefix", m_install_prefix );

    bool overwrite = true ; 
    int rc = SSys::setenvvar(INSTALL_PREFIX_KEY, m_install_prefix, overwrite );  
    // always set for uniformity 

    LOG(verbose) << "OpticksResource::adoptInstallPrefix " 
               << " install_prefix " << m_install_prefix  
               << " key " << INSTALL_PREFIX_KEY
               << " rc " << rc
              ;   
 
    assert(rc==0); 

    // for test geometry config underscore has special meaning, so duplicate the envvar without underscore in the key
    int rc2 = SSys::setenvvar(INSTALL_PREFIX_KEY2 , m_install_prefix, true );  
    assert(rc2==0); 


    // The CMAKE_INSTALL_PREFIX from opticks-;opticks-cmake 
    // is set to the result of the opticks-prefix bash function 
    // at configure time.
    // This is recorded into a config file by okc-/CMakeLists.txt 
    // and gets compiled into the OpticksCore library.
    //  
    // Canonically it is :  /usr/local/opticks 

    m_res->addPath("g4env_ini", InstallPathG4ENV() );
    m_res->addPath("okdata_ini", InstallPathOKDATA() );

}



std::string BOpticksResource::getGeoCachePath(const char* rela, const char* relb, const char* relc, const char* reld ) const 
{
    std::string path = BFile::FormPath(m_geocache_dir, rela, relb, relc, reld ) ;
    return path ;
}

std::string BOpticksResource::getResultsPath(const char* rela, const char* relb, const char* relc, const char* reld ) const 
{
    std::string path = BFile::FormPath(m_results_dir, rela, relb, relc, reld ) ;
    return path ;
}


std::string BOpticksResource::getIdPathPath(const char* rela, const char* relb, const char* relc, const char* reld ) const 
{
    const char* idpath = getIdPath(); 
    LOG(debug) << " idpath " << idpath ; 

    std::string path = BFile::FormPath(idpath, rela, relb, relc, reld ) ;
    return path ;
}






void BOpticksResource::initTopDownDirs()
{ 
    m_opticksdata_dir      = OpticksDataDir() ;   // eg /usr/local/opticks/opticksdata
    m_geocache_dir         = GeoCacheDir() ;      // eg /usr/local/opticks/geocache
    m_results_dir          = ResultsDir() ;       // eg /usr/local/opticks/results
    m_resource_dir         = ResourceDir() ;      // eg /usr/local/opticks/opticksdata/resource
    m_gensteps_dir         = GenstepsDir() ;      // eg /usr/local/opticks/opticksdata/gensteps
    m_export_dir           = ExportDir() ;        // eg /usr/local/opticks/opticksdata/export
    m_installcache_dir     = InstallCacheDir() ;  // eg /usr/local/opticks/installcache

    m_rng_installcache_dir = RNGInstallPath() ;   // eg /usr/local/opticks/installcache/RNG
    m_okc_installcache_dir = OKCInstallPath() ;   // eg /usr/local/opticks/installcache/OKC
    m_ptx_installcache_dir = PTXInstallPath() ;   // eg /usr/local/opticks/installcache/PTX


    m_res->addDir("opticksdata_dir", m_opticksdata_dir);
    m_res->addDir("geocache_dir",    m_geocache_dir );
    m_res->addDir("results_dir",     m_results_dir );
    m_res->addDir("resource_dir",    m_resource_dir );
    m_res->addDir("gensteps_dir",    m_gensteps_dir );
    m_res->addDir("export_dir",      m_export_dir);
    m_res->addDir("installcache_dir", m_installcache_dir );

    m_res->addDir("rng_installcache_dir", m_rng_installcache_dir );
    m_res->addDir("okc_installcache_dir", m_okc_installcache_dir );
    m_res->addDir("ptx_installcache_dir", m_ptx_installcache_dir );

    m_tmpuser_dir = MakeTmpUserDir("opticks", NULL) ; 
    m_res->addDir( "tmpuser_dir", m_tmpuser_dir ); 
}

void BOpticksResource::initDebuggingIDPATH()
{
    // directories based on IDPATH envvar ... this is for debugging 
    // and as workaround for npy level tests to access geometry paths 
    // NB should only be used at that level... at higher levels use OpticksResource for this

    
    m_debugging_idpath = SSys::getenvvar("IDPATH") ;

    if(!m_debugging_idpath) return ; 

    std::string idfold = BFile::ParentDir(m_debugging_idpath) ;
    m_debugging_idfold = strdup(idfold.c_str());

}




const char* BOpticksResource::getDebuggingTreedir(int argc, char** argv)
{
    int arg1 = BStr::atoi(argc > 1 ? argv[1] : "-1", -1 );
    const char* idfold = getDebuggingIDFOLD() ;

    std::string treedir ; 

    if(arg1 > -1) 
    {   
        // 1st argument is an integer
        treedir = BFile::FormPath( idfold, "extras", BStr::itoa(arg1) ) ; 
    }   
    else if( argc > 1)
    {
        // otherwise string argument
        treedir = argv[1] ;
    }
    else
    {   
        treedir = BFile::FormPath( idfold, "extras") ;
    }   
    return treedir.empty() ? NULL : strdup(treedir.c_str()) ; 
}




const char* BOpticksResource::InstallCacheDir(){return MakeInstallPath(ResolveInstallPrefix(), "installcache",  NULL); }
const char* BOpticksResource::OpticksDataDir(){ return MakeInstallPath(ResolveInstallPrefix(), "opticksdata",  NULL); }
const char* BOpticksResource::GeoCacheDir(){    return MakeInstallPath(ResolveInstallPrefix(), "geocache",  NULL); }
const char* BOpticksResource::ResourceDir(){    return MakeInstallPath(ResolveInstallPrefix(), "opticksdata", "resource" ); }
const char* BOpticksResource::GenstepsDir(){    return MakeInstallPath(ResolveInstallPrefix(), "opticksdata", "gensteps" ); }
const char* BOpticksResource::ExportDir(){      return MakeInstallPath(ResolveInstallPrefix(), "opticksdata", "export" ); }

const char* BOpticksResource::PTXInstallPath(){ return MakeInstallPath(ResolveInstallPrefix(), "installcache", "PTX"); }
const char* BOpticksResource::RNGInstallPath(){ return MakeInstallPath(ResolveInstallPrefix(), "installcache", "RNG"); }
const char* BOpticksResource::OKCInstallPath(){ return MakeInstallPath(ResolveInstallPrefix(), "installcache", "OKC"); }

// problematic in readonly installs : because results do not belong with install paths 
const char* BOpticksResource::ResultsDir(){     return MakeInstallPath(ResolveInstallPrefix(), "results",  NULL); }



std::string BOpticksResource::PTXPath(const char* name, const char* target)
{
    const char* ptx_installcache_dir = PTXInstallPath();
    return PTXPath(name, target, ptx_installcache_dir);
}


const char* BOpticksResource::getInstallDir() {         return m_install_prefix ; }   
const char* BOpticksResource::getOpticksDataDir() {     return m_opticksdata_dir ; }   
const char* BOpticksResource::getGeoCacheDir() {        return m_geocache_dir ; }   
const char* BOpticksResource::getResultsDir() {         return m_results_dir ; }   
const char* BOpticksResource::getResourceDir() {        return m_resource_dir ; } 
const char* BOpticksResource::getExportDir() {          return m_export_dir ; } 

const char* BOpticksResource::getInstallCacheDir() {    return m_installcache_dir ; } 
const char* BOpticksResource::getRNGInstallCacheDir() { return m_rng_installcache_dir ; } 
const char* BOpticksResource::getOKCInstallCacheDir() { return m_okc_installcache_dir ; } 
const char* BOpticksResource::getPTXInstallCacheDir() { return m_ptx_installcache_dir ; } 
const char* BOpticksResource::getTmpUserDir() const {   return m_tmpuser_dir ; } 


const char* BOpticksResource::getDebuggingIDPATH() {    return m_debugging_idpath ; } 
const char* BOpticksResource::getDebuggingIDFOLD() {    return m_debugging_idfold ; } 




const char* BOpticksResource::MakeSrcPath(const char* srcpath, const char* ext) 
{
    std::string path = BFile::ChangeExt(srcpath, ext ); 
    return strdup(path.c_str());
}
const char* BOpticksResource::MakeSrcDir(const char* srcpath, const char* sub) 
{
    std::string srcdir = BFile::ParentDir(srcpath); 
    std::string path = BFile::FormPath(srcdir.c_str(), sub ); 
    return strdup(path.c_str());
}
const char* BOpticksResource::MakeTmpUserDir(const char* sub, const char* rel) 
{
    const char* base = "/tmp" ; 
    const char* user = SSys::username(); 
    std::string path = BFile::FormPath(base, user, sub, rel ) ; 
    return strdup(path.c_str());
}






// cannot be static as IDPATH not available statically 
const char* BOpticksResource::makeIdPathPath(const char* rela, const char* relb, const char* relc, const char* reld) 
{
    std::string path = getIdPathPath(rela, relb, relc, reld);  
    return strdup(path.c_str());
}


void BOpticksResource::setSrcPath(const char* srcpath)  
{
    // invoked by setupViaSrc or setupViaID
    //
    //
    // example srcpath : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    // NB not in geocache, points to actual G4DAE export from opticksdata 

    assert( srcpath );
    m_srcpath = strdup( srcpath );

    std::string srcfold = BFile::ParentDir(m_srcpath);
    m_srcfold = strdup(srcfold.c_str());

    std::string srcbase = BFile::ParentDir(srcfold.c_str());
    m_srcbase = strdup(srcbase.c_str());

    m_res->addDir("srcfold", m_srcfold ); 
    m_res->addDir("srcbase", m_srcbase ); 

    m_daepath = MakeSrcPath(m_srcpath,".dae"); 
    m_srcgdmlpath = MakeSrcPath(m_srcpath,".gdml"); 
    m_srcgltfpath = MakeSrcPath(m_srcpath,".gltf"); 
    m_metapath = MakeSrcPath(m_srcpath,".ini"); 
    m_idmappath = MakeSrcPath(m_srcpath,".idmap"); 


    m_res->addPath("srcpath", m_srcpath );
    m_res->addPath("daepath", m_daepath );
    m_res->addPath("srcgdmlpath", m_srcgdmlpath );
    m_res->addPath("srcgltfpath", m_srcgltfpath );
    m_res->addPath("metapath", m_metapath );
    m_res->addPath("idmappath", m_idmappath );

    m_g4codegendir = MakeSrcDir(m_srcpath,"g4codegen"); 
    m_res->addDir("g4codegendir", m_g4codegendir ); 

    std::string idname = BFile::ParentName(m_srcpath);
    m_idname = strdup(idname.c_str());   // idname is name of dir containing the srcpath eg DayaBay_VGDX_20140414-1300

    std::string idfile = BFile::Name(m_srcpath);
    m_idfile = strdup(idfile.c_str());    // idfile is name of srcpath geometry file, eg g4_00.dae

    m_res->addName("idname", m_idname ); 
    m_res->addName("idfile", m_idfile ); 




}

void BOpticksResource::setSrcDigest(const char* srcdigest)
{
    assert( srcdigest );
    m_srcdigest = strdup( srcdigest );
}




void BOpticksResource::setupViaID(const char* idpath)
{
    assert( !m_setup );
    m_setup = true ; 

    m_id = new BPath( idpath ); // juicing the IDPATH
    const char* srcpath = m_id->getSrcPath(); 
    const char* srcdigest = m_id->getSrcDigest(); 

    setSrcPath( srcpath );
    setSrcDigest( srcdigest );
}

/**
BOpticksResource::setupViaKey
===============================

Invoked from OpticksResource::init only when 
BOpticksKey::SetKey called prior to Opticks instanciation.   

Legacy mode alternative to this is setupViaSrc.

This is used for live/direct mode running, 
see for example CerenkovMinimal ckm- 

**/

void BOpticksResource::setupViaKey()
{
    // * this in invoked only for G4LIVE running 
    // * there is no opticksdata srcpath or associated 
    //   opticksdata paths for live running 

    assert( !m_setup ) ;  
    m_setup = true ; 
    assert( m_key ) ; // BOpticksResource::setupViaKey called with a NULL key 

    LOG(LEVEL) << std::endl << m_key->desc()  ;  

    m_layout = m_key->getLayout(); 


    const char* layout = BStr::itoa(m_layout) ;
    m_res->addName("OPTICKS_RESOURCE_LAYOUT", layout );

    const char* srcdigest = m_key->getDigest(); 
    setSrcDigest( srcdigest );
     
    const char* idname = m_key->getIdname() ;  // eg OpNovice_World_g4live
    assert(idname) ; 
    m_idname = strdup(idname);         
    m_res->addName("idname", m_idname ); 

    const char* idsubd = m_key->getIdsubd() ; //  eg  g4ok_gltf
    assert(idsubd) ; 
    m_idsubd = strdup(idsubd); 
    m_res->addName("idsubd", m_idsubd ); 

    const char* idfile = m_key->getIdfile() ; //  eg  g4ok.gltf
    assert(idfile) ; 
    m_idfile = strdup(idfile); 
    m_res->addName("idfile", m_idfile ); 

    const char* idgdml = m_key->getIdGDML() ; //  eg  g4ok.gdml
    assert(idgdml) ; 
    m_idgdml = strdup(idgdml); 
    m_res->addName("idgdml", m_idgdml ); 


    std::string fold = getGeoCachePath(  m_idname ) ; 
    m_idfold = strdup(fold.c_str()) ; 
    m_res->addDir("idfold", m_idfold ); 

    std::string idpath = getGeoCachePath( m_idname, m_idsubd, m_srcdigest, layout );

    LOG(LEVEL)
           <<  " idname " << m_idname 
           <<  " idfile " << m_idfile 
           <<  " srcdigest " << m_srcdigest
           <<  " idpath " << idpath 
            ;

    m_idpath = strdup(idpath.c_str()) ; 
    m_res->addDir("idpath", m_idpath ); 

    m_gltfpath = makeIdPathPath(m_idfile) ; // not a srcpath for G4LIVE, but potential cache file 
    m_res->addPath("gltfpath", m_gltfpath ); 

    m_gdmlpath = makeIdPathPath(m_idgdml) ; // output gdml 
    m_res->addPath("gdmlpath", m_gdmlpath ); 


    m_g4codegendir = makeIdPathPath("g4codegen" );
    m_res->addDir("g4codegendir", m_g4codegendir ); 

    m_cachemetapath = makeIdPathPath("cachemeta.json");  
    m_res->addPath("cachemetapath", m_cachemetapath ); 

    m_primariespath = makeIdPathPath("primaries.npy");  
    m_res->addPath("primariespath", m_primariespath ); 

    m_directgensteppath = makeIdPathPath("directgenstep.npy");  
    m_res->addPath("directgensteppath", m_directgensteppath ); 

    m_directphotonspath = makeIdPathPath("directphotons.npy");  
    m_res->addPath("directphotonspath", m_directphotonspath ); 

    const char* user = SSys::username(); 
    m_srcevtbase = makeIdPathPath("source"); 
    m_res->addDir( "srcevtbase", m_srcevtbase ); 

    const char* exename = SAr::Instance->exename(); 
    m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("tmp", user, exename ) ;  
    ///  should this always be KeySource ???
    ///      NO : KeySource means that the current executable is same as the exename 
    ///           enshrined into the geocache : ie the geocache creator  

    m_res->addDir( "evtbase", m_evtbase ); 

/**
   
* first geometry + genstep collecting and writing executable is special, 
  it writes its event and genstep into a distinctive "standard" directory (resource "srcevtbase") 
  within the geocache keydir 

* all other executables sharing the same keydir can put their events underneath 
  a relpath named after the executable (resource "evtbase")   

**/

}


/**
BOpticksResource::setupViaSrc  LEGACY approach to resource setup, based on envvars pointing at src dae
------------------------------------------------------------------------------------------------------------

Invoked from OpticksResource::init OpticksResource::readEnvironment

Direct mode equivalent (loosely speaking) is setupViaKey


**/

void BOpticksResource::setupViaSrc(const char* srcpath, const char* srcdigest)
{  
    LOG(LEVEL) 
        << " srcpath " << srcpath 
        << " srcdigest " << srcdigest
        ;

 
    assert( !m_setup );
    m_setup = true ; 

    setSrcPath(srcpath);
    setSrcDigest(srcdigest);
    
    const char* layout = BStr::itoa(m_layout) ;
    m_res->addName("OPTICKS_RESOURCE_LAYOUT", layout );


    if(m_layout == 0)  // geocache co-located with the srcpath typically from opticksdata
    {
        m_idfold = strdup(m_srcfold);
     
        std::string kfn = BStr::insertField( m_srcpath, '.', -1 , m_srcdigest );
        m_idpath = strdup(kfn.c_str());

        // internal setting of envvar 
        assert(SSys::setenvvar("IDPATH", m_idpath, true )==0);  // uses putenv for windows mingw compat 

        // Where is IDPATH internal envvar used ? 
        //    Mainly by NPY tests as a resource access workaround as NPY 
        //    is lower level than optickscore- so lacks its resource access machinery.
        //
        //  TODO: eliminate use of IDPATH internal envvar now that BOpticksResource has the info
    } 
    else if(m_layout > 0)  // geocache decoupled from opticksdata
    {
        std::string fold = getGeoCachePath(  m_idname ) ; 
        m_idfold = strdup(fold.c_str()) ; 

        std::string idpath = getGeoCachePath( m_idname, m_idfile, m_srcdigest, layout );
        m_idpath = strdup(idpath.c_str()) ; 
    }

    m_res->addDir("idfold", m_idfold );
    m_res->addDir("idpath", m_idpath );

    m_res->addDir("idpath_tmp", m_idpath_tmp );


    m_gltfpath = makeIdPathPath("ok.gltf") ;
    m_res->addPath("gltfpath", m_gltfpath ); 

    m_cachemetapath = makeIdPathPath("cachemeta.json");  
    m_res->addPath("cachemetapath", m_cachemetapath ); 


/**
Legacy mode equivalents for resource dirs:

srcevtbase 
    directory within opticksdata with the gensteps ?

evtbase
    user tmp directory for outputting events 

**/


}





std::string BOpticksResource::getPropertyLibDir(const char* name) const 
{
    return BFile::FormPath( m_idpath, name ) ;
}




const char* BOpticksResource::getSrcPath() const { return m_srcpath ; }
const char* BOpticksResource::getSrcDigest() const { return m_srcdigest ; }
const char* BOpticksResource::getDAEPath() const { return m_daepath ; }
const char* BOpticksResource::getGDMLPath() const { return m_gdmlpath ; }
const char* BOpticksResource::getSrcGDMLPath() const { return m_srcgdmlpath ; } 
const char* BOpticksResource::getSrcGLTFPath() const { return m_srcgltfpath ; } 

const char* BOpticksResource::getSrcGLTFBase() const
{
    std::string base = BFile::ParentDir(m_srcgltfpath) ;
    return strdup(base.c_str()); 
}
const char* BOpticksResource::getSrcGLTFName() const
{
    std::string name = BFile::Name(m_srcgltfpath) ;
    return strdup(name.c_str()); 
}

const char* BOpticksResource::getG4CodeGenDir() const { return m_g4codegendir ; }
const char* BOpticksResource::getCacheMetaPath() const { return m_cachemetapath ; }
const char* BOpticksResource::getPrimariesPath() const { return m_primariespath ; } 
//const char* BOpticksResource::getDirectGenstepPath() const { return m_directgensteppath ; } 
//const char* BOpticksResource::getDirectPhotonsPath() const { return m_directphotonspath ; } 
const char* BOpticksResource::getGLTFPath() const { return m_gltfpath ; } 
const char* BOpticksResource::getMetaPath() const { return m_metapath ; }
const char* BOpticksResource::getIdMapPath() const { return m_idmappath ; } 

//const char* BOpticksResource::getSrcEventBase() const { return m_srcevtbase ; } 
const char* BOpticksResource::getEventBase() const { return m_evtbase ; } 


bool  BOpticksResource::hasKey() const
{
    return m_key != NULL ; 
}
BOpticksKey*  BOpticksResource::getKey() const
{
    return m_key ; 
}

bool BOpticksResource::isKeySource() const 
{
    return m_key ? m_key->isKeySource() : false ; 
}



void BOpticksResource::setIdPathOverride(const char* idpath_tmp)  // used for test saves into non-standard locations
{
   m_idpath_tmp = idpath_tmp ? strdup(idpath_tmp) : NULL ;  
} 
const char* BOpticksResource::getIdPath() const 
{
    LOG(verbose) << "getIdPath"
              << " idpath_tmp " << m_idpath_tmp 
              << " idpath " << m_idpath
              ; 

    return m_idpath_tmp ? m_idpath_tmp : m_idpath  ;
}
const char* BOpticksResource::getIdFold() const 
{
    return m_idfold ;
}


void BOpticksResource::Summary(const char* msg)
{
    LOG(info) << msg << " layout " << m_layout ; 

    const char* prefix = m_install_prefix ; 

    std::cerr << "prefix   : " <<  (prefix ? prefix : "NULL" ) << std::endl ; 

    const char* name = "generate.cu.ptx" ;
    std::string ptxpath = getPTXPath(name); 
    std::cerr << "getPTXPath(" << name << ") = " << ptxpath << std::endl ;   

    std::string ptxpath_static = PTXPath(name); 
    std::cerr << "PTXPath(" << name << ") = " << ptxpath_static << std::endl ;   

    std::cerr << "debugging_idpath  " << ( m_debugging_idpath ? m_debugging_idpath : "-" )<< std::endl ; 
    std::cerr << "debugging_idfold  " << ( m_debugging_idfold ? m_debugging_idfold : "-" )<< std::endl ; 

    std::string usertmpdir = BFile::FormPath("$TMP") ; 
    std::cerr << "usertmpdir ($TMP) " <<  usertmpdir << std::endl ; 

    std::string usertmptestdir = BFile::FormPath("$TMPTEST") ; 
    std::cerr << "($TMPTEST)        " <<  usertmptestdir << std::endl ; 


    m_res->dumpPaths("dumpPaths");
    m_res->dumpDirs("dumpDirs");
    m_res->dumpNames("dumpNames");

}

const char* BOpticksResource::MakeInstallPath( const char* prefix, const char* main, const char* sub )  // static
{
    fs::path ip(prefix);   
    if(main) ip /= main ;        
    if(sub)  ip /= sub  ; 

    std::string path = ip.string();
    return strdup(path.c_str());
}

std::string BOpticksResource::BuildDir(const char* proj)
{
    return BFile::FormPath(ResolveInstallPrefix(), "build", proj );
}
std::string BOpticksResource::BuildProduct(const char* proj, const char* name)
{
    std::string builddir = BOpticksResource::BuildDir(proj);
    return BFile::FormPath(builddir.c_str(), name);
}



std::string BOpticksResource::PTXName(const char* cu_name, const char* cmake_target)
{
    std::stringstream ss ; 
    ss << cmake_target << "_generated_" << cu_name << ".ptx" ;    
    // formerly the user had to suppliy the name with ".ptx" appended
    // adjusting to cu_name for conformity with OKConf::PTXPath
    return ss.str();
}
std::string BOpticksResource::getPTXPath(const char* cu_name, const char* cmake_target)
{
    return PTXPath(cu_name, cmake_target, m_ptx_installcache_dir);
}


std::string BOpticksResource::PTXPath(const char* cu_name, const char* cmake_target, const char* prefix)
{
    fs::path ptx(prefix);   
    std::string ptxname = PTXName(cu_name, cmake_target);
    ptx /= ptxname ;
    std::string path = ptx.string(); 
    return path ;
}



