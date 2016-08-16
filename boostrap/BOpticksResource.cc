#include <cassert>
#include <cstring>
#include <iostream>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "SSys.hh"

#include "BFile.hh"
#include "BOpticksResource.hh"

// CMake generated defines from binary_dir/inc
#include "BOpticksResourceCMakeConfig.hh"  

#include "PLOG.hh"


BOpticksResource::BOpticksResource(const char* envprefix)
   :
     m_envprefix(strdup(envprefix)),
     m_install_prefix(NULL),
     m_opticksdata_dir(NULL),
     m_resource_dir(NULL),
     m_gensteps_dir(NULL),
     m_installcache_dir(NULL),
     m_rng_installcache_dir(NULL),
     m_okc_installcache_dir(NULL),
     m_ptx_installcache_dir(NULL)
{
    init();
}



BOpticksResource::~BOpticksResource()
{
}

void BOpticksResource::init()
{
    adoptInstallPrefix() ;
    setTopDownDirs();
}

void BOpticksResource::adoptInstallPrefix()
{
   m_install_prefix = strdup(OPTICKS_INSTALL_PREFIX) ; 

   const char* key = "INSTALL_PREFIX" ; 

   int rc = SSys::setenvvar(m_envprefix, key, m_install_prefix, true );  

   LOG(trace) << "OpticksResource::adoptInstallPrefix " 
              << " install_prefix " << m_install_prefix  
              << " envprefix " << m_envprefix  
              << " key " << key 
              << " rc " << rc
              ;   
 
   assert(rc==0); 

   // The CMAKE_INSTALL_PREFIX from opticks-;opticks-cmake 
   // is set to the result of the opticks-prefix bash function 
   // at configure time.
   // This is recorded into a config file by okc-/CMakeLists.txt 
   // and gets compiled into the OpticksCore library.
   //  
   // Canonically it is :  /usr/local/opticks 
}

void BOpticksResource::setTopDownDirs()
{ 
    m_opticksdata_dir      = OpticksDataDir() ;      // eg /usr/local/opticks/opticksdata
    m_resource_dir         = ResourceDir() ;  // eg /usr/local/opticks/opticksdata/resource
    m_gensteps_dir         = GenstepsDir() ;  // eg /usr/local/opticks/opticksdata/gensteps
    m_installcache_dir     = InstallCacheDir() ;      // eg  /usr/local/opticks/installcache

    m_rng_installcache_dir = RNGInstallPath() ;  // eg  /usr/local/opticks/installcache/RNG
    m_okc_installcache_dir = OKCInstallPath() ;  // eg  /usr/local/opticks/installcache/OKC
    m_ptx_installcache_dir = PTXInstallPath() ;  // eg  /usr/local/opticks/installcache/PTX
}


const char* BOpticksResource::InstallCacheDir(){return makeInstallPath(OPTICKS_INSTALL_PREFIX, "installcache",  NULL); }
const char* BOpticksResource::OpticksDataDir(){ return makeInstallPath(OPTICKS_INSTALL_PREFIX, "opticksdata",  NULL); }
const char* BOpticksResource::ResourceDir(){    return makeInstallPath(OPTICKS_INSTALL_PREFIX, "opticksdata", "resource" ); }
const char* BOpticksResource::GenstepsDir(){    return makeInstallPath(OPTICKS_INSTALL_PREFIX, "opticksdata", "gensteps" ); }

const char* BOpticksResource::PTXInstallPath(){ return makeInstallPath(OPTICKS_INSTALL_PREFIX, "installcache", "PTX"); }
const char* BOpticksResource::RNGInstallPath(){ return makeInstallPath(OPTICKS_INSTALL_PREFIX, "installcache", "RNG"); }
const char* BOpticksResource::OKCInstallPath(){ return makeInstallPath(OPTICKS_INSTALL_PREFIX, "installcache", "OKC"); }


std::string BOpticksResource::PTXPath(const char* name, const char* target)
{
    const char* ptx_installcache_dir = PTXInstallPath();
    return PTXPath(name, target, ptx_installcache_dir);
}





const char* BOpticksResource::getInstallDir() {         return m_install_prefix ; }   
const char* BOpticksResource::getOpticksDataDir() {     return m_opticksdata_dir ; }   
const char* BOpticksResource::getResourceDir() {        return m_resource_dir ; } 

const char* BOpticksResource::getInstallCacheDir() {    return m_installcache_dir ; } 
const char* BOpticksResource::getRNGInstallCacheDir() { return m_rng_installcache_dir ; } 
const char* BOpticksResource::getOKCInstallCacheDir() { return m_okc_installcache_dir ; } 
const char* BOpticksResource::getPTXInstallCacheDir() { return m_ptx_installcache_dir ; } 



void BOpticksResource::Summary(const char* msg)
{
    std::cerr << msg << std::endl ; 
    const char* prefix = m_install_prefix ; 

    std::cerr << "prefix   : " <<  (prefix ? prefix : "NULL" ) << std::endl ; 
    std::cerr << "envprefix: " <<  (m_envprefix?m_envprefix:"NULL") << std::endl; 

    std::cerr << "opticksdata_dir      " << m_opticksdata_dir     << std::endl ; 
    std::cerr << "resource_dir         " << m_resource_dir     << std::endl ; 
    std::cerr << "gensteps_dir         " << m_gensteps_dir     << std::endl ; 
    std::cerr << "installcache_dir     " << m_installcache_dir << std::endl ; 
    std::cerr << "rng_installcache_dir " << m_rng_installcache_dir << std::endl ; 
    std::cerr << "okc_installcache_dir " << m_okc_installcache_dir << std::endl ; 
    std::cerr << "ptx_installcache_dir " << m_ptx_installcache_dir << std::endl ; 


    const char* name = "generate.cu.ptx" ;
    std::string ptxpath = getPTXPath(name); 
    std::cerr << "getPTXPath(" << name << ") = " << ptxpath << std::endl ;   

    std::string ptxpath_static = PTXPath(name); 
    std::cerr << "PTXPath(" << name << ") = " << ptxpath_static << std::endl ;   



}


const char* BOpticksResource::makeInstallPath( const char* prefix, const char* main, const char* sub )
{
    fs::path ip(prefix);   
    if(main) ip /= main ;        
    if(sub)  ip /= sub  ; 

    std::string path = ip.string();
    return strdup(path.c_str());
}

std::string BOpticksResource::BuildDir(const char* proj)
{
    return BFile::FormPath(OPTICKS_INSTALL_PREFIX, "build", proj );
}
std::string BOpticksResource::BuildProduct(const char* proj, const char* name)
{
    std::string builddir = BOpticksResource::BuildDir(proj);
    return BFile::FormPath(builddir.c_str(), name);
}



std::string BOpticksResource::PTXName(const char* name, const char* target)
{
    std::stringstream ss ; 
    ss << target << "_generated_" << name ; 
    return ss.str();
}
std::string BOpticksResource::getPTXPath(const char* name, const char* target)
{
    return PTXPath(name, target, m_ptx_installcache_dir);
}





std::string BOpticksResource::PTXPath(const char* name, const char* target, const char* prefix)
{
    fs::path ptx(prefix);   
    std::string ptxname = PTXName(name, target);
    ptx /= ptxname ;
    std::string path = ptx.string(); 
    return path ;
}


