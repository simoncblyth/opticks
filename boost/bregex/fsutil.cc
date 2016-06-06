
#include "fsutil.hh"
#include "dbg.hh"

#include <cstring>
#include <cstdlib>
#include <iostream>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


char* fsutil::OPTICKS_PATH_PREFIX = NULL ;

void fsutil::setOpticksPathPrefix(const char* prefix)
{
    OPTICKS_PATH_PREFIX = prefix ? strdup(prefix) : NULL ;
}
void fsutil::dumpOpticksPathPrefix(const char* msg)
{
     std::cout << msg
               << " OPTICKS_PATH_PREFIX " << ( OPTICKS_PATH_PREFIX ? OPTICKS_PATH_PREFIX : "NULL" ) 
               << std::endl ;
}

void fsutil::setOpticksPathPrefixFromEnv(const char* envvar)
{
    char* prefix = getenv(envvar);
    if(prefix)
    {
        setOpticksPathPrefix(prefix);
        dumpOpticksPathPrefix("fsutil::setOpticksPathPrefixFromEnv envvar ");
    }
}


std::string fsutil::FormPath(const char* path, const char* sub, const char* name)
{
   if(!OPTICKS_PATH_PREFIX)
       setOpticksPathPrefixFromEnv();

   fs::path p ; 

   if(OPTICKS_PATH_PREFIX) p /= OPTICKS_PATH_PREFIX ;

   if(path)  p /= path ;    
   if(sub)   p /= sub ;    
   if(name)  p /= name ;    


   p.make_preferred();

   std::string preferred = p.string();  // platform native
   return preferred ;
}


void fsutil::CreateDir(const char* path, const char* sub)
{
    std::string ppath = FormPath(path, sub) ;

    fs::path dir(ppath);
    if(!fs::exists(dir))
    {    
        DBG("fsutil::","dir does not exists",path)  ; 
        if (fs::create_directories(dir))
        {    
            DBG("fsutil::","created directory",path)  ; 
        }    
    }    
    else
    {
        DBG("fsutil::","dir exists",path)  ; 
    }
}




