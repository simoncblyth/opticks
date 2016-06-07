
#include "fsutil.hh"
#include "dbg.hh"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <boost/regex.hpp>
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




void dump(boost::cmatch& m)
{
    std::cout << " prefix " << m.prefix() << std::endl ;  
    for(unsigned int i=0 ; i < m.size() ; i++)
    {       
       std::string sm = m[i] ; 
       std::cout << std::setw(3) << i 
                 << " sm " << sm  
                 << " first [" << m[i].first << "]"
                 << " second [" << m[i].second << "]"
                 << " matched [" << m[i].matched << "]"
                 << std::endl ;
    }
}



std::string expandvar(const char* s)
{
    fs::path p ; 

    std::string dollar("$");
    boost::regex e("(\\$)(\\w+)(.*?)"); // eg $HOME/.opticks/hello
    boost::cmatch m ; 

    if(boost::regex_match(s,m,e))  
    {
        //dump(m);  

        unsigned int size = m.size();

        if(size == 4 && dollar.compare(m[1]) == 0)
        {
           std::string key = m[2] ;  
           char* evalue = getenv(key.c_str()) ;
              
           p /= evalue ? evalue : key ; 

           std::string tail = m[3] ;  

           p /= tail ;            
        }  
    }
    else
    {
        p /= s ;
    } 


    p.make_preferred(); 

    std::string x = p.string() ; 
    return x ; 
}






std::string fsutil::FormPath(const char* path, const char* sub, const char* name)
{
   if(!OPTICKS_PATH_PREFIX)
       setOpticksPathPrefixFromEnv();

   fs::path p ; 

   if(path && path[0] == '$')
   {
      std::string xpath = expandvar(path);
      p /= xpath ;    
   } 
   else if(OPTICKS_PATH_PREFIX)
   { 
      p /= OPTICKS_PATH_PREFIX ;
      if(path)  p /= path ;    
   } 

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
        DBG("fsutil::","dir exists", ( path ? path : "NULL" ) )  ; 
    }
}




