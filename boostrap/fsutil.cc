
#include "fsutil.hh"
#include "dbg.hh"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


#include "BLog.hh"


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


std::string fsutil::Stem(const char* path)
{
    fs::path fsp(path);
    std::string stem = fsp.stem().string() ;
    return stem ;
}

std::string fsutil::Name(const char* path)
{
    fs::path fsp(path);
    std::string name = fsp.filename().string() ;
    return name ; 
}




bool fsutil::ExistsDir(const char* path, const char* sub, const char* name)
{
    std::string p = FormPath(path, sub, name) ;
    assert(!p.empty());
    fs::path fsp(p);
    return fs::exists(fsp) && fs::is_directory(fsp) ;
}
bool fsutil::ExistsNativeDir(const std::string& native)
{
    fs::path fsp(native);
    return fs::exists(fsp) && fs::is_directory(fsp) ;
}



bool fsutil::ExistsFile(const char* path, const char* sub, const char* name)
{
    std::string p = FormPath(path, sub, name) ;
    assert(!p.empty());
    fs::path fsp(p);
    return fs::exists(fsp) && fs::is_regular_file(fsp) ;
}
bool fsutil::ExistsNativeFile(const std::string& native)
{
    fs::path fsp(native);
    return fs::exists(fsp) && fs::is_regular_file(fsp) ;
}



std::string fsutil::FindFile(const char* dirlist, const char* sub, const char* name, const char* dirlist_delim)
{
    std::vector<std::string> dirs ; 
    boost::split(dirs,dirlist,boost::is_any_of(dirlist_delim));

    LOG(info) << "fsutil::FindFile"
              << " dirlist " << dirlist 
              << " sub " << sub
              << " name " << name
              << " dirlist_delim " << dirlist_delim
              << " elems " << dirs.size()
              ;

    std::string path ; 
    for(unsigned int i=0 ; i < dirs.size() ; i++)
    {
        std::string candidate = fsutil::FormPath(dirs[i].c_str(), sub, name );  
        if(fsutil::ExistsNativeFile(candidate))
        {
            path = candidate ;
            break ;  
        }
    }
    return path ; 
}



std::string fsutil::FormPath(const char* path, const char* sub, const char* name)
{
   std::string empty ; 
   if(!path)
   {
       LOG(debug) << "fsutil::FormPath return empty "
                  << " path " << ( path ? path : "NULL" )
                  << " sub " << ( sub ? sub : "NULL" )
                  << " name " << ( name ? name : "NULL" )
                  ;
       return empty ; 
   }


   if(strlen(path)<2)
   {
       LOG(debug) << "fsutil::FormPath return empty "
                  << " strlen(path) " << strlen(path)
                  << " path " << ( path ? path : "NULL" )
                  << " sub " << ( sub ? sub : "NULL" )
                  << " name " << ( name ? name : "NULL" )
                  ;
       return empty ; 
   }





   if(!OPTICKS_PATH_PREFIX)
       setOpticksPathPrefixFromEnv();

   fs::path p ; 

   bool dbg(false);


   std::string xpath ; 

   if(path[0] == '$')
   {
      xpath.assign(expandvar(path));
   } 
   else if(OPTICKS_PATH_PREFIX)
   { 
      //  eg windows prefix C:\msys64
      if(path[1] == ':') 
      { 
          if(dbg) std::cerr << "fsutil::FormPath path is already prefixed " << path << std::endl ; 
      } 
      else
      {
          p /= OPTICKS_PATH_PREFIX ;
      } 
   } 

   p /= xpath.empty() ? path : xpath ; 

   if(sub)   p /= sub ;    
   if(name)  p /= name ;    


   p.make_preferred();

   std::string preferred = p.string();  // platform native
   return preferred ;
}


void fsutil::CreateDir(const char* path, const char* sub)
{

#ifdef DEBUG    
    std::cerr << "fsutil::CreateDir"
              << " path " << ( path ? path : "NULL" ) 
              << " sub " << ( sub ?  sub : "NULL" ) 
              << std::endl ;
              ;
#endif

    std::string ppath = FormPath(path, sub) ;
    assert(!ppath.empty());

    fs::path dir(ppath);

    bool exists = fs::exists(dir) ;

#ifdef DEBUG    
    std::cerr << "fsutil::CreateDir"
              << " ppath " << ppath
              << " exists " << exists 
              << std::endl ;
              ;
#endif

    if(!exists && fs::create_directories(dir))
    {    
       std::cerr << "fsutil::CreateDir"
                 << " created " << ppath
                 << std::endl ;
                 ;
    }    

}




