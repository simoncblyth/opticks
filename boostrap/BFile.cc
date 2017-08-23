
#include "SSys.hh"
#include "BFile.hh"
#include "BStr.hh"

#include <cstring>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>

namespace fs = boost::filesystem;


#include "PLOG.hh"


char* BFile::OPTICKS_PATH_PREFIX = NULL ;

void BFile::setOpticksPathPrefix(const char* prefix)
{
    OPTICKS_PATH_PREFIX = prefix ? strdup(prefix) : NULL ;
}
void BFile::dumpOpticksPathPrefix(const char* msg)
{
     std::cout << msg
               << " OPTICKS_PATH_PREFIX " << ( OPTICKS_PATH_PREFIX ? OPTICKS_PATH_PREFIX : "NULL" ) 
               << std::endl ;
}

void BFile::setOpticksPathPrefixFromEnv(const char* envvar)
{
    const char* prefix = SSys::getenvvar(envvar);
    if(prefix)
    {
        setOpticksPathPrefix(prefix);
        dumpOpticksPathPrefix("BFile::setOpticksPathPrefixFromEnv envvar ");
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



std::string usertmpdir(const char* base="/tmp", const char* sub="opticks")
{
    fs::path p(base) ; 

#ifdef _MSC_VER
    const char* user = SSys::getenvvar("USERNAME") ;
#else
    const char* user = SSys::getenvvar("USER") ;
#endif

    if(user) p /= user ;
    if(sub) p /= sub ; 

    std::string x = p.string() ; 
    return x ; 
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

           const char* evalue_ = SSys::getenvvar(key.c_str()) ;

           std::string evalue = evalue_ ? evalue_ : key ; 

           if(evalue.compare("TMP")==0) //  TMP envvar not defined
           {
               evalue = usertmpdir();
               LOG(trace) << "expandvar replacing TMP with " << evalue ; 
           }
           else if(evalue.compare("OPTICKS_EVENT_BASE")==0) 
           {
               evalue = usertmpdir();
               LOG(trace) << "expandvar replacing OPTICKS_EVENT_BASE  with " << evalue ; 
           }


           p /= evalue ;

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


std::string expandhome(const char* s)
{
    assert(strcmp(s,"~")==0);
#ifdef _WIN32
    const char* home = "$USERPROFILE" ;
#else
    const char* home = "$HOME" ;
#endif
    return expandvar(home);
}




std::string BFile::Stem(const char* path)
{
    fs::path fsp(path);
    std::string stem = fsp.stem().string() ;
    return stem ;
}

std::string BFile::Name(const char* path)
{
    fs::path fsp(path);
    std::string name = fsp.filename().string() ;
    return name ; 
}

std::string BFile::ParentDir(const char* path)
{
    fs::path fsp(path);
    std::string fold = fsp.parent_path().string() ;
    return fold ; 
}




bool BFile::ExistsDir(const char* path, const char* sub, const char* name)
{
    std::string p = FormPath(path, sub, name) ;
    assert(!p.empty());
    fs::path fsp(p);
    return fs::exists(fsp) && fs::is_directory(fsp) ;
}
bool BFile::ExistsNativeDir(const std::string& native)
{
    fs::path fsp(native);
    return fs::exists(fsp) && fs::is_directory(fsp) ;
}



std::time_t* BFile::LastWriteTime(const char* path, const char* sub, const char* name)
{
    std::string psn = FormPath(path, sub, name) ;

    std::time_t* t = NULL ;  
    fs::path p(psn);
    if(fs::exists(p))
    {
        t = new std::time_t( boost::filesystem::last_write_time( p ) );
    }
    return t ; 
}



std::time_t* BFile::SinceLastWriteTime(const char* path, const char* sub, const char* name)
{
    std::time_t age = NULL ;  
    std::time_t* lwt = BFile::LastWriteTime(path, sub, name);
    if(lwt)
    {
        std::time_t  now = std::time(NULL) ;
        age = (now - *lwt);
    }
    return new std::time_t(age) ; 
} 









bool BFile::ExistsFile(const char* path, const char* sub, const char* name)
{
    std::string p = FormPath(path, sub, name) ;

    if(p.empty())
    {
        LOG(error) << "BFile::ExistsFile BAD PATH"
                     << " path " << ( path ? path : "NULL" )
                     << " sub " << ( sub ? sub : "NULL" )
                     << " name " << ( name ? name : "NULL" )
                     ;
         return false ;  
    } 

    //assert(!p.empty());
    fs::path fsp(p);
    return fs::exists(fsp) && fs::is_regular_file(fsp) ;
}
bool BFile::ExistsNativeFile(const std::string& native)
{
    fs::path fsp(native);
    return fs::exists(fsp) && fs::is_regular_file(fsp) ;
}



std::string BFile::FindFile(const char* dirlist, const char* sub, const char* name, const char* dirlist_delim)
{
    std::vector<std::string> dirs ; 
    boost::split(dirs,dirlist,boost::is_any_of(dirlist_delim));

    LOG(info) << "BFile::FindFile"
              << " dirlist " << dirlist 
              << " sub " << sub
              << " name " << name
              << " dirlist_delim " << dirlist_delim
              << " elems " << dirs.size()
              ;

    std::string path ; 
    for(unsigned int i=0 ; i < dirs.size() ; i++)
    {
        std::string candidate = BFile::FormPath(dirs[i].c_str(), sub, name );  
        if(BFile::ExistsNativeFile(candidate))
        {
            path = candidate ;
            break ;  
        }
    }
    return path ; 
}



std::string BFile::FormPath(const char* path, const char* sub, const char* name)
{

   bool prepare = false ; 
   LOG(trace) << "BFile::FormPath"
              << " path " << path 
              << " sub " << sub
              << " name " << name
              << " prepare " << prepare
              ;   



   std::string empty ; 
   if(!path)
   {
       LOG(debug) << "BFile::FormPath return empty "
                  << " path " << ( path ? path : "NULL" )
                  << " sub " << ( sub ? sub : "NULL" )
                  << " name " << ( name ? name : "NULL" )
                  ;
       return empty ; 
   }


   if(strlen(path)<1)
   {
       LOG(debug) << "BFile::FormPath return empty "
                  << " strlen(path) " << strlen(path)
                  << " path " << ( path ? path : "NULL" )
                  << " sub " << ( sub ? sub : "NULL" )
                  << " name " << ( name ? name : "NULL" )
                  ;
       return empty ; 
   }



#ifdef MSC_VER_
    // hmm this is getting called on every path, so restrict to windows where its needed
   if(!OPTICKS_PATH_PREFIX)
   { 
      setOpticksPathPrefixFromEnv();
   } 
#endif


   fs::path p ; 

   bool dbg(false);


   std::string xpath ; 

   if(path[0] == '$')
   {
      xpath.assign(expandvar(path));
   } 
   else if(path[0] == '~')
   {
      xpath.assign(expandhome(path));
   }
   else if(OPTICKS_PATH_PREFIX)
   { 
      //  eg windows prefix C:\msys64
      if(strlen(path) > 1 && path[1] == ':') 
      { 
          if(dbg) std::cerr << "BFile::FormPath path is already prefixed " << path << std::endl ; 
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

   if(prepare)
   {
       preparePath(preferred.c_str(), true);
   }


   return preferred ;
}




void BFile::RemoveDir(const char* path, const char* sub, const char* name)
{
    std::string p = FormPath(path, sub, name) ;
    assert(!p.empty());
    fs::path dir(p);
    bool exists = fs::exists(dir) ;

    if(!exists)
    {
         std::cout << "BFile::RemoveDir"
                   << " path does not exist " << p 
                   << std::endl 
                  ; 
    }
    else
    {
         LOG(debug) << "BFile::RemoveDir"
                   << " deleting path " << p 
                  ; 
        unsigned long long nrm = fs::remove_all(dir);
        LOG(debug) << "BFile::RemoveDir removed " << nrm ; 
    }
}



std::string BFile::CreateDir(const char* base, const char* asub, const char* bsub)
{

#ifdef DEBUG    
    LOG(debug) << "BFile::CreateDir"
              << " base " << ( base ? base : "NULL" ) 
              << " asub " << ( asub ? asub : "NULL" ) 
              << " bsub " << ( bsub ? bsub : "NULL" ) 
              ;
#endif

    std::string ppath = FormPath(base, asub, bsub) ;
    assert(!ppath.empty());

    fs::path dir(ppath);

    bool exists = fs::exists(dir) ;

#ifdef DEBUG    
    LOG(debug) << "BFile::CreateDir"
              << " ppath" << ppath
              << " exists " << exists 
              ;
#endif

    if(!exists && fs::create_directories(dir))
    {    
       LOG(debug) << "BFile::CreateDir"
                 << " created " << ppath
                 ;
    }    

    return ppath ; 
}



std::string BFile::preparePath(const char* dir_, const char* reldir_, const char* name, bool create )
{
    fs::path fpath(dir_);
    fpath /= reldir_ ;
    return preparePath(fpath.string().c_str(), name, create);
}


std::string BFile::preparePath(const char* path_, bool create )
{
    std::string dir = BFile::ParentDir(path_);
    std::string name = BFile::Name(path_);
    return preparePath(dir.c_str(), name.c_str(), create);
}


std::string BFile::preparePath(const char* dir_, const char* name, bool create )
{
    std::string dir = BFile::FormPath(dir_) ; 
    fs::path fdir(dir.c_str());
    if(!fs::exists(fdir) && create)
    {
        if (fs::create_directories(fdir))
        {
            LOG(info)<< "preparePath : created directory " << dir ;
        }
    }
    if(fs::exists(fdir) && fs::is_directory(fdir))
    {
        fs::path fpath(dir);
        fpath /= name ;
        return fpath.string();
    }
    else
    {
        LOG(warning)<< "preparePath : FAILED " 
                    << " dir " << dir 
                    << " dir_ " << dir_ 
                    << " name " << name ;
    }
    std::string empty ; 
    return empty ; 
}



std::string BFile::prefixShorten( const char* path, const char* prefix_)
{
    std::string prefix = BFile::FormPath(prefix_);  
    if(strncmp(path, prefix.c_str(), strlen(prefix.c_str()))==0)
        return path + strlen(prefix.c_str()) ;
    else
        return path  ;
}



std::string BFile::ChangeExt( const char* path, const char* ext)
{
    std::string dir = BFile::ParentDir(path);
    std::string name = BFile::Stem(path);
    name += ext ;   
    return FormPath(dir.c_str(), name.c_str());
}


bool BFile::pathEndsWithInt(const char* path)
{
    if(!path) return false ; 
    int fallback = -666 ; 
    std::string name = BFile::Name(path) ; 
    int check = BStr::atoi(name.c_str(), fallback);
    return check != fallback ; 
}


