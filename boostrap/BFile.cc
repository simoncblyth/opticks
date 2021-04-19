/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

 
#include "SSys.hh"
#include "BFile.hh"
#include "BStr.hh"
#include "BResource.hh"

#include <csignal>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>

namespace fs = boost::filesystem;


#include "PLOG.hh"


const plog::Severity BFile::LEVEL = PLOG::EnvLevel("BFile", "DEBUG"); 


char* BFile::OPTICKS_PATH_PREFIX = NULL ;

/**
Formerly uses the below prefix on a windows build:

case $(uname -s) in 
MINGW*) export OPTICKS_PATH_PREFIX="C:\\msys64" ;;
esac
**/

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



/**
BFile::UserTmpDir
-----------------

When TMP envvar is defined return that, otherwise 
defaults to /tmp/<username>/opticks

**/

std::string BFile::UserTmpDir()
{
    const char* tmp = SSys::getenvvar(OPTICKS_USER_TMP_KEY) ;
    if(tmp) return tmp ; 
    return usertmpdir("/tmp", "opticks", NULL ) ; 
}

/**
BFile::UserTmpPath
--------------------

Returns a string suitable for temporary paths including 
random hex higits to avoid clashes.

**/

std::string BFile::UserTmpPath_(const char* pfx_)
{
    fs::path tmpl = boost::filesystem::path(UserTmpDir()) ; 

    std::stringstream ss ; 
    ss << pfx_ << "_%%%%-%%%%-%%%%-%%%%" ; 
    std::string pfx = ss.str(); 
    tmpl /= pfx ; 

    fs::path p = boost::filesystem::unique_path(tmpl);
    const std::string tmp = p.native();
    return tmp ; 
}

const char* BFile::UserTmpPath(const char* pfx)
{
    std::string path = UserTmpPath_(pfx);
    return strdup(path.c_str()); 
}



std::string BFile::usertmpdir(const char* base, const char* sub, const char* rel )
{
    fs::path p(base) ; 

    const char* user = SSys::username(); 

    if(user) p /= user ;
    if(sub) p /= sub ; 
    if(rel) p /= rel ; 

    std::string x = p.string() ; 
    return x ; 
}






/**
BFile::expandvar
------------------

Expands dollar tokens in strings according 
to envvars or internal BResource vars   

**/

std::string BFile::expandvar(const char* s)
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

           std::string evalue = ResolveKey(key.c_str()); 

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








const std::vector<std::string> BFile::envvars = { 
   "TMP", 
   "HOME",
   "OPTICKS_INSTALL_PREFIX",    // needed for OpticksFlags to find the enum header
   "OPTICKS_PREFIX", 
   "OPTICKS_GEOCACHE_PREFIX",  
   "OPTICKS_EVENT_BASE",
   "OPTICKS_HOME",              // needed by OInterpolationTest to find a python script
   "OPTICKS_USER_HOME",         // overrides HOME within Opticks
   "IDPATH"
} ; 

/**
BFile::IsAllowedEnvvar
-----------------------

Envvars listed as allowed will be resolved from strings such as $OPTICKS_GEOCACHE_PREFIX 
by BFile path handling  

**/

bool BFile::IsAllowedEnvvar(const char* key_)
{
    std::string key(key_); 
    return std::find( envvars.begin(), envvars.end(), key ) != envvars.end() ;  
}


/**
BFile::ResolveKey
---------------------

NB only allowed envvars are replaced, for clarity and control 

The HOME key is special cased, it is replaced by OPTICKS_USER_HOME 
if that envvar is defined.  This is convenient to internally 
change the HOME directory for Opticks executables on clusters 
where HOME is prone to permissions problems, eg from expiring AFS tokens.

::

   OPTICKS_USER_HOME=/hpcfs/juno/junogpu/blyth BFileTest


**/

const char* BFile::OPTICKS_USER_HOME_KEY = "OPTICKS_USER_HOME" ; 
const char* BFile::OPTICKS_USER_TMP_KEY  = "TMP" ; 




std::string BFile::ResolveKey( const char* _key )
{
    bool is_home = strcmp(_key, "HOME") == 0  ; 
    bool has_home_override = SSys::getenvvar(OPTICKS_USER_HOME_KEY) != NULL ; 
    const char* key = is_home && has_home_override ? OPTICKS_USER_HOME_KEY  : _key ;   

    const char* envvar = SSys::getenvvar(key) ;
    std::string evalue ; 

    if( IsAllowedEnvvar(key) )
    {
        if( envvar != NULL )  
        {
            evalue = envvar ; 
            LOG(verbose) << "replacing allowed envvar token " << key << " with value of the envvar " << evalue ; 
        }   
        else
        {
            evalue = UserTmpDir();
            LOG(LEVEL) << "replacing allowed envvar token " << key << " with default value " << evalue << " as envvar not defined " ; 
        }
    }
    else if(strcmp(key,"KEYDIR")==0 ) 
    {
        const char* idpath = BResource::GetDir("idpath") ; 
        assert( idpath ); 
        evalue = idpath ;  
        LOG(error) << "replacing $IDPATH with " << evalue ; 
    }
    else if(strcmp(key,"PREFIX")==0 ) 
    {
        const char* prefix = BResource::GetDir("install_prefix") ; 
        assert( prefix ); 
        evalue = prefix ;  
        LOG(LEVEL) << "replacing $PREFIX with " << evalue ; 
    }
    else if(strcmp(key,"DATADIR")==0 ) 
    {
        const char* datadir = BResource::GetDir("opticksdata_dir") ; 
        assert( datadir ); 
        evalue = datadir ;  
        LOG(error) << "replacing $DATADIR with " << evalue ; 
    }
    else if(strcmp(key,"OPTICKS_EVENT_BASE")==0) 
    {
        const char* evtbase = BResource::GetDir("evtbase") ;   // <-- from BOpticksResource
        if( evtbase != NULL )
        {
            evalue = evtbase ; 
        }
        else if( envvar )   // if no internal BResource set allow use of external envvar
        {
            evalue = envvar ; 
        }
        else
        {
            evalue = UserTmpDir();
        } 
        LOG(LEVEL) 
           << "replacing $OPTICKS_EVENT_BASE  " 
           << " evalue " << evalue
           << " evtbase " << evtbase
           ; 

       if(evalue.empty()) LOG(fatal) << "Failed to resolve OPTICKS_EVENT_BASE " ; 
       assert(!evalue.empty());  

    }
    else if(strcmp(key,"INSTALLCACHE_DIR")==0) 
    {
        const char* installcache_dir = BResource::GetDir("installcache_dir") ; 
        evalue = installcache_dir ; 
    }
    else
    {
        evalue = key ; 
    }  
    return evalue ; 
}




std::string BFile::CWD()
{
    fs::path p = fs::current_path();  
    return p.string();
}

const char* BFile::CurrentDirectoryName()
{
    fs::path p = fs::current_path();  
    fs::path s = p.stem(); 

    std::string cdn = s.string(); 
    return strdup(cdn.c_str()); 
}


std::string BFile::Absolute(const char* rela, const char* relb, const char* relc)
{
    fs::path r(rela);
    if(relb) r /= relb ; 
    if(relc) r /= relc ; 
 
    const fs::path base = fs::current_path() ;  
    fs::path a = fs::absolute(r, base);    
    return a.string();
}

std::string BFile::AbsoluteCanonical(const char* relpath)
{
    fs::path r(relpath); 
    const fs::path base = fs::current_path() ;  
    fs::path a = fs::absolute(r, base);    

    try 
    {
        a = fs::canonical(a); 
    }
    catch( fs::filesystem_error& e  )
    {
        LOG(error) << e.what() ; 
    }

    return a.string();
}



std::string BFile::expandhome(const char* s)
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

std::string BFile::ParentParentDir(const char* path)
{
    fs::path fsp(path);
    std::string fold = fsp.parent_path().parent_path().string() ;
    return fold ; 
}


std::string BFile::ParentName(const char* path)
{
    if(!path) return "" ; 
    fs::path fsp(path);
    std::string fold = fsp.parent_path().string() ;
    std::string name = BFile::Name(fold.c_str()); 
    return name ; 
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
    std::time_t age(0) ;  
    std::time_t* lwt = BFile::LastWriteTime(path, sub, name);
    if(lwt)
    {
        std::time_t  now = std::time(NULL) ;
        age = (now - *lwt);
    }
    return new std::time_t(age) ; 
} 



bool BFile::LooksLikePath(const char* path)
{
    if(!path) return false ;
    if(strlen(path) < 2) return false ; 
    return path[0] == '/' || path[0] == '$' ; 
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


std::string BFile::FormPath(const std::vector<std::string>& elem, unsigned i0, unsigned i1)
{
   assert( i0 < elem.size() );
   assert( i1 <= elem.size() );

   fs::path p ; 
   for(unsigned i=i0 ; i < i1 ; i++) 
   {
       if(!elem[i].empty()) p /= elem[i]  ; 
   }
   p.make_preferred();
   std::string preferred = p.string();  // platform native
   return preferred ;
}


std::string BFile::FormRelativePath(const char* a, const char* b, const char* c, const char* d, const char* e, const char* f)
{
    fs::path pp ; 

    if(a) pp /= a ; 
    if(b) pp /= b ; 
    if(c) pp /= c ; 
    if(d) pp /= d ; 
    if(e) pp /= e ; 
    if(f) pp /= f ; 

    return pp.string();  // platform native
}


std::string BFile::FormPath(const char* path, const char* sub, const char* name, const char* extra1, const char* extra2)
{

   bool prepare = false ; 
   LOG(LEVEL) 
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
       LOG(LEVEL) << "expandvar for dollar " << path ; 
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
   if(extra1)  p /= extra1 ;    
   if(extra2)  p /= extra2 ;    


   p.make_preferred();

   std::string preferred = p.string();  // platform native

   if(prepare)
   {
       preparePath(preferred.c_str(), true);
   }


   std::string check ; 
   //check="/tmp/blyth/opticks/evt/dayabay/machinery/1/gs.npy" ; 

   if(!check.empty() && check.compare(preferred) == 0) 
   {
       LOG(fatal) << "forming a checked path " << preferred ; 
       assert(0); 
   }    


   return preferred ;
}

void BFile::CreateFile(const char* path, const char* sub, const char* name)
{
    std::string p = FormPath(path, sub, name) ;
    assert(!p.empty());

    if( p.size() < 2 ) 
    {
        LOG(error) << " path sanity check fail " << p  ; 
        return ; 
    }

    preparePath( p.c_str() ) ; 
    
    fs::path fsp(p);
    if( fs::exists(fsp) ) 
    {
        LOG(error) << " path exists already " << p ; 
        return ; 
    }

    std::ofstream ofs(p.c_str()) ;
    //fs::ofstream ofs{fsp};
 
}

void BFile::RemoveFile(const char* path, const char* sub, const char* name)
{
    std::string p = FormPath(path, sub, name) ;
    assert(!p.empty());

    if( p.size() < 2 ) 
    {
        LOG(error) << " path sanity check fail " << p  ; 
        return ; 
    }

    fs::path fsp(p);

    if( !fs::exists(fsp) ) 
    {
        LOG(error) << " path does not exist " << p ; 
        return ; 
    }
    if( !fs::is_regular_file(fsp) )
    {  
        LOG(error) << " path is not a regular file " << p  ; 
        return ; 
    }

    bool ret = fs::remove(fsp) ; 
    assert( ret == true );  
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
    if(reldir_) fpath /= reldir_ ;
    std::string p = preparePath(fpath.string().c_str(), name, create);
    LOG(LEVEL) << " dir_ " << dir_ << " reldir_ " << reldir_ << " name " << name << " create " << create << " -> " << p ; 
    return p ; 
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
            LOG(info)<< "created directory " << dir ;
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
        LOG(error)
            << " FAILED " 
            << " dir " << dir 
            << " dir_ " << dir_ 
            << " name " << name 
            ;
         //std::raise(SIGINT); 
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


void BFile::SplitPath(std::vector<std::string>& elem, const char* path )
{
    fs::path fpath(path);   
    for (fs::path::iterator it=fpath.begin() ; it != fpath.end(); ++it) 
    {
        std::string s = it->filename().string() ; 
        elem.push_back(s);
    }
}


std::size_t BFile::FileSize( const char* path_ )
{
    std::string path = BFile::FormPath(path_ ); 
    const char* xpath = path.c_str() ;     
    fs::path fsp(xpath);
    bool exists = fs::exists(fsp) && fs::is_regular_file(fsp) ;
    return exists ? fs::file_size(fsp) : 0 ; 
}


const char* BFile::ResolveScript(const char* script_name, const char* fallback_directory)
{
    std::string path = SSys::Which(script_name); 
    if(path.empty() && fallback_directory != NULL)
    {
        path = FormPath(fallback_directory, script_name);   
    }
    return path.empty() ? NULL : strdup(path.c_str()) ; 
}



std::string BFile::MakeIndex(int index, unsigned width) // static 
{
    std::stringstream ss ;
    ss  
        << std::setw(width) 
        << std::setfill('0')
        << index 
        ;   
    return ss.str();
}

/**
BFile::MakeName
-----------------

Using index -1 returns a format string of form "%0.5d" rather than  the index 

**/

std::string BFile::MakeName(int index, unsigned width, const char* prefix, const char* ext ) // static 
{
    std::stringstream ff ;
    ff << "%0." << width << "d" ; 
    std::string fmt = ff.str(); 
 
    std::stringstream ss ;
    ss  
       << prefix 
       << ( index == -1 ? fmt : MakeIndex(index, width) )
       << ext 
       ;   

    return ss.str();
}



