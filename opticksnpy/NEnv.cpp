
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>

#include "SSys.hh"
#include "BFile.hh"
#include "BStr.hh"

#include "Map.hpp"
#include "NEnv.hpp"
#include "PLOG.hh"

#ifdef _MSC_VER
#else
#include <unistd.h>
extern char **environ;
#endif

NEnv* NEnv::load(const char* dir, const char* name) 
{
   NEnv* e = new NEnv ;  
   e->readFile(dir, name);
   return e ; 
}

NEnv* NEnv::load(const char* path) 
{
   NEnv* e = new NEnv ;  
   e->readFile(path);
   return e ; 
}



NEnv::NEnv(char** envp) 
   :  
    m_envp(envp), 
    m_prefix(NULL), 
    m_path(NULL), 
    m_all(NULL),
    m_selection(NULL)

{
   init();
   readEnv();
}

void NEnv::init()  
{
}

void NEnv::readEnv()
{
    m_all = new MSS ;
    if(!m_envp) return ; 

    while(*m_envp)
    {
       std::string kv = *m_envp++ ; 
       const size_t pos = kv.find("=") ;
       if(pos == std::string::npos) continue ;     
       std::string k = kv.substr(0, pos);
       std::string v = kv.substr(pos+1);

       m_all->add(k,v);  
    }
}


void NEnv::readFile(const char* dir, const char* name)
{
    m_all  = MSS::load(dir, name); 
}
void NEnv::readFile(const char* path)
{
    m_path = path ? strdup(path) : NULL ;  
    m_all  = MSS::load(path); 
}

void NEnv::save(const char* dir, const char* name)
{
    MSS* mss = m_selection ? m_selection : m_all ; 
    mss->save(dir, name);
}
void NEnv::save(const char* path)
{
    MSS* mss = m_selection ? m_selection : m_all ; 
    mss->save(path);
}





void NEnv::setPrefix(const char* prefix)
{
    delete m_selection ;
    if(prefix)
    {
        char delim = ',' ;
        m_prefix = strdup(prefix);
        m_selection = m_all->makeSelection(prefix, delim);
    }
    else
    {
        m_selection = NULL ; 
        m_prefix = NULL ; 
    }
}


void NEnv::dump(const char* msg)
{
    LOG(info) << msg << " prefix " << ( m_prefix ? m_prefix : "NULL" ) ; 
    typedef std::map<std::string, std::string> SS ; 

    MSS* mss = m_selection ? m_selection : m_all ; 
    SS m = mss->getMap();

    for(SS::iterator it=m.begin() ; it != m.end() ; it++)
    {
        std::string k = it->first ; 
        std::string v = it->second ; 

        std::cout << " k " << std::setw(30) << k 
                  << " v " <<std::setw(100) << v 
                  << std::endl ;  
    }
}



std::string NEnv::nativePath(const char* val)
{
    std::string p = val ; 

    bool is_fspath = 
                p.find("/") == 0 || 
                p.find("\\") == 0 || 
                p.find(":") == 1  ;
 

    if(!is_fspath) return val ; 

    bool is_gitbashpath = p.find("/c/") == 0 ;

    std::string bpath = is_gitbashpath ? p.substr(2) : p ;

    std::string npath = BFile::FormPath(bpath.c_str());

    LOG(trace) << "NEnv::nativePath"
               << " val " << val
               << " bpath " << bpath
               << " npath " << npath
               ;    


    return npath ; 
}


void NEnv::setEnvironment(bool overwrite, bool native)
{
   if(m_selection == NULL && m_path == NULL)
   {
       LOG(warning) << "NEnv::setEnvironment SAFETY RESTRICTION : MUST setPrefix non-NULL to effect a selection OR load from a file  " ; 
       return ;  
   } 

    typedef std::map<std::string, std::string> SS ; 

    MSS* mss = m_selection ? m_selection : m_all ; 
    SS m = mss->getMap();

    for(SS::iterator it=m.begin() ; it != m.end() ; it++)
    {
        std::string k = it->first ; 
        std::string v = it->second ; 

        std::string nv = native ? nativePath(v.c_str()) : v ; 

        int rc = SSys::setenvvar( NULL, k.c_str(), nv.c_str(), overwrite);
        assert(rc == 0 );

        LOG(info) << "NEnv::setEnvironment" 
                   << " overwrite " << overwrite 
                   << " k " << std::setw(30) << k 
                   << " v " <<std::setw(100) << v 
                   ;
    }
}



#ifdef _MSC_VER
void NEnv::dumpEnvironment(const char* msg, const char* )
{
   LOG(warning) << msg << " NOT IMPLEMENTED ON WINDOWS " ;     
}
#else
void NEnv::dumpEnvironment(const char* msg, const char* prefix)
{
    std::vector<std::string> elem ;   
    char delim = ',' ;
    BStr::split(elem, prefix, delim );
    size_t npfx = elem.size();

    // multiple selection via delimited prefix 


   LOG(info) << msg << " prefix " << ( prefix ? prefix : "NULL" ) ;
   int i = 0;
   while(environ[i]) 
   {
      std::string kv = environ[i++] ;

      bool select = false ;   

      if(npfx == 0)
      {
         select = true ;
      }
      else 
      {
         for(size_t p=0 ; p < npfx ; p++) if(kv.find(elem[p].c_str()) == 0) select = true ; 
      }

      if(select)
      {
          std::cerr << kv << std::endl ; 
      }
   }
}
#endif




