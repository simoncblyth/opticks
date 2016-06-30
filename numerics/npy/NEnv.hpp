#pragma once

#include <cstddef>
#include <string>

#include "NPY_API_EXPORT.hh"

template <typename A, typename B> class Map ; 

class NPY_API NEnv {
   public:
      typedef Map<std::string, std::string> MSS ; 
      static NEnv* load(const char* dir, const char* name);
      static NEnv* load(const char* path);
      static void dumpEnvironment(const char* msg="NEnv::dumpEnvironment", const char* prefix="G4,OPTICKS,DAE,IDPATH");
   public:
      NEnv(char** envp=NULL);
      void save(const char* dir, const char* name);
      void save(const char* path);
      void dump(const char* msg="NEnv::dump");
      void setPrefix(const char* prefix);
      void setEnvironment(bool overwrite=true);
   private:
      void init();
      void readEnv();
      void readFile(const char* dir, const char* name);
      void readFile(const char* path);
   private:
      char**      m_envp ;
      const char* m_prefix ;
      const char* m_path ;
      MSS*        m_all ;        
      MSS*        m_selection ;        

};



