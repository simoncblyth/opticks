#pragma once

#include <cstddef>
#include <string>

#include "BRAP_API_EXPORT.hh"

template <typename A, typename B> class Map ; 

class BRAP_API BEnv {
   public:
      typedef Map<std::string, std::string> MSS ; 
      static BEnv* load(const char* dir, const char* name);
      static BEnv* load(const char* path);
      static void dumpEnvironment(const char* msg="BEnv::dumpEnvironment", const char* prefix="G4,OPTICKS,DAE,IDPATH");
   public:
      BEnv(char** envp=NULL);
      void save(const char* dir, const char* name);
      void save(const char* path);
      void dump(const char* msg="BEnv::dump");
      void setPrefix(const char* prefix);
      void setEnvironment(bool overwrite=true, bool native=true);

   private:
      void init();
      void readEnv();
      void readFile(const char* dir, const char* name);
      void readFile(const char* path);

      std::string nativePath(const char* val);

   private:
      char**      m_envp ;
      const char* m_prefix ;
      const char* m_path ;
      MSS*        m_all ;        
      MSS*        m_selection ;        

};



