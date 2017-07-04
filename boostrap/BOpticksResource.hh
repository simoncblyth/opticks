#pragma once

#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API  BOpticksResource {
   public:
        BOpticksResource(const char* envprefix="OPTICKS_");
        virtual ~BOpticksResource();
        virtual void Summary(const char* msg="BOpticksResource::Summary");

        static std::string BuildDir(const char* proj);
        static std::string BuildProduct(const char* proj, const char* name);
        static std::string PTXPath(const char* name, const char* target="OptiXRap");

        static const char* OpticksDataDir();
        static const char* ResourceDir();
        static const char* GenstepsDir();
        static const char* InstallCacheDir();
        static const char* PTXInstallPath();
        static const char* RNGInstallPath();
        static const char* OKCInstallPath();
   private:
        static std::string PTXPath(const char* name, const char* target, const char* prefix);
        static std::string PTXName(const char* name, const char* target);
        static const char* makeInstallPath( const char* prefix, const char* main, const char* sub );
   public:       
        const char* getInstallDir();

        const char* getOpticksDataDir();
        const char* getInstallCacheDir();
        const char* getResourceDir();
        const char* getGenstepsDir();

        const char* getRNGInstallCacheDir();
        const char* getOKCInstallCacheDir();
        const char* getPTXInstallCacheDir();

        std::string getPTXPath(const char* name, const char* target="OptiXRap");
   public:       
        const char* getDebuggingIDPATH();
        const char* getDebuggingIDFOLD();
   private:
        void init();
        void adoptInstallPrefix();
        void setTopDownDirs();
        void setDebuggingIDPATH();

        //void readG4Environment();
        //void readOpticksEnvironment();
   protected:
        const char* m_envprefix ; 
        const char* m_install_prefix ;   // from BOpticksResourceCMakeConfig header
        const char* m_opticksdata_dir ; 
        const char* m_resource_dir ; 
        const char* m_gensteps_dir ; 
        const char* m_installcache_dir ; 
        const char* m_rng_installcache_dir ; 
        const char* m_okc_installcache_dir ; 
        const char* m_ptx_installcache_dir ; 
   protected:
        const char* m_debugging_idpath ; 
        const char* m_debugging_idfold ; 
};

#include "BRAP_TAIL.hh"

