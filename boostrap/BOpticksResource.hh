#pragma once

#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API  BOpticksResource {
   public:
        BOpticksResource(const char* envprefix="OPTICKS_", unsigned version=0);
        virtual ~BOpticksResource();
        virtual void Summary(const char* msg="BOpticksResource::Summary");

        static std::string BuildDir(const char* proj);
        static std::string BuildProduct(const char* proj, const char* name);
        static std::string PTXPath(const char* name, const char* target="OptiXRap");

        static const char* GeoDirName(const char* srcpath);   // ParentName of dae/gdml srcpath eg DayaBay_VGDX_20140414-1300
        static const char* GeoFileName(const char* srcpath);   // FileName   of dae/gdml srcpath eg g4_00.dae

        static const char* OpticksDataDir();
        static const char* GeoCacheDir();
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
        const char* getGeoCacheDir();
        const char* getInstallCacheDir();
        const char* getResourceDir();
        const char* getGenstepsDir();

        const char* getRNGInstallCacheDir();
        const char* getOKCInstallCacheDir();
        const char* getPTXInstallCacheDir();

        const char* getDebuggingTreedir(int argc, char** argv);
        std::string getPTXPath(const char* name, const char* target="OptiXRap");
   public:       
        const char* getDebuggingIDPATH();
        const char* getDebuggingIDFOLD();
   public:       
       const char* getIdPath();
       const char* getIdFold();  // parent directory of idpath containing g4_00.dae
       const char* getIdBase();  // parent directory of idfold, typically the "export" folder
       void setIdPathOverride(const char* idpath_tmp=NULL);  // used for test saves into non-standard locations
   private:
        void init();
        void adoptInstallPrefix();
        void setTopDownDirs();
        void setDebuggingIDPATH(); 
   protected:
        void setSrcPathDigest(const char* srcpath, const char* srcdigest);
   protected:
        const char* m_envprefix ; 
        unsigned    m_version ; 
        const char* m_install_prefix ;   // from BOpticksResourceCMakeConfig header
        const char* m_opticksdata_dir ; 
        const char* m_geocache_dir ; 
        const char* m_resource_dir ; 
        const char* m_gensteps_dir ; 
        const char* m_installcache_dir ; 
        const char* m_rng_installcache_dir ; 
        const char* m_okc_installcache_dir ; 
        const char* m_ptx_installcache_dir ; 
   protected:
        const char* m_srcpath ; 
        const char* m_srcdigest ; 
        const char* m_idfold ; 
        const char* m_idname ; 
        const char* m_idbase ; 
        const char* m_idpath ; 
        const char* m_idpath_tmp ; 
   protected:
        const char* m_debugging_idpath ; 
        const char* m_debugging_idfold ; 
};

#include "BRAP_TAIL.hh"

