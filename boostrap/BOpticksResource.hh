#pragma once

#include <vector>
#include <map>
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API  BOpticksResource {
       
    private:
       static const char* G4ENV_RELPATH ; 
       static const char* OKDATA_RELPATH ;
    protected:
       static const char* InstallPathOKDATA() ;
       static const char* InstallPathG4ENV() ;
       static const char* InstallPath(const char* relpath) ;
   public:
       static const char* MakeSrcPath(const char* srcpath, const char* ext) ;
       static const char* IdMapSrcPath(); // requires OPTICKS_SRCPATH  envvar
   public:
        BOpticksResource(const char* envprefix="OPTICKS_");
        virtual ~BOpticksResource();
        virtual void Summary(const char* msg="BOpticksResource::Summary");

        static std::string BuildDir(const char* proj);
        static std::string BuildProduct(const char* proj, const char* name);
        static std::string PTXPath(const char* name, const char* target="OptiXRap");

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
        const char* getInstallPrefix();
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
       std::string getGeoCachePath(const char* rela, const char* relb=NULL, const char* relc=NULL, const char* reld=NULL) const ;
       std::string getPropertyLibDir(const char* name) const ;
       std::string getInstallPath(const char* relpath) const ;
       const char* getIdPath();
       const char* getIdFold();  // parent directory of idpath containing g4_00.dae
       void setIdPathOverride(const char* idpath_tmp=NULL);  // used for test saves into non-standard locations

    public:
       const char* getDAEPath() const ;
       const char* getGDMLPath() const ;
       const char* getGLTFPath() const ;
       const char* getMetaPath() const ;
   public:       
        void addDir( const char* label, const char* dir);
        void addPath( const char* label, const char* path);
        void addName( const char* label, const char* name);

       // resource existance dumping 
       void dumpPaths(const char* msg) const ;
       void dumpDirs(const char* msg) const ;
       void dumpNames(const char* msg) const ;

       const char* getPath(const char* label) const  ;
   private:
        void init();
        void adoptInstallPrefix();
        void setTopDownDirs();
        void setDebuggingIDPATH(); 
   protected:
        friend struct BOpticksResourceTest ; 
        void setSrcPathDigest(const char* srcpath, const char* srcdigest);
   protected:
        const char* m_envprefix ; 
        int         m_layout ; 
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
        const char* m_srcfold ; 
        const char* m_srcbase ; 
        const char* m_srcdigest ; 
        const char* m_idfold ; 
        const char* m_idfile ; 
        const char* m_idname ; 
        const char* m_idpath ; 
        const char* m_idpath_tmp ; 
   protected:
        const char* m_debugging_idpath ; 
        const char* m_debugging_idfold ; 
   protected:
       const char* m_daepath ;
       const char* m_gdmlpath ;
       const char* m_gltfpath ;
       const char* m_metapath ;
       const char* m_idmappath ;
   protected:
        std::vector<std::pair<std::string, std::string> >  m_paths  ; 
        std::vector<std::pair<std::string, std::string> >  m_dirs  ; 
        std::vector<std::pair<std::string, std::string> >  m_names  ; 

};

#include "BRAP_TAIL.hh"

