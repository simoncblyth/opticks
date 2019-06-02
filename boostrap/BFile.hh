#pragma once

#include <ctime>
#include <vector>
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BFile {
    public:
       static std::size_t FileSize( const char* path ) ; 
       static void SplitPath(std::vector<std::string>& elem, const char* path ) ; 
       static bool LooksLikePath(const char* path) ; 
   
       static std::string CWD();
       static std::string Absolute(const char* rela, const char* relb=NULL, const char* relc=NULL);   
       static std::string AbsoluteCanonical(const char* relpath);  // paths must exist to be canonicalized, otherwise the .. dots remain 

       static std::string expandvar(const char* s);
       static std::string expandhome(const char* s);
       static std::string usertmpdir(const char* base, const char* sub, const char* rel );

       static bool IsAllowedEnvvar(const char* key);
       static std::string ResolveKey( const char* key );

       static std::string FormRelativePath(const char* a, const char* b=NULL, const char* c=NULL, const char* d=NULL, const char* e=NULL, const char* f=NULL);
       static std::string FormPath(const std::vector<std::string>& elem, unsigned i0, unsigned i1 ); 
       static std::string FormPath(const char* path, const char* sub=NULL, const char* name=NULL, const char* extra1=NULL, const char* extra2=NULL );
       static std::string FindFile(const char* dirlist, const char* sub, const char* name=NULL, const char* dirlist_delim=";");
       static std::string Stem(const char* path);
       static std::string Name(const char* path);
       static std::string ParentDir(const char* path);
       static std::string ParentName(const char* path);

       static std::string ChangeExt(const char* path, const char* ext=".json");
       static bool pathEndsWithInt(const char* path); 

       static bool ExistsNativeFile(const std::string& native);
       static bool ExistsNativeDir(const std::string& native);
       static bool ExistsFile(const char* path, const char* sub=NULL, const char* name=NULL);
       static bool ExistsDir(const char* path, const char* sub=NULL, const char* name=NULL);
       static void RemoveDir(const char* path, const char* sub=NULL, const char* name=NULL);
       static void RemoveFile(const char* path, const char* sub=NULL, const char* name=NULL);
       static void CreateFile(const char* path, const char* sub=NULL, const char* name=NULL);

       static std::time_t* LastWriteTime(const char* path,  const char* sub=NULL, const char* name=NULL);
       static std::time_t* SinceLastWriteTime(const char* path,  const char* sub=NULL, const char* name=NULL);

       static std::string CreateDir(const char* base, const char* asub=NULL, const char* bsub=NULL);

    public:
        // refugees from BJson in need of de-duping
        static std::string preparePath(const char* path_, bool create=true );
        static std::string preparePath(const char* dir_, const char* name, bool create=true );
        static std::string preparePath(const char* dir_, const char* reldir_, const char* name, bool create=true );
        static std::string prefixShorten( const char* path, const char* prefix_);


    private:
       static void setOpticksPathPrefix(const char* prefix);
       static void setOpticksPathPrefixFromEnv(const char* envvar="OPTICKS_PATH_PREFIX");
       static void dumpOpticksPathPrefix(const char* msg="BFile::dumpOpticksPathPrefix");
    private:
       static char* OPTICKS_PATH_PREFIX ; 
       static const std::vector<std::string>  envvars ;


};


#include "BRAP_TAIL.hh"

