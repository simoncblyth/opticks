#pragma once

#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_FLAGS.hh"

class BRAP_API BFile {
    public:
       static std::string FormPath(const char* path, const char* sub=NULL, const char* name=NULL);
       static std::string FindFile(const char* dirlist, const char* sub, const char* name=NULL, const char* dirlist_delim=";");
       static std::string Stem(const char* path);
       static std::string Name(const char* path);

       static bool ExistsNativeFile(const std::string& native);
       static bool ExistsNativeDir(const std::string& native);
       static bool ExistsFile(const char* path, const char* sub=NULL, const char* name=NULL);
       static bool ExistsDir(const char* path, const char* sub=NULL, const char* name=NULL);
       static void CreateDir(const char* path, const char* sub=NULL);


    public:
        // refugees from BJson in need of de-duping
        static bool existsPath(const char* path );
        static bool existsPath(const char* dir, const char* name );
        static std::string preparePath(const char* dir_, const char* name, bool create=false );
        static std::string preparePath(const char* dir_, const char* reldir_, const char* name, bool create=false );
        static std::string prefixShorten( const char* path, const char* prefix_);


    private:
       static void setOpticksPathPrefix(const char* prefix);
       static void setOpticksPathPrefixFromEnv(const char* envvar="OPTICKS_PATH_PREFIX");
       static void dumpOpticksPathPrefix(const char* msg="BFile::dumpOpticksPathPrefix");
    private:
       static char* OPTICKS_PATH_PREFIX ; 
};



