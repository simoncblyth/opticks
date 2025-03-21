#pragma once

#include <cstring>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <errno.h> 
#include <vector>
#include <algorithm>
#include "dirent.h"

struct sdirectory
{
    static std::string DirName( const char* filepath );
    static int MakeDirsForFile(const char* filepath, int mode=0 );  
    static int MakeDirs(const char* dirpath, int mode=0 );  
    static void DirList( std::vector<std::string>& names, const char* path, const char* pfx=nullptr, const char* ext=nullptr); 
};


inline std::string sdirectory::DirName( const char* filepath )
{
    std::string p = filepath ; 
    std::size_t pos = p.find_last_of("/") ; 
    return pos == std::string::npos ? "" : p.substr(0, pos); 
}

inline int sdirectory::MakeDirsForFile( const char* filepath, int mode_ )
{
    if(filepath == nullptr) return 1 ; 
    std::string dirpath = DirName(filepath);
    return MakeDirs(dirpath.c_str(), mode_ );
}


/**
sdirectory::MakeDirs
----------------------

While loop is needed because mkdir can only create one 
level of directory at once. 
This follows NPU.hh U::MakeDirs

**/

inline int sdirectory::MakeDirs( const char* dirpath_, int mode_ )
{
    mode_t default_mode = S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH ;
    mode_t mode = mode_ == 0 ? default_mode : mode_ ;

    char* dirpath = strdup(dirpath_);
    char* p = dirpath ;
    int rc = 0 ;

    while (*p != '\0' && rc == 0)
    {
        p++;   // advance past leading character, probably slash, and subsequent slashes the next line gets to  
        while(*p != '\0' && *p != '/') p++;  // advance p until subsequent slash 
        char v = *p;                         // store the slash      
        *p = '\0' ;                          // replace slash with string terminator
        //printf("%s\n", path );                   
        rc = mkdir(dirpath, mode) == -1 && errno != EEXIST ? 1 : 0 ;  // set rc non-zero for mkdir errors other than exists already  
        *p = v;                              // put back the slash  
    }   
    free(dirpath);
    return rc ; 
}

/**
sdirectory::DirList
--------------------

**/


inline void sdirectory::DirList( std::vector<std::string>& names, const char* path, const char* pfx, const char* ext)
{
    DIR* dir = opendir(path) ;
    if(!dir) std::cout << "sdirectory::DirList FAILED TO OPEN DIR " << ( path ? path : "-" ) << std::endl ; 
    if(!dir) return ; 
    struct dirent* entry ;
    while ((entry = readdir(dir)) != nullptr) 
    {   
        const char* name = entry->d_name ;
        bool dot_name = strcmp(name,".") == 0 || strcmp(name,"..") == 0 ; 
        if(dot_name) continue ; 
 
        bool pfx_match = pfx == nullptr ? true : ( strlen(name) > strlen(pfx) && strncmp(name, pfx, strlen(pfx)) == 0 ) ; 
        bool ext_match = ext == nullptr ? true : ( strlen(name) > strlen(ext) && strcmp(name + strlen(name) - strlen(ext), ext)==0)  ;
        if(ext_match && pfx_match) names.push_back(name); 
    }   
    closedir (dir);
    std::sort( names.begin(), names.end() );  

    if(names.size() == 0 ) std::cout 
        << "sdirectory::DirList" 
        << " path " << ( path ? path : "-" ) 
        << " pfx " << ( pfx ? pfx : "-" ) 
        << " ext " << ( ext ? ext : "-" ) 
        << " NO ENTRIES FOUND "
        << std::endl
        ;   
}




