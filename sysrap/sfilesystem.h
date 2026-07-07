#pragma once
/**
sfilesystem.h
==============

~/o/sysrap/tests/sfilesystem_test.sh


**/

#include <iostream>
#include <filesystem>
#include <string>
#include <cstring>
#include <algorithm>
#include <charconv> // For fast, safe string-to-int conversion
#include <iomanip>
#include <sstream>


#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h> // readlink on Linux
#endif



struct sfilesystem
{
    static bool        all_digits(const char* str_);
    static bool        is_indexed_dirname( const char* dirname, const char* prefix );
    static long long   get_index_from_chars( const char* index_part );
    static long long   get_index_from_indexed_dirname( const char* dirname, const char* prefix );

    static long long   find_index_of_max_indexed_dirname(const char* container_dir, const char* prefix_);
    static std::string form_indexed_dirname( long long index, const char* prefix, int num_digits=7 );

    static std::string ExecutablePath();
    static std::string ExecutablePathSibling(const char* name);

};


inline bool sfilesystem::all_digits(const char* str_)
{
    const std::string str = str_ ;
    return str.find_first_not_of("0123456789") == std::string::npos ;
}

inline bool sfilesystem::is_indexed_dirname( const char* path, const char* prefix ) // static
{
    namespace fs = std::filesystem;
    fs::path p(path);
    std::string dir_name = p.filename().string();
    const char* dirname = dir_name.c_str();

    bool starts_with_prefix = dirname && 0 == strncmp(dirname, prefix, strlen(prefix)) ;
    if(!starts_with_prefix) return false ;
    bool long_enough = dirname && strlen(dirname) > strlen(prefix) ;
    if(!long_enough) return false ;
    return all_digits( dirname + strlen(prefix) );
}

inline long long sfilesystem::get_index_from_chars( const char* index_part ) // static
{
    long long index = 0 ;
    auto [ptr, ec] = std::from_chars(index_part, index_part + strlen(index_part), index);
    bool parse_succeeded = ec == std::errc{} ; // essentially checking error code to be zero
    return parse_succeeded ? index : -2 ;
}

inline long long sfilesystem::get_index_from_indexed_dirname( const char* dirname, const char* prefix ) // static
{
    bool is_indexed = is_indexed_dirname(dirname, prefix);
    if(!is_indexed) return -2 ;
    const char* index_part = dirname + strlen(prefix) ;
    return get_index_from_chars(index_part);
}



/**
sfilesystem::find_index_of_max_indexed_dirname
-----------------------------------------------

Examines directories within eg *container_dir* looking for
directories with the provided *prefix* that are of the below form::

    container_dir/sreport_00000/
    container_dir/sreport_00001/
    container_dir/sreport_00101/

When such directories are found the higest index integer found
is returned or -1 if no such directory is present.
If the container_dir does not yet exist -1 is also returned.

**/

inline long long sfilesystem::find_index_of_max_indexed_dirname(const char* container_dir_, const char* prefix)
{
    long long max_index = -1; // Returns -1 if the container_dir does not exist OR no matching directories are found

    namespace fs = std::filesystem;
    fs::path container_dir = container_dir_ ;

    if(fs::exists(container_dir) && fs::is_directory(container_dir))
    {
        for (const auto& entry : fs::directory_iterator(container_dir))
        {
            if (!entry.is_directory()) continue ;
            std::string dir_name = entry.path().filename().string();
            const char* dirname = dir_name.c_str();
            if(!is_indexed_dirname(dirname, prefix)) continue ;
            long long parsed_index = get_index_from_indexed_dirname(dirname, prefix);
            max_index = std::max(max_index, parsed_index);
        }
    }
    else
    {
        std::cout << "sfilesystem::find_index_of_max_indexed_dirname container_dir_ does not exist yet {" << ( container_dir_ ? container_dir_ : "-" ) << "}\n" ;
        //throw std::runtime_error("sfilesystem::find_index_of_max_indexed_dirname - Target directory does not exist or is not a directory.");
    }
    return max_index;
}

inline std::string sfilesystem::form_indexed_dirname( long long index, const char* prefix, int num_digits )
{
    std::stringstream ss;
    ss << prefix << std::setw(num_digits) << std::setfill('0') << index ;
    std::string str = ss.str();
    return str ;
}




/**
sfilesystem::ExecutablePath
----------------------------

returns absolute path of the running executable

**/

std::string sfilesystem::ExecutablePath()
{
    namespace fs = std::filesystem;

    char buffer[1024];
#if defined(_WIN32)
    GetModuleFileNameA(NULL, buffer, sizeof(buffer));
    return fs::path(buffer).string() ;
#elif defined(__APPLE__)
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) return fs::path(buffer).string() ;
#else
    // Linux standard
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        return fs::path(buffer).string() ;
    }
#endif
    return "";
}


/**
sfilesystem::ExecutablePathSibling
------------------------------------

returns absolute path of sibling to the executable

**/

std::string sfilesystem::ExecutablePathSibling(const char* sibling_name)
{
    std::string _bin = ExecutablePath();
    if (_bin.empty()) return "";

    namespace fs = std::filesystem;
    fs::path bin(_bin);
    fs::path sib = bin.parent_path() / sibling_name  ;
    return sib.string();
}

