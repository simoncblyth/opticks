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

struct sfilesystem
{
    static bool        all_digits(const char* str_);
    static bool        is_indexed_dirname( const char* dirname, const char* prefix );
    static long long   get_index_from_chars( const char* index_part );
    static long long   get_index_from_indexed_dirname( const char* dirname, const char* prefix );

    static long long   find_index_of_max_indexed_dirname(const char* container_dir, const char* prefix_);
    static std::string form_indexed_dirname( long long index, const char* prefix, int num_digits=7 );
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

**/

inline long long sfilesystem::find_index_of_max_indexed_dirname(const char* container_dir_, const char* prefix)
{
    namespace fs = std::filesystem;

    fs::path container_dir = container_dir_ ;

    if (!fs::exists(container_dir) || !fs::is_directory(container_dir)) {
        throw std::runtime_error("Target directory does not exist or is not a directory.");
    }

    long long max_index = -1; // Returns -1 if no matching directories are found
    for (const auto& entry : fs::directory_iterator(container_dir))
    {
        if (!entry.is_directory()) continue ;
        std::string dir_name = entry.path().filename().string();
        const char* dirname = dir_name.c_str();
        if(!is_indexed_dirname(dirname, prefix)) continue ;
        long long parsed_index = get_index_from_indexed_dirname(dirname, prefix);
        max_index = std::max(max_index, parsed_index);
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


