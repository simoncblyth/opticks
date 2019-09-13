#pragma once

/**
CDevice
============

**/

#include <cstddef>
#include <string>
#include <vector>
#include <iostream>

#include "CUDARAP_API_EXPORT.hh"
struct CUDARAP_API CDevice {

    int ordinal ; 
    int index ; 

    char name[256] ; 
    char uuid[16] ; 
    int major  ; 
    int minor  ; 
    int compute_capability ; 
    int multiProcessorCount ; 
    size_t totalGlobalMem ; 

    void read( std::istream& in ); 
    void write( std::ostream& out ) const ; 
    bool matches(const CDevice& other) const ; 
    const char* desc() const ; 

    static const char* CVD ; 
    static const char* FILENAME ; 
    static int Size(); 
    static void Visible(std::vector<CDevice>& visible, const char* dirpath, bool nosave=false ); 
    static void Collect(std::vector<CDevice>& devices, bool ordinal_from_index=false ); 
    static int FindIndexOfMatchingDevice( const CDevice& d, const std::vector<CDevice>& all );

    static std::string Path(const char* dirpath); 
    static void Dump(const std::vector<CDevice>& devices ); 
    static void Save(const std::vector<CDevice>& devices, const char* dirpath); 
    static void Load(      std::vector<CDevice>& devices, const char* dirpath); 

};




