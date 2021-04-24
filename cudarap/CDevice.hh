#pragma once

/**
CDevice
============

This is used from OContext::initDevices

CDevice instance persists summary details about a single device.
Static methods with std::vector arguments used to handle multiple
devices.  

By persisting CDevice for all attached devices when the CUDA_VISIBLE_DEVICES
envvar is not defined it becomes possible to get the "absolute" ordinal 
when the envvar is used and only a subset of all devices are visible.

**/

#include <cstddef>
#include <string>
#include <vector>
#include <iostream>
#include "plog/Severity.h"

#include "CUDARAP_API_EXPORT.hh"
struct CUDARAP_API CDevice {

    static const plog::Severity LEVEL ; 

    int ordinal ; 
    int index ; 

    char name[256] ; 
    char uuid[16] ; 
    int major  ; 
    int minor  ; 
    int compute_capability ; 
    int multiProcessorCount ; 
    size_t totalGlobalMem ; 

    float totalGlobalMem_GB() const ;
    void read( std::istream& in ); 
    void write( std::ostream& out ) const ; 
    bool matches(const CDevice& other) const ; 

    const char* brief() const ;
    const char* desc() const ; 

    static const char* CVD ; 
    static const char* FILENAME ; 
    static int Size(); 
    static void Visible(std::vector<CDevice>& visible, const char* dirpath, bool nosave=false ); 
    static void Collect(std::vector<CDevice>& devices, bool ordinal_from_index=false ); 
    static int FindIndexOfMatchingDevice( const CDevice& d, const std::vector<CDevice>& all );

    static std::string Path(const char* dirpath); 
    static void Dump(const std::vector<CDevice>& devices, const char* msg ); 
    static void Save(const std::vector<CDevice>& devices, const char* dirpath); 
    static void Load(      std::vector<CDevice>& devices, const char* dirpath); 

    static void PrepDir(const char* dirpath); 
    static std::string Brief( const std::vector<CDevice>& devices ); 

};




