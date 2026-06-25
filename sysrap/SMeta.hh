#pragma once

#include "SYSRAP_API_EXPORT.hh"
#include "json.hpp"
#include <iostream>
#include "plog/Severity.h"


/**
SMeta
=======

Lightweight metadata based on https://nlohmann.github.io/json/

::

    [lo] A[blyth@localhost opticks]$ opticks-fl SMeta
    ./CSGOptiX/CSGOptiX.h
    ./CSGOptiX/CSGOptiX.cc
    ./sysrap/CMakeLists.txt
    ./sysrap/SMeta.cc
    ./sysrap/SMeta.hh
    ./sysrap/tests/SMetaTest.cc
    ./sysrap/tests/CMakeLists.txt


Issues with SMeta:

* near name collision with unrelated smeta.h
* looses header only convenience of json.hpp just for plog ?
* adoption of header only spath.h should make most of the save and Load methods redundant
* currently the only use of SMeta.hh is from the below for jpg json sidecars::

   CSGOptiX::saveMeta(const char* jpg_path) const

Little use, and issues - mean this can easily be replaced

* sjson.h

**/

struct SYSRAP_API SMeta
{
   static SMeta* Load(const char* path);
   static SMeta* Load(const char* dir, const char* name);
   static const plog::Severity LEVEL ;

   nlohmann::json js ;
   void save(const char* dir, const char* reldir, const char* name) const ;
   void save(const char* dir, const char* name) const ;
   void save(const char* path) const ;

};


inline std::ostream& operator<<(std::ostream& os, const SMeta& sm)
{
    os << sm.js ;
    return os;
}



