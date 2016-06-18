#pragma once

#include <cstring>
#include <string>
#include <vector>

// vi Composition.hh Scene.hh Trackball.hh Camera.hh View.hh Clipper.hh
// vi Composition.cc Scene.cc Trackball.cc Camera.cc View.cc Clipper.cc

#include "NPY_API_EXPORT.hh"

class NPY_API NConfigurable {
    public: 
       virtual const char* getPrefix() = 0 ; 
       virtual void configure(const char* name, const char* value) = 0;
       virtual std::vector<std::string> getTags() = 0;
       virtual std::string get(const char* name) = 0 ;
       virtual void set(const char* name, std::string& val) = 0 ;
};



