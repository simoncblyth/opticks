#pragma once

#include <string>
#include <map>

// TODO: migrate to optickscore-
// attack class to kill Types


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

/* 
Typ
====

Setup from GGeo::setupTyp GGeo::loadGeometry

*/


class NPY_API Typ {
    public:
        Typ();
        void setMaterialNames(std::map<unsigned int, std::string> material_names);
        void setFlagNames(std::map<unsigned int, std::string> flag_names);
        std::string findMaterialName(unsigned int);

        void dump(const char* msg="Typ::dump") const ;
        void dumpMap(const char* msg, const std::map<unsigned, std::string>& m ) const ;

    private:
        std::map<unsigned int, std::string> m_material_names ; 
        std::map<unsigned int, std::string> m_flag_names ; 

};

#include "NPY_TAIL.hh"

