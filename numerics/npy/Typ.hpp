#pragma once

#include <string>
#include <map>

// TODO: migrate to optickscore-
// attack class to kill Types


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API Typ {
    public:
        Typ();
        void setMaterialNames(std::map<unsigned int, std::string> material_names);
        void setFlagNames(std::map<unsigned int, std::string> flag_names);
        std::string findMaterialName(unsigned int);
    private:
        std::map<unsigned int, std::string> m_material_names ; 
        std::map<unsigned int, std::string> m_flag_names ; 

};

#include "NPY_TAIL.hh"

