#pragma once

#include <string>
#include <map>

// TODO: migrate to optickscore-

// attack class to kill Types
class Typ {
    public:
        Typ();
        void setMaterialNames(std::map<unsigned int, std::string> material_names);
        void setFlagNames(std::map<unsigned int, std::string> flag_names);
        std::string findMaterialName(unsigned int);
    private:
        std::map<unsigned int, std::string> m_material_names ; 
        std::map<unsigned int, std::string> m_flag_names ; 

};

inline Typ::Typ()
{
}

inline void Typ::setMaterialNames(std::map<unsigned int, std::string> material_names)
{
    m_material_names = material_names ; 
}
inline void Typ::setFlagNames(std::map<unsigned int, std::string> flag_names)
{
    m_flag_names = flag_names ; 
}



inline std::string Typ::findMaterialName(unsigned int index)
{
    return m_material_names.count(index) == 1 ? m_material_names[index] : "?" ; 
}



