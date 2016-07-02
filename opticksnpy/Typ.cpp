#include "Typ.hpp"

Typ::Typ()
{
}

void Typ::setMaterialNames(std::map<unsigned int, std::string> material_names)
{
    m_material_names = material_names ; 
}
void Typ::setFlagNames(std::map<unsigned int, std::string> flag_names)
{
    m_flag_names = flag_names ; 
}

std::string Typ::findMaterialName(unsigned int index)
{
    return m_material_names.count(index) == 1 ? m_material_names[index] : "?" ; 
}



