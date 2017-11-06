#include "PLOG.hh"
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



void Typ::dumpMap(const char* msg, const std::map<unsigned, std::string>& m ) const 
{
    typedef std::map<unsigned, std::string> MUS ; 

    LOG(info) << msg ; 
    for(MUS::const_iterator it=m.begin() ; it != m.end() ; it++)
    {
       std::cout 
          << std::setw(5) << it->first
          << " : " 
          << std::setw(30) << it->second
          << std::endl 
          ;
    } 

}


void Typ::dump(const char* msg) const 
{
    LOG(info) << msg ; 

    dumpMap( "material_names", m_material_names );
    dumpMap( "flag_names", m_flag_names );
}




