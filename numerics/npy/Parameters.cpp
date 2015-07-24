#include "Parameters.hpp"

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <iomanip>

template <typename T>
void Parameters::add(const char* name, T value)
{
    m_parameters.push_back(SS(name, boost::lexical_cast<std::string>(value) ));
}

void Parameters::dump(const char* msg)
{
   prepLines();
   std::cout << msg << std::endl ; 
   for(VS::const_iterator it=m_lines.begin() ; it != m_lines.end() ; it++) std::cout << *it << std::endl ;  
}

void Parameters::prepLines()
{
    m_lines.clear();
    for(VSS::const_iterator it=m_parameters.begin() ; it != m_parameters.end() ; it++)
    {
        std::string name  = it->first ; 
        std::string value = it->second ; 

        std::stringstream ss ;  
        ss 
             << std::fixed
             << std::setw(15) << name
             << " : " 
             << std::setw(15) << value 
             ;
        
        m_lines.push_back(ss.str());
    }
}



template void Parameters::add(const char* name, int value);
template void Parameters::add(const char* name, unsigned int value);
template void Parameters::add(const char* name, std::string value);
template void Parameters::add(const char* name, float value);


