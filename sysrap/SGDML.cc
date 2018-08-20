#include <sstream>
#include "SGDML.hh"


std::string SGDML::GenerateName(const char* name, const void* const ptr, bool addPointerToName )
{
    std::stringstream ss; 
    ss << name;
    if(addPointerToName) ss << ptr ; 
    std::string nameOut = ss.str();

    if(nameOut.find(' ') != std::string::npos)
         nameOut.erase(std::remove(nameOut.begin(),nameOut.end(),' '),nameOut.end());
    return nameOut;
}


