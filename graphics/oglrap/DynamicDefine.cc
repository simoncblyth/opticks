#include <cstring>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

#include "BFile.hh"
#include "PLOG.hh"

#include "DynamicDefine.hh"



DynamicDefine::DynamicDefine() 
{
}    


void DynamicDefine::write(const char* dir, const char* name)
{

    LOG(info) << "DynamicDefine::write"
              << " dir " << dir
              << " name " << name
              ;


     bool create = true ; 
     std::string path = BFile::preparePath(dir,  name, create );   

     if(path.empty())
     {
         LOG(warning) << "DynamicDefine::write failed to preparePath " << dir << " " << name ; 
         return ; 
     }

     typedef std::vector<std::pair<std::string, std::string> >::const_iterator  VSSI ;

     std::stringstream ss ; 
     ss << "// see oglrap-/DynamicDefine::write invoked by Scene::write App::prepareScene " << std::endl ; 
     for(VSSI it=m_defines.begin() ; it != m_defines.end() ; it++)
     {
         ss << "#define " << it->first << " " << it->second << std::endl ; 
     }  

     std::string txt = ss.str() ;

     LOG(debug) << "DynamicDefine::write " << path ;
     LOG(debug) << txt ; 

     std::ofstream out(path.c_str(), std::ofstream::binary);
     out << txt ;
}


template <typename T>
void DynamicDefine::add(const char* name, T value)
{
    LOG(info) << "DynamicDefine::add"
              << " name " << name
              << " value " << value
              ; 


    m_defines.push_back(std::pair<std::string, std::string>(name, boost::lexical_cast<std::string>(value))) ;
}


// explicit instanciation
template OGLRAP_API void DynamicDefine::add<int>(const char* name, int value);
template OGLRAP_API void DynamicDefine::add<unsigned int>(const char* name, unsigned int value);
template OGLRAP_API void DynamicDefine::add<float>(const char* name, float value);
