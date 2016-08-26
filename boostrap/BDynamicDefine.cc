#include <cstring>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

#include "BFile.hh"
#include "PLOG.hh"

#include "BDynamicDefine.hh"



BDynamicDefine::BDynamicDefine() 
{
}    


void BDynamicDefine::write(const char* dir, const char* name)
{

    LOG(trace) << "BDynamicDefine::write"
              << " dir " << dir
              << " name " << name
              ;


     bool create = true ; 
     std::string path = BFile::preparePath(dir,  name, create );   

     if(path.empty())
     {
         LOG(warning) << "BDynamicDefine::write failed to preparePath " << dir << " " << name ; 
         return ; 
     }

     typedef std::vector<std::pair<std::string, std::string> >::const_iterator  VSSI ;

     std::stringstream ss ; 
     ss << "// see boostrap-/BDynamicDefine::write invoked by Scene::write App::prepareScene " << std::endl ; 
     for(VSSI it=m_defines.begin() ; it != m_defines.end() ; it++)
     {
         ss << "#define " << it->first << " " << it->second << std::endl ; 
     }  

     std::string txt = ss.str() ;

     LOG(debug) << "BDynamicDefine::write " << path ;
     LOG(debug) << txt ; 

     std::ofstream out(path.c_str(), std::ofstream::binary);
     out << txt ;
}


template <typename T>
void BDynamicDefine::add(const char* name, T value)
{
    LOG(trace) << "BDynamicDefine::add"
              << " name " << name
              << " value " << value
              ; 


    m_defines.push_back(std::pair<std::string, std::string>(name, boost::lexical_cast<std::string>(value))) ;
}


// explicit instanciation
template BRAP_API void BDynamicDefine::add<int>(const char* name, int value);
template BRAP_API void BDynamicDefine::add<unsigned int>(const char* name, unsigned int value);
template BRAP_API void BDynamicDefine::add<float>(const char* name, float value);
