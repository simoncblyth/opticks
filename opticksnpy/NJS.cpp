#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

#include "BFile.hh"


#include "NJS.hpp"
#include "PLOG.hh"

#ifdef _MSC_VER
#define strdup _strdup
#endif


NJS::NJS()
   :
   m_js()
{
}

NJS::NJS(const nlohmann::json& js)
   :
   m_js(js)
{
}

nlohmann::json& NJS::get()
{
    return m_js ; 
}  
 

void NJS::read(const char* path_)
{
    std::string path = BFile::FormPath(path_);

    LOG(info) << "read from " << path ; 

    std::ifstream in(path.c_str(), std::ios::in);

    if(!in.is_open()) 
    {   
        LOG(fatal) << "NJS::read failed to open " << path ; 
        return ;
    }   
    in >> m_js ; 
}

void NJS::write(const char* path_) const 
{
    std::string path = BFile::FormPath(path_);

    std::string pdir = BFile::ParentDir(path.c_str());

    BFile::CreateDir(pdir.c_str()); 

    LOG(info) << "write to " << path ; 

    std::ofstream out(path.c_str(), std::ios::out);

    if(!out.is_open()) 
    {   
        LOG(fatal) << "NJS::write failed to open" << path ; 
        return ;
    }   

    out << m_js ; 
    out.close();
}

void NJS::dump(const char* msg) const  
{
    LOG(info) << msg ; 
    std::cout << std::setw(4) << m_js << std::endl ; 
}



