#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "SSys.hh"
#include "BFile.hh"
#include "BList.hh"
#include "BMeta.hh"
#include "PLOG.hh"



const char* BMeta::EXT = ".ini" ; 


std::string BMeta::Name(const char* label) // static
{
    std::stringstream ss ; 
    ss << label << EXT ;
    return ss.str();
}

BMeta* BMeta::Load(const char* dir, const char* label )  // static
{
    BMeta* t = new BMeta(label) ;
    t->load(dir);
    return t ; 
}



BMeta::BMeta(const char* label)
    :
    m_label(strdup(label))
{
}

void BMeta::add(const char* k, const char* v)
{
    m_kv.push_back(SS(k, v));
}

void BMeta::addEnvvar(const char* k)
{
    const char* v = SSys::getenvvar(k) ; 
    add(k, v == NULL ? "" : v ) ; 
}


void BMeta::dump(const char* msg) const 
{
    LOG(info) << m_label << " " << msg ; 
    for(VSS::const_iterator it=m_kv.begin() ; it != m_kv.end() ; it++)
    {
         std::cout 
               << std::setw(20) << it->first 
               << " : " 
               << it->second
               << std::endl 
               ;
    }
}

void BMeta::save(const char* dir) 
{
    std::string name = Name(m_label);
    std::string path = BFile::preparePath(dir, name.c_str(), true);
    LOG(debug) << path ;
    BList<std::string, std::string>::save( &m_kv, dir, name.c_str());
}

void BMeta::load(const char* dir)
{
    std::string name = Name(m_label);
    BList<std::string, std::string>::load( &m_kv, dir, name.c_str());
}


