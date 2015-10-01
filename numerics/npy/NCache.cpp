#include "NCache.hpp"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

std::string NCache::path(const char* relative)
{   
    fs::path cpath(m_cache/relative); 
    return cpath.string();
} 

std::string NCache::path(const char* tmpl, const char* incl)
{   
    char p[128];
    snprintf(p, 128, tmpl, incl);
    fs::path cpath(m_cache/p); 
    return cpath.string();
}   

std::string NCache::path(const char* tmpl, unsigned int incl)
{   
    char p[128];
    snprintf(p, 128, tmpl, incl);
    fs::path cpath(m_cache/p); 
    return cpath.string();
} 


 
