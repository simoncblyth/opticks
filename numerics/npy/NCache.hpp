#pragma once

#include <string>
#include <boost/filesystem.hpp>

class NCache {
   public:
      NCache(const char* dir);
   public:
      std::string path(const char* relative);
      std::string path(const char* tmpl, const char* incl);
      std::string path(const char* tmpl, unsigned int incl);
   private:
      boost::filesystem::path m_cache ; 

};


inline NCache::NCache(const char* dir) 
   : 
      m_cache(dir) 
{
} 

