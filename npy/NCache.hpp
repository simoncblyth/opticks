#pragma once

#include <string>
#include <boost/filesystem.hpp>


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

// TODO: adopt BFile::FormPath

class NPY_API NCache {
   public:
      NCache(const char* dir);
   public:
      std::string path(const char* relative);
      std::string path(const char* tmpl, const char* incl);
      std::string path(const char* tmpl, unsigned int incl);
   private:
      boost::filesystem::path m_cache ; 

};

#include "NPY_TAIL.hh"

