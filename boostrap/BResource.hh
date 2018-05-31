#pragma once

#include <vector>
#include <map>
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"


class BRAP_API BResource 
{
   public:
        BResource();
        virtual ~BResource();
   public:       
        void addDir( const char* label, const char* dir);
        void addPath( const char* label, const char* path);
        void addName( const char* label, const char* name);

       // resource existance dumping 
       void dumpPaths(const char* msg) const ;
       void dumpDirs(const char* msg) const ;
       void dumpNames(const char* msg) const ;

       const char* getPath(const char* label) const  ;
   protected:
        std::vector<std::pair<std::string, std::string> >  m_paths  ; 
        std::vector<std::pair<std::string, std::string> >  m_dirs  ; 
        std::vector<std::pair<std::string, std::string> >  m_names  ; 


};



 
