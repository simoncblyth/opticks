#pragma once

#include <vector>
#include <map>
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

/**
BResource
==========

Holds key,value string pairs in three categories

* names
* dirs
* paths


**/

class BRAP_API BResource 
{
   private:
        static const BResource* INSTANCE ; 
   public:
        static const BResource* GetInstance() ;
        static const char* Get(const char* label ) ;
        static void Dump(const char* msg="BResource::Dump") ;
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

       const char* getPath( const char* label ) const ; 
       const char* getDir( const char* label ) const ; 
       const char* getName( const char* label ) const ; 
   private:
       typedef std::pair<std::string, std::string> SS ; 
       typedef std::vector<SS> VSS ; 
       const char* get(const char* label, const VSS& vss) const  ;
   protected:
        VSS m_paths ; 
        VSS m_dirs ; 
        VSS m_names ; 

};



 
