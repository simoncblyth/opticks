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
        static BResource* INSTANCE ; 
   public:
        static const BResource* GetInstance() ;
   public:
        static const char* GetPath(const char* label ) ;
        static const char* GetDir(const char* label ) ;
        static const char* GetName(const char* label ) ;
   public:
        static void SetPath(const char* label, const char* value ) ;
        static void SetDir(const char* label, const char* value ) ;
        static void SetName(const char* label, const char* value ) ;

        static void Dump(const char* msg="BResource::Dump") ;
   public:
        BResource();
        virtual ~BResource();
   public:       
        void addDir( const char* label, const char* dir);
        void addPath( const char* label, const char* path);
        void addName( const char* label, const char* name);

        void setDir( const char* label, const char* dir);
        void setPath( const char* label, const char* path);
        void setName( const char* label, const char* name);


       // resource existance dumping 
       void dumpPaths(const char* msg) const ;
       void dumpDirs(const char* msg) const ;
       void dumpNames(const char* msg) const ;

       const char* getPath( const char* label ) const ; 
       const char* getDir( const char* label ) const ; 
       const char* getName( const char* label ) const ; 

       bool hasPath( const char* label ) const ; 
       bool hasDir( const char* label ) const ; 
       bool hasName( const char* label ) const ; 

   private:
       typedef std::pair<std::string, std::string> SS ; 
       typedef std::vector<SS> VSS ; 
       static const char* get(const char* label, const VSS& vss) ;
       static void        set(const char* label, const char* value, VSS& vss) ;
       static void       add(const char* label, const char* value, VSS& vss) ;
       static unsigned  count(const char* label, const VSS& vss) ;
   protected:
        VSS m_paths ; 
        VSS m_dirs ; 
        VSS m_names ; 

};



 
