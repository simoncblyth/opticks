#pragma once

#include <map>
#include <string>


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

/**
NLookup
=========

Provides A2B or B2A int->int mappings between two indexing schemes.
The A and B indexing schemes being represented by name to index maps
and the correspondence being established based on the names. 
After instanciating an NLookup the A and B are set either from 
loaded json or from std::map and only on closing the NLookup
are the A2B and B2A cross referencing maps formed.


See ggeo/tests/NLookupTest.cc for usage example

**/

class NPY_API NLookup {
   public:  
       typedef std::map<unsigned,unsigned> NLookup_t ;
       typedef std::map<std::string, unsigned> MSU ;
   public:  
       static void mockup(const char* dir="/tmp", const char* aname="mockA.json", const char* bname="mockB.json");
   private:
       static void mockA(const char* adir, const char* aname);
       static void mockB(const char* bdir, const char* bname);
   public:  
       NLookup();
       void close(const char* msg="NLookup::close");
       bool isClosed() const ; 
    
       const std::map<std::string, unsigned>& getA() const ;
       const std::map<std::string, unsigned>& getB() const ;


       void setA( const char* json );
       void setA( const std::map<std::string, unsigned>& A, const char* aprefix="/dd/Materials/", const char* alabel="-");
       void setB( const std::map<std::string, unsigned>& B, const char* bprefix=""              , const char* blabel="-" );

       void loadA(const char* adir, const char* aname="ChromaMaterialMap.json",               const char* aprefix="/dd/Materials/");
       void loadB(const char* bdir, const char* bname="GBoundaryLibMetadataMaterialMap.json", const char* bprefix="");
   public:  
       // use
       std::string brief() const ;
       int a2b(unsigned a) const ;
       int b2a(unsigned b) const ;
       void dump(const char* msg) const ;
       std::string acode2name(unsigned acode) const ;
       std::string bcode2name(unsigned bcode) const ;

   private:  
       void dumpMap(const char* msg, MSU& map) const ;
       std::string find(const MSU& m, unsigned code) const ;
       int lookup(const NLookup_t& lkup, unsigned x) const ; 
       std::map<unsigned, unsigned>  create(MSU& a, MSU&b) const ;
   private:  
       void crossReference(); // cross referencing A2B and B2A codes which correspond to the same names
   private:
       std::map<std::string, unsigned>  m_A ; 
       std::map<std::string, unsigned>  m_B ; 
       std::map<unsigned, unsigned>     m_A2B ; 
       std::map<unsigned, unsigned>     m_B2A ; 

       const char* m_alabel ; 
       const char* m_blabel ; 
       bool  m_closed ; 

 
};

#include "NPY_TAIL.hh"



