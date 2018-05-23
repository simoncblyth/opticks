#pragma once

#include <map>
#include <string>

// see ggeo-/tests/NLookupTest.cc for usage

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NLookup {
   public:  
       typedef std::map<unsigned int, unsigned int> NLookup_t ;
       typedef std::map<std::string, unsigned int> MSU ;
   public:  
       static void mockup(const char* dir="/tmp", const char* aname="mockA.json", const char* bname="mockB.json");
   private:
       static void mockA(const char* adir, const char* aname);
       static void mockB(const char* bdir, const char* bname);
   public:  
       NLookup();
       void close(const char* msg="NLookup::close");
       bool isClosed() const ; 
    
       const std::map<std::string, unsigned int>& getA();
       const std::map<std::string, unsigned int>& getB();


       void setA( const char* json );
       void setA( const std::map<std::string, unsigned int>& A, const char* aprefix="/dd/Materials/", const char* alabel="-");
       void setB( const std::map<std::string, unsigned int>& B, const char* bprefix=""              , const char* blabel="-" );

       void loadA(const char* adir, const char* aname="ChromaMaterialMap.json",               const char* aprefix="/dd/Materials/");
       void loadB(const char* bdir, const char* bname="GBoundaryLibMetadataMaterialMap.json", const char* bprefix="");
   public:  
       // use
       std::string brief();

       int a2b(unsigned int a);
       int b2a(unsigned int b);
       void dump(const char* msg);
       std::string acode2name(unsigned int acode);
       std::string bcode2name(unsigned int bcode);

   private:  
       void crossReference(); // cross referencing A2B and B2A codes which correspond to the same names
       void dumpMap(const char* msg, MSU& map);
       std::map<unsigned int, unsigned int>  create(MSU& a, MSU&b);
       std::string find(MSU& m, unsigned int code);
       int lookup(NLookup_t& lkup, unsigned int x); 
   private:
       std::map<std::string, unsigned int>  m_A ; 
       std::map<std::string, unsigned int>  m_B ; 
       std::map<unsigned int, unsigned int> m_A2B ; 
       std::map<unsigned int, unsigned int> m_B2A ; 

       const char* m_alabel ; 
       const char* m_blabel ; 
       bool  m_closed ; 

 
};

#include "NPY_TAIL.hh"



