#pragma once

#include <map>
#include <string>

// see ggeo-/tests/LookupTest.cc for usage
class Lookup {
   public:  

   typedef std::map<unsigned int, unsigned int> Lookup_t ;
   typedef std::map<std::string, unsigned int> Map_t ;

   public:  
       Lookup();
       void crossReference();
       std::map<std::string, unsigned int>& getA();
       std::map<std::string, unsigned int>& getB();
       void loadA(const char* adir, const char* aname="ChromaMaterialMap.json",               const char* aprefix="/dd/Materials/");
       void loadB(const char* bdir, const char* bname="GBoundaryLibMetadataMaterialMap.json", const char* bprefix="");
   public:  
       // use
       int a2b(unsigned int a);
       int b2a(unsigned int b);
       void dump(const char* msg);
       std::string acode2name(unsigned int acode);
       std::string bcode2name(unsigned int bcode);

   private:  
       void dumpMap(const char* msg, Map_t& map);
       std::map<unsigned int, unsigned int>  create(Map_t& a, Map_t&b);
       std::string find(Map_t& m, unsigned int code);
       int lookup(Lookup_t& lkup, unsigned int x); 
   private:
       std::map<std::string, unsigned int>  m_A ; 
       std::map<std::string, unsigned int>  m_B ; 
       std::map<unsigned int, unsigned int> m_A2B ; 
       std::map<unsigned int, unsigned int> m_B2A ; 
 
};


inline Lookup::Lookup()
{
}

inline std::map<std::string, unsigned int>& Lookup::getA()
{
    return m_A ; 
}
inline std::map<std::string, unsigned int>& Lookup::getB()
{
    return m_B ; 
}





