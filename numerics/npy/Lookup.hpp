#pragma once

#include <map>
#include <string>

class Lookup {

   public:  
   typedef std::map<unsigned int, unsigned int> Lookup_t ;
   typedef std::map<std::string, unsigned int> Map_t ;
   public:  
       Lookup();
       void create(const char* apath, const char* aprefix, const char* bpath, const char* bprefix, bool dump=false);
       int a2b(unsigned int a);
       int b2a(unsigned int b);

   private:  
       //Map_t parseTree(boost::property_tree::ptree& tree, const char* prefix) ;
       void dumpMap(const char* msg, Map_t& map);

       std::map<unsigned int, unsigned int> _create(const char* apath, const char* aprefix, const char* bpath, const char* bprefix);
       std::map<unsigned int, unsigned int> _create(Map_t& a, Map_t&b);
       std::string find(Map_t& m, unsigned int code);

       int lookup(Lookup_t& lkup, unsigned int x); 

   private:
        Lookup_t m_a2b ; 
        Lookup_t m_b2a ; 
 
};



