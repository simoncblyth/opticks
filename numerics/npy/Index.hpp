#pragma once

#include "string.h"
#include <string>
#include <map>
#include <vector>

class Index {
   public:
        static Index* load(const char* idpath, const char* itemtype);
        Index(const char* itemtype);
        void save(const char* idpath);
        const char* getItemType();
   private:
        void loadMaps(const char* idpath);
        void crossreference();
   public:
        void add(const char* name, unsigned int source);
        void sortNames(); // currently by ascending local index : ie addition order
        std::vector<std::string>& getNames();
        bool operator() (const std::string& a, const std::string& b);
   public:
        std::string getPrefixedString(const char* tail);
        unsigned int getIndexLocal(const char* name, unsigned int missing=0);
        unsigned int getIndexSource(const char* name, unsigned int missing=0);
        const char* getNameLocal(unsigned int local, const char* missing=NULL);
        const char* getNameSource(unsigned int source, const char* missing=NULL);

        unsigned int convertLocalToSource(unsigned int local, unsigned int missing=0);
        unsigned int convertSourceToLocal(unsigned int source, unsigned int missing=0);

   public:
        unsigned int getNumItems();
        void test(const char* msg="GItemIndex::test", bool verbose=true);
        void dump(const char* msg="GItemIndex::dump");

   private:
        const char*                          m_itemtype ; 
        std::map<std::string, unsigned int>  m_source ; 
        std::map<std::string, unsigned int>  m_local ; 
        std::map<unsigned int, unsigned int> m_source2local ; 
        std::map<unsigned int, unsigned int> m_local2source ; 
        std::vector<std::string>             m_names ; 
   private:
        // populated by formTable
        std::vector<std::string>             m_labels ; 
        std::vector<unsigned int>            m_codes ; 
};

inline Index::Index(const char* itemtype)
   : 
   m_itemtype(strdup(itemtype))
{
}
inline const char* Index::getItemType()
{
    return m_itemtype ; 
}
inline std::vector<std::string>& Index::getNames()
{
    return m_names ;  
}


