#pragma once

#include "string.h"
#include <string>
#include <map>

class GItemIndex {
        friend class GSurfaceIndex ; 
        friend class GMaterialIndex ; 
   public:
        GItemIndex(const char* itemtype);
        void save(const char* idpath);

   public:
        // invoked from GBoundaryLib::createWavelengthAndOpticalBuffers
        void add(const char* name, unsigned int index);
        unsigned int getIndexLocal(const char* name, unsigned int missing=0);

   public:
        unsigned int getIndexSource(const char* name, unsigned int missing=0);
        const char* getNameLocal(unsigned int local, const char* missing=NULL);
        const char* getNameSource(unsigned int source, const char* missing=NULL);

        unsigned int convertLocalToSource(unsigned int local, unsigned int missing=0);
        unsigned int convertSourceToLocal(unsigned int source, unsigned int missing=0);

   public:
        unsigned int getNumItems();
        void dump(const char* msg="GItemIndex::dump");
        void test(const char* msg="GItemIndex::test");
        bool operator() (const std::string& a, const std::string& b);

   private:
        std::string getPrefixedString(const char* tail);
        const char* getItemType();
        void loadMaps(const char* idpath);
        void crossreference();

   private:
        const char*                          m_itemtype ; 
        std::map<std::string, unsigned int>  m_source ; 
        std::map<std::string, unsigned int>  m_local ; 
        std::map<unsigned int, unsigned int> m_source2local ; 
        std::map<unsigned int, unsigned int> m_local2source ; 
};

inline GItemIndex::GItemIndex(const char* itemtype)
   : m_itemtype(strdup(itemtype))
{
}
inline const char* GItemIndex::getItemType()
{
    return m_itemtype ; 
}

