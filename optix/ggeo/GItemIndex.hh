#pragma once

class GColors ; 
class GColorMap ; 
class GBuffer ; 

#include "string.h"
#include <string>
#include <map>

class GItemIndex {
        friend class GSurfaceIndex ; 
        friend class GMaterialIndex ; 
   public:
        GItemIndex(const char* itemtype);
        void setColorSource(GColors* colors);
        void setColorMap(GColorMap* colormap);
        void save(const char* idpath);
   public:
        GBuffer* makeColorBuffer();
        GBuffer* getColorBuffer();
   public:
        // invoked from GBoundaryLib::createWavelengthAndOpticalBuffers
        void add(const char* name, unsigned int index);
        unsigned int getIndexLocal(const char* name, unsigned int missing=0);

   public:
        std::string getPrefixedString(const char* tail);
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
        const char* getItemType();
        void loadMaps(const char* idpath);
        void crossreference();

   private:
        const char*                          m_itemtype ; 
        std::map<std::string, unsigned int>  m_source ; 
        std::map<std::string, unsigned int>  m_local ; 
        std::map<unsigned int, unsigned int> m_source2local ; 
        std::map<unsigned int, unsigned int> m_local2source ; 
        GColors*                             m_colors ; 
        GColorMap*                           m_colormap ; 
        GBuffer*                             m_colorbuffer ; 
};

inline GItemIndex::GItemIndex(const char* itemtype)
   : 
   m_itemtype(strdup(itemtype)),
   m_colors(NULL),
   m_colormap(NULL),
   m_colorbuffer(NULL)
{
}


inline void GItemIndex::setColorSource(GColors* colors)
{
   m_colors = colors ; 
}
inline void GItemIndex::setColorMap(GColorMap* colormap)
{
   m_colormap = colormap ; 
}

inline const char* GItemIndex::getItemType()
{
    return m_itemtype ; 
}

