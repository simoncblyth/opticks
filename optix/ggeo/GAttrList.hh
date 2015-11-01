#pragma once

/*
History 

   GItemIndex 
        became overcomplicated to compensate for lack of persistable GMaterialLib, GSurfaceLib
        (GBoundaryLib attempted to skip a few steps and live without these)

   GItemList
        simple list of strings, used by GPropertyLib and subclasses 

   GAttrList
        a list with attributes like colors, abbreviations 

        provide the singing and dancing add-ons around the persistable GItemList 
        (by pulling from GItemIndex) without persistency 

*/

#include <map>
#include <string>
#include <vector>

class GCache ; 
class GItemList ; 

class GAttrList {
    public:
        GAttrList(GCache* cache, const char* type);
        const char* getType();
        void setColor(std::map<std::string, std::string>& color);
        void setAbbrev(std::map<std::string, std::string>& abbrev);
        void setNames(GItemList* names);
    public:
        std::string  getAbbr(const char* shortname);
        unsigned int getColorCode(const char* key );
        const char*  getColorName(const char* key);
        std::vector<unsigned int>& getColorCodes();
        std::vector<std::string>&  getLabels();
    public:
        void dump(const char* items, const char* msg="GAttrList::dump");
    private:
        GCache*                              m_cache ; 
        const char*                          m_type ; 
        std::map<std::string, std::string>   m_abbrev ;
        std::map<std::string, std::string>   m_color ;
        GItemList*                           m_names ; 
    private:
        std::vector<unsigned int>            m_color_codes ; 
        std::vector<std::string>             m_labels ; 

};

inline GAttrList::GAttrList(GCache* cache, const char* type)
   :
   m_cache(cache),
   m_type(strdup(type)),
   m_names(NULL)
{
}

inline void GAttrList::setColor(std::map<std::string, std::string>& color)
{
    m_color = color ; 
}
inline void GAttrList::setAbbrev(std::map<std::string, std::string>& abbrev)
{
    m_abbrev = abbrev ; 
}
inline void GAttrList::setNames(GItemList* names)
{
    m_names = names ; 
}
inline const char* GAttrList::getType()
{
    return m_type ; 
}



