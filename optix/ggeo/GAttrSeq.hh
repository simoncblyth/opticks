#pragma once

/*
History 

   GItemIndex 
        became overcomplicated to compensate for lack of persistable GMaterialLib, GSurfaceLib
        (GBoundaryLib attempted to skip a few steps and live without these)

   GItemList
        simple list of strings, used by GPropertyLib and subclasses 

   GAttrSeq
        a list with attributes like colors, abbreviations 

        provide the singing and dancing add-ons around the persistable GItemList 
        (by pulling from GItemIndex) without persistency 

*/

#include <map>
#include <string>
#include <vector>

class GCache ; 
class NSequence ; 

/*
Classes fulfilling NSequence include GItemList 
*/
class GAttrSeq {
    public:
        static unsigned int ERROR_COLOR ; 
    public:
        GAttrSeq(GCache* cache, const char* type);
        void loadPrefs();
        const char* getType();

        std::map<std::string, unsigned int>& getOrder();
        //std::map<std::string, std::string>& getColor();
        //std::map<std::string, std::string>& getAbbrev();

        void setSequence(NSequence* seq);
    public:
        std::string  getAbbr(const char* key);
        unsigned int getColorCode(const char* key );
        const char*  getColorName(const char* key);
        std::vector<unsigned int>& getColorCodes();
        std::vector<std::string>&  getLabels();
    public:
        void dump(const char* keys=NULL, const char* msg="GAttrSeq::dump");
        void dumpKey(const char* key);
    private:
        GCache*                              m_cache ; 
        const char*                          m_type ; 
        NSequence*                           m_sequence ; 
    private:
        std::map<std::string, std::string>   m_abbrev ;
        std::map<std::string, std::string>   m_color ;
        std::map<std::string, unsigned int>  m_order ;
    private:
        std::vector<unsigned int>            m_color_codes ; 
        std::vector<std::string>             m_labels ; 

};

inline GAttrSeq::GAttrSeq(GCache* cache, const char* type)
   :
   m_cache(cache),
   m_type(strdup(type)),
   m_sequence(NULL)
{
}

/*
inline std::map<std::string, std::string>&  GAttrSeq::getColor()
{
   return m_color ;
}
inline std::map<std::string, std::string>&  GAttrSeq::getAbbrev()
{
   return m_abbrev ;
}
*/

inline std::map<std::string, unsigned int>&  GAttrSeq::getOrder()
{
   return m_order ;
}







inline const char* GAttrSeq::getType()
{
    return m_type ; 
}



