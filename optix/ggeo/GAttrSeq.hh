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
#include <cstring>
#include <string>
#include <vector>

class OpticksResource ;

class GCache ; 
class NSequence ; 
class Index ; 

/*
Classes fulfilling NSequence include GItemList 
*/
class GAttrSeq {
    public:
        static unsigned int ERROR_COLOR ; 

        enum {
                REVERSE    = 0x1 << 0,  
                ABBREVIATE = 0x1 << 1,
                ONEBASED   = 0x1 << 2,
                HEXKEY     = 0x1 << 3
             };

        enum {
                 SEQUENCE_DEFAULTS = REVERSE|ABBREVIATE|ONEBASED|HEXKEY,
                 VALUE_DEFAULTS = ABBREVIATE|ONEBASED
             };

    public:
        GAttrSeq(GCache* cache, const char* type);
        void setCtrl(unsigned char ctrl);
        void loadPrefs();
        const char* getType();

        std::map<std::string, unsigned int>& getOrder();
        //std::map<std::string, std::string>& getColor();
        //std::map<std::string, std::string>& getAbbrev();

        void setSequence(NSequence* seq);
        bool hasSequence();
    public:
        std::string  getLabel(Index* index, const char* key, unsigned int& colorcode);
        std::string  getAbbr(const char* key);
        unsigned int getColorCode(const char* key );
        const char*  getColorName(const char* key);
        std::vector<unsigned int>& getColorCodes();
        std::vector<std::string>&  getLabels();
    public:
        std::map<unsigned int, std::string> getNamesMap(unsigned char ctrl=ONEBASED);
    public:
        std::string decodeHexSequenceString(const char* seq, unsigned char ctrl=REVERSE|ABBREVIATE|ONEBASED );
        std::string decodeString(const char* seq, unsigned char ctrl=ABBREVIATE|ONEBASED);
    public:
        void dump(const char* keys=NULL, const char* msg="GAttrSeq::dump");
        void dumpKey(const char* key);
    public:
        void dumpTable(   Index* table, const char* msg="GAttrSeq::dumpTable");
    private:
        void init();
    private:
        GCache*                              m_cache ; 
        OpticksResource*                     m_resource ; 
        const char*                          m_type ; 
        unsigned char                        m_ctrl ; 
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
   m_resource(NULL),
   m_type(strdup(type)),
   m_ctrl(0),
   m_sequence(NULL)
{
   init();
}

inline const char* GAttrSeq::getType()
{
    return m_type ; 
}

inline std::map<std::string, unsigned int>&  GAttrSeq::getOrder()
{
    return m_order ;
}

inline void GAttrSeq::setCtrl(unsigned char ctrl)
{
    m_ctrl = ctrl ; 
}

inline bool GAttrSeq::hasSequence()
{
    return m_sequence != NULL ; 
}

