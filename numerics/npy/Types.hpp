#pragma once

#include <string>
#include <map>
#include <vector>
#include "string.h"

#include "glm/fwd.hpp"

class Index ; 

/*

TODO: rejig material codes just plucking most common as index 1-15, and using code 0 for "Other" 
      this is needed to fit into 4 bits per material in sequence machinery  

[2015-06-30 20:36:21.012404] [0x000007fff79cb931] [info]    SequenceNPY::countMaterials  m1/m2 codes in all records 5187214 unique material codes 16
    0   13   2084363            Acrylic .
    1   12    839627         MineralOil .
    2   15    714965          GdDopedLS .
    3   14    683857 LiquidScintillator .
    4   11    273739     StainlessSteel .
    5   10    265339           IwsWater .
    6   18     89379 UnstStainlessSteel .
    7    3     63342                Air .
    8   16     56527              Pyrex .
    9    1     52492             Vacuum .
   10   19     28483                ESR .
   11    2     16029               Rock .
   12   17     13187           Bialkali .
   13   20      4466              Water .
   14   21      1288           Nitrogen .
   15   24       131                PVC .

*/

// TODO: split mechanics into a base class 
//       with specializations for History and Material
//
class Types {
   public:
       static const char* TAIL ; 

       static const char* HISTORY_ ; 
       static const char* MATERIAL_ ; 
       static const char* HISTORYSEQ_ ; 
       static const char* MATERIALSEQ_ ; 

       typedef enum { HISTORY, MATERIAL, HISTORYSEQ, MATERIALSEQ } Item_t ;

   public:
       Types(); 

       void         setTail(const char* tail);
       const char*  getTail();
       void         setAbbrev(bool abbrev);

   public:
       Index*                    getFlagsIndex();
       Index*                    getMaterialsIndex();
       void                      setMaterialsIndex(Index* index);
   public:
       std::string getMaskString(unsigned int mask, Item_t etype);
       std::string getMaterialString(unsigned int flags);
       std::string getHistoryString(unsigned int flags);
       unsigned int getHistoryFlag(std::string label);
       unsigned int getMaterialCode(std::string label);
       std::string getSequenceString(unsigned long long seq);
       const char* getItemName(Item_t item);

   public:
       void getMaterialStringTest();
       void readMaterialsOld(const char* idpath, const char* name="GMaterialIndexLocal.json");    
       void readMaterials(const char* idpath, const char* name="GMaterialIndex");    
       void dumpMaterials(const char* msg="Types::dumpMaterials");
       std::string findMaterialName(unsigned int index);
   private:
       void makeMaterialAbbrev();

   public:
       std::string getStepFlagString(unsigned char flag);
       void getHistoryStringTest();

       void readFlags(const char* path); // parse enum flags from photon.h
       void dumpFlags(const char* msg="Types::dumpFlags");
   private:
       void makeFlagAbbrev();
   public:
       glm::ivec4                getFlags();
       std::vector<std::string>& getFlagLabels();
       bool*                     initBooleanSelection(unsigned int n);
       bool*                     getFlagSelection();
   public:
       std::string getHistoryAbbrev(std::string label);
       std::string getMaterialAbbrev(std::string label);
       std::string getAbbrev(std::string label, Item_t etype);
   public:
       std::string getHistoryAbbrevInvert(std::string label, bool hex=false);
       std::string getMaterialAbbrevInvert(std::string label, bool hex=false);
       std::string getAbbrevInvert(std::string label, Item_t etype, bool hex=false);
       unsigned int getHistoryAbbrevInvertAsCode(std::string label, bool hex=false);
       unsigned int getMaterialAbbrevInvertAsCode(std::string label, bool hex=false);
       unsigned int getAbbrevInvertAsCode(std::string label, Item_t etype, bool hex=false);

   public:
       // formerly in RecordsNPY
       unsigned long long convertSequenceString(std::string& seq, Item_t etype, bool hex=false);
       void prepSequenceString(std::string& lseq, unsigned int& elen, unsigned int& nelem, bool hex);
       std::string decodeSequenceString(std::string& seq, Item_t etype, bool hex=false);

       std::string abbreviateHexSequenceString(std::string& seq, Item_t etype);

   private:
       Index*                                               m_materials_index ; 
       std::map<std::string, unsigned int>                  m_materials ;
       std::map<std::string, std::string>                   m_material2abbrev ; 
       std::map<std::string, std::string>                   m_abbrev2material ; 

   private:
       Index*                                               m_flags ; 
       std::vector<std::string>                             m_flag_labels ; 
       std::vector<unsigned int>                            m_flag_codes ; 
       bool*                                                m_flag_selection ; 
   private:
       std::map<std::string, std::string>                   m_flag2abbrev ; 
       std::map<std::string, std::string>                   m_abbrev2flag ; 

   private:
       const char*  m_tail ;
       bool         m_abbrev ; 

};


inline Types::Types() :
     m_materials_index(NULL),
     m_flags(NULL),
     m_tail(TAIL),
     m_abbrev(false)
{
};


inline std::vector<std::string>& Types::getFlagLabels()
{
     return m_flag_labels ; 
}
inline bool* Types::getFlagSelection()
{
     return m_flag_selection ; 
}
inline Index* Types::getFlagsIndex()
{
     return m_flags ; 
}


inline void Types::setTail(const char* tail)
{
     m_tail = strdup(tail);
}
inline void Types::setAbbrev(bool abbrev)
{
     m_abbrev = abbrev ;
}
inline const char* Types::getTail()
{
    return m_tail ; 
}



inline Index* Types::getMaterialsIndex()
{
    return m_materials_index ; 
}

