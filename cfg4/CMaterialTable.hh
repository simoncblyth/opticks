#pragma once

#include <string>
#include <map>

class G4Material ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CMaterialTable {
    public:
         CMaterialTable(const char* prefix="/dd/Materials/");
         void dump(const char* msg="CMaterialMap::dump");
         void fillMaterialIndexMap( std::map<std::string, unsigned>&  mixm );
         std::map<std::string, unsigned>& getMaterialMap();
    public:
         void dumpMaterial(unsigned index);
         unsigned getMaterialIndex(const char* shortname);
         void dumpMaterial(const char* shortname);
         void dumpMaterial(G4Material* material);
    private:
         void init();
         void initNameIndex();
    private:
         const char* m_prefix ; 
         std::map<std::string, unsigned> m_name2index ; 
         std::map<unsigned, std::string> m_index2name ; 

};

#include "CFG4_TAIL.hh"

