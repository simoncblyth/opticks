#pragma once

#include <string>
#include <map>

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CMaterialTable {
    public:
         CMaterialTable(const char* prefix="/dd/Materials/");
         void dump(const char* msg="CMaterialMap::dump");
         void fillMaterialIndexMap( std::map<std::string, unsigned>&  mixm );
    private:
         void init();
    private:
         const char* m_prefix ; 
         std::map<std::string, unsigned> m_name2index ; 
         std::map<unsigned, std::string> m_index2name ; 

};

#include "CFG4_TAIL.hh"

