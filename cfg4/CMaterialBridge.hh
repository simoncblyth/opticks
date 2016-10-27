#pragma once

#include <string>
#include <map>
class GMaterialLib ;  

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class G4Material ; 

class CFG4_API CMaterialBridge 
{
    public:
        CMaterialBridge( GMaterialLib* mlib );

        unsigned getMaterialIndex(const G4Material* mat); // G4Material instance to 0-based Opticks material index
        const char* getMaterialName(unsigned int index);  // 0-based Opticks material index to shortname
        const G4Material* getG4Material(unsigned int index); // 0-based Opticks material index to G4Material

        std::string MaterialSequence(unsigned long long seqmat);

        void dump(const char* msg="CMaterialBridge::dump");
        void dumpMap(const char* msg="CMaterialBridge::dumpMap");
        bool operator()(const G4Material* a, const G4Material* b);
    private:
        void initMap();
    private:
        GMaterialLib*   m_mlib ; 
        std::map<const G4Material*, unsigned> m_g4toix ; 
        std::map<unsigned int, std::string>   m_ixtoname ; 

};

#include "CFG4_TAIL.hh"

