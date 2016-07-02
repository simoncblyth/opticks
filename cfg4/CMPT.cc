#include "CFG4_BODY.hh"
#include <cassert>
#include <sstream>
#include <iomanip>

#include "G4MaterialPropertiesTable.hh"

#include "CMPT.hh"

CMPT::CMPT(G4MaterialPropertiesTable* mpt)
    :
     m_mpt(mpt)
{
}


std::string CMPT::description(const char* msg)
{
   std::vector<std::string> pkeys = getPropertyKeys() ;
   std::vector<std::string> pdesc = getPropertyDesc() ;
   std::vector<std::string> ckeys = getConstPropertyKeys() ;
   std::vector<double>      cvals = getConstPropertyValues() ;
   
   assert(pkeys.size() == pdesc.size());
   assert(ckeys.size() == cvals.size());

   std::stringstream ss ; 
   
   ss << std::setw(30) << msg ;
   ss << " props " << pkeys.size() << " " ; 
   for(unsigned int i=0 ; i < pdesc.size() ; i++) ss << pdesc[i] << "," ;
   ss << " " ; 

   ss << " cprops " << ckeys.size() << " " ; 
   for(unsigned int i=0 ; i < ckeys.size() ; i++) ss << ckeys[i] << ":" << cvals[i] << " " ;
   ss << " " ; 

   return ss.str();
}

std::vector<std::string> CMPT::getPropertyKeys()
{
    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = m_mpt->GetPropertiesMap() ;
    std::vector<std::string> keys ; 
    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)  keys.push_back(it->first) ;
    return keys ; 
}

std::vector<std::string> CMPT::getPropertyDesc()
{
    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = m_mpt->GetPropertiesMap() ;
    std::vector<std::string> desc ; 
    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)  
    {
        G4String pname = it->first ;
        G4MaterialPropertyVector* pvec = it->second ; 


        double pmin = pvec->GetMinValue() ;
        double pmax = pvec->GetMaxValue() ;

        std::stringstream ss ; 
        ss << pname << ":"  ;

        if(pmin == pmax)
           ss << ":" << pmin ;
        else 
           ss << ":" << pmin
              << ":" << pmax
              << ":" << pvec->GetMaxEnergy()  
           // << ":" << pvec->GetVectorLength()
               ;

        desc.push_back(ss.str()) ;
    }
    return desc ; 
}



std::vector<std::string> CMPT::getConstPropertyKeys()
{
    typedef const std::map< G4String, G4double, std::less<G4String> > MKC ; 
    MKC* cm = m_mpt->GetPropertiesCMap() ;
    std::vector<std::string> keys ; 
    for(MKC::const_iterator it=cm->begin() ; it != cm->end() ; it++)  keys.push_back(it->first) ;
    return keys ; 
}

std::vector<double> CMPT::getConstPropertyValues()
{
    typedef const std::map< G4String, G4double, std::less<G4String> > MKC ; 
    MKC* cm = m_mpt->GetPropertiesCMap() ;
    std::vector<double> vals ; 
    for(MKC::const_iterator it=cm->begin() ; it != cm->end() ; it++)  vals.push_back(it->second) ;
    return vals ; 
}




