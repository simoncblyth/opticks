#include "CFG4_BODY.hh"
#include <cassert>
#include <sstream>
#include <iomanip>

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4MaterialPropertiesTable.hh"
#include "GProperty.hh"

#include "CMPT.hh"

#include "PLOG.hh"


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




void CMPT::addProperty(const char* lkey,  GProperty<float>* prop, bool spline)
{
    unsigned int nval  = prop->getLength();

    LOG(debug) << "CMPT::addProperty" 
               << " lkey " << std::setw(40) << lkey
               << " nval " << std::setw(10) << nval
               ;   

    G4double* ddom = new G4double[nval] ;
    G4double* dval = new G4double[nval] ;

    for(unsigned int j=0 ; j < nval ; j++)
    {
        float fnm = prop->getDomain(j) ;
        float fval = prop->getValue(j) ; 

        G4double wavelength = G4double(fnm)*nm ; 
        G4double energy = h_Planck*c_light/wavelength ;

        G4double value = G4double(fval) ;

        ddom[nval-1-j] = G4double(energy) ;   // reverse wavelength order to give increasing energy order
        dval[nval-1-j] = G4double(value) ;
    }

    G4MaterialPropertyVector* mpv = m_mpt->AddProperty(lkey, ddom, dval, nval);
    mpv->SetSpline(spline);
    // see issue/optical_local_time_goes_backward.rst

    delete [] ddom ; 
    delete [] dval ; 
}









