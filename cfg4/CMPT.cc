#include "CFG4_BODY.hh"
#include <cassert>
#include <sstream>
#include <iomanip>

#include <boost/algorithm/string.hpp>

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4PhysicsOrderedFreeVector.hh"
#include "GProperty.hh"


#include "NPY.hpp"
#include "CMPT.hh"

#include "PLOG.hh"


CMPT::CMPT(G4MaterialPropertiesTable* mpt)
    :
     m_mpt(mpt)
{
}


void CMPT::dump(const char* msg)
{
    LOG(info) << msg ;
    std::vector<std::string> pdesc = getPropertyDesc() ;

    for(unsigned int i=0 ; i < pdesc.size() ; i++) 
        std::cout << pdesc[i] << std::endl ; 

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



void CMPT::dumpProperty(const char* _keys)
{
    LOG(info) << _keys ;  

    std::vector<std::string> keys ; 
    boost::split(keys, _keys, boost::is_any_of(","));   
    unsigned nkey = keys.size();

    std::vector<G4PhysicsOrderedFreeVector*> vecs ; 
    for(unsigned i=0 ; i < nkey ; i++ )
    {
        const char* key = keys[i].c_str(); 
        G4MaterialPropertyVector* mpv = m_mpt->GetProperty(key); 
        G4PhysicsOrderedFreeVector* pofv = static_cast<G4PhysicsOrderedFreeVector*>(mpv);
        vecs.push_back(pofv);
    }
 
    std::cout << std::setw(5) << "nm" << std::setw(15) << "eV" ; 
    for(unsigned i=0 ; i < nkey ; i++ ) std::cout << std::setw(15) << keys[i].c_str()  ;
    std::cout << std::endl ; 

    for(unsigned wl=60 ; wl < 821 ; wl+= 20)
    {
         G4double wavelength = wl*nm ; 
         G4double photonMomentum = h_Planck*c_light/wavelength ; 
         std::cout 
                   << std::setw(5) << wl
                   << std::setw(15) << photonMomentum/eV   
                   ;

         for(unsigned i=0 ; i < nkey ; i++)
         {
              G4PhysicsOrderedFreeVector* pofv = vecs[i] ;
              G4double value = pofv->Value( photonMomentum );
              std::cout  << std::setw(15) << value ;
         }
         std::cout << std::endl ; 
    }
}



void CMPT::sample(NPY<float>* a, unsigned offset, const char* _keys, float low, float step, unsigned nstep )
{

    std::vector<std::string> keys ; 
    boost::split(keys, _keys, boost::is_any_of(","));   
    unsigned nkey = keys.size();
    std::vector<G4PhysicsOrderedFreeVector*> vecs ; 
    for(unsigned i=0 ; i < nkey ; i++ )
    {
        const char* key = keys[i].c_str(); 
        if(strlen(key) == 0) continue ; 

        G4MaterialPropertyVector* mpv = m_mpt->GetProperty(key); 
        if(!mpv) LOG(fatal) << " missing mpv for key " << key ; 
        assert(mpv);

        G4PhysicsOrderedFreeVector* pofv = static_cast<G4PhysicsOrderedFreeVector*>(mpv);
        vecs.push_back(pofv);
    }
     
    unsigned ndim = a->getDimensions() ;
    assert(ndim == 5);
    unsigned nl = a->getShape(3);
    unsigned nm = a->getShape(4);

    float* values = a->getValues() + offset ;

    assert( nl == nstep );
    assert( nm == nkey );

    for(unsigned l=0 ; l < nl ; l++)
    {   
        G4double wavelength = (low + l*step)*CLHEP::nm ;
        G4double photonMomentum = h_Planck*c_light/wavelength ; 

        for(unsigned m=0 ; m < nm ; m++)
        {
            if(m >= vecs.size()) break ; 

            G4PhysicsOrderedFreeVector* pofv = vecs[m] ;
            G4double value = pofv->Value( photonMomentum );

            *(values + l*nm + m) = value ;

        }
    }   
}


