#include "CFG4_BODY.hh"
#include <cassert>
#include <cstring>
#include <sstream>
#include <iomanip>

//#include <boost/algorithm/string.hpp>
#include "BStr.hh"
#include "SDigest.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4PhysicsOrderedFreeVector.hh"
#include "GProperty.hh"
#include "GAry.hh"


#include "NPY.hpp"
#include "CVec.hh"
#include "CMPT.hh"

#include "PLOG.hh"


CMPT::CMPT(G4MaterialPropertiesTable* mpt, const char* name)
    :
     m_mpt(mpt),
     m_name(name ? strdup(name) : NULL)
{
}


void CMPT::dump(const char* msg) const
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



std::string CMPT::Digest(G4MaterialPropertiesTable* mpt)  
{
    if(!mpt) return "" ; 

    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    SDigest dig ;

    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)  
    {
        G4String pname = it->first ;
        G4MaterialPropertyVector* pvec = it->second ; 
        std::string s = CVec::Digest(pvec) ; 
        dig.update( const_cast<char*>(s.data()), s.size() );  
    }
    return dig.finalize();
}

std::string CMPT::digest() const 
{
    return Digest(m_mpt) ; 
}



std::vector<std::string> CMPT::getPropertyDesc() const
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

        // cf CInputPhotonSource::convertPhoton

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
    //boost::split(keys, _keys, boost::is_any_of(","));   
    BStr::split(keys, _keys, ',');

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


unsigned CMPT::getVecLength(const char* _keys)
{
    std::vector<std::string> keys ; 
    unsigned nkey = splitKeys(keys, _keys);
    unsigned vlen(0) ; 

    for(unsigned i=0 ; i < nkey ; i++ )
    {
        const char* key = keys[i].c_str(); 
        G4PhysicsOrderedFreeVector* v = getVec(key);

        if(vlen == 0)
             vlen = v->GetVectorLength() ;
        else
             assert(vlen == v->GetVectorLength());
    } 
    return vlen ; 
}


unsigned CMPT::splitKeys(std::vector<std::string>& keys, const char* _keys)
{
    //boost::split(keys, _keys, boost::is_any_of(","));   
    BStr::split(keys, _keys, ',');
    return keys.size();
}


NPY<float>* CMPT::makeArray(const char* _keys, bool reverse)
{
    unsigned vlen = getVecLength(_keys);
    std::vector<std::string> keys ; 
    unsigned nkey = splitKeys(keys, _keys);
 
    NPY<float>* vals = NPY<float>::make(nkey, vlen, 4);
    vals->zero();

    for(unsigned i=0 ; i < nkey ; i++ )
    {
        const char* key = keys[i].c_str(); 
        G4PhysicsOrderedFreeVector* v = getVec(key);
        assert( v->GetVectorLength() == vlen );

        for(unsigned j=0 ; j < vlen ; j++)
        {
            unsigned jj = reverse ? vlen - 1 - j : j ; 
            G4double en = v->Energy(jj);
            G4double wl = h_Planck*c_light/en ;
            G4double vl =  (*v)[jj] ;

            float x = en/eV ; 
            float y = wl/nm ;
            float z = vl ; 
            float w = 0.f ; 

            vals->setQuad(i, j, x, y, z, w );
        }
    }
    return vals ; 
}



void CMPT::dumpRaw(const char* _keys)
{
    LOG(fatal) << "CMPT::dumpRaw " <<  _keys ;  

    std::vector<std::string> keys ; 
    unsigned nkey = splitKeys(keys, _keys);

    for(unsigned i=0 ; i < nkey ; i++ )
    {
        const char* key = keys[i].c_str(); 
        G4PhysicsOrderedFreeVector* v = getVec(key);

        unsigned vlen = v->GetVectorLength() ;

        LOG(info) << std::setw(15) << key 
                  << " MinValue " << v->GetMinValue()
                  << " MaxValue " << v->GetMaxValue()
                  << " MaxLowEdgeEnergy " << v->GetMaxLowEdgeEnergy()
                  << " MinLowEdgeEnergy " << v->GetMinLowEdgeEnergy()
                  << " VectorLength " << vlen
                  << " c_light " << c_light 
                  << " c_light*ns/nm " << c_light*ns/nm 
                  ;

        G4double wl_prev(0) ; 

        for(unsigned j=0 ; j < vlen ; j++)
        {
            unsigned i = vlen - 1 - j ; 
            G4double en = v->Energy(i) ;
            G4double wl = h_Planck*c_light/en ;
            G4double vl =  (*v)[i] ;

            LOG(info) << " i " << std::setw(4) << i 
                      << " en(eV) " << std::setw(10) << en/eV
                      << " wl(nm) " << std::setw(10) << wl/nm
                      << " wl - prv " << std::setw(10) << (wl - wl_prev)/nm
                      << " val " << std::setw(10) << vl
                      ;

             wl_prev = wl ; 
        }


    }
}



G4PhysicsOrderedFreeVector* CMPT::getVec(const char* key) const 
{
    G4PhysicsOrderedFreeVector* pofv = NULL ; 
    G4MaterialPropertyVector* mpv = m_mpt->GetProperty(key); 
    if(mpv) pofv = static_cast<G4PhysicsOrderedFreeVector*>(mpv);
    return pofv ; 
}

CVec* CMPT::getCVec(const char* lkey) const
{
    G4PhysicsOrderedFreeVector* vec = getVec(lkey) ;
    return new CVec(vec);
}







void CMPT::sample(NPY<float>* a, unsigned offset, const char* _keys, float low, float step, unsigned nstep )
{
   // CAUTION: used by cfg4/tests/CInterpolationTest.cc 

    std::vector<std::string> keys ; 
    //boost::split(keys, _keys, boost::is_any_of(","));   
    BStr::split(keys, _keys, ',') ; 

    unsigned nkey = keys.size();
    
    unsigned ndim = a->getDimensions() ;
    assert(ndim == 5);
    unsigned nl = a->getShape(3);
    unsigned nm_ = a->getShape(4);  // 4 corresponding to float4 of props used in tex 

    float* values = a->getValues() + offset ;

    assert( nl == nstep );
    assert( nm_ == nkey );

    for(unsigned l=0 ; l < nl ; l++)
    {   
        G4double wavelength = (low + l*step)*CLHEP::nm ;
        G4double photonMomentum = h_Planck*c_light/wavelength ; 

        for(unsigned m=0 ; m < nm_ ; m++)
        {
            const char* key = keys[m].c_str(); 
            G4PhysicsOrderedFreeVector* pofv = getVec(key);
            G4double value = pofv ? pofv->Value( photonMomentum ) : 0.f ;
            *(values + l*nm_ + m) = value ;
        }
    }   
}


/**
 Mapping from G4 props to Opticks detect/absorb/reflect_specular/reflect_diffuse is non-trivial
**/

GProperty<double>* CMPT::makeProperty(const char* key, float low, float step, unsigned nstep)
{
    G4PhysicsOrderedFreeVector* vec = key == NULL ? NULL : getVec(key);

    GAry<double>* dom = new GAry<double>(nstep);
    GAry<double>* val = new GAry<double>(nstep);

    for(unsigned i=0 ; i < nstep ; i++)
    {
        G4double wavelength = (low + i*step)*CLHEP::nm ;
        G4double photonMomentum = h_Planck*c_light/wavelength ; 
        dom->setValue(i, photonMomentum);
        val->setValue(i, vec == NULL ? 0 : vec->Value(photonMomentum) );
    }
    return new GProperty<double>(val, dom);
}



void CMPT::sampleSurf(NPY<float>* a, unsigned offset, float low, float step, unsigned nstep, bool specular)
{
   // used from cfg4/tests/CInterpolationTest.cc

    GProperty<double>* efficiency = makeProperty("EFFICIENCY", low, step, nstep);
    GProperty<double>* reflectivity = makeProperty("REFLECTIVITY", low, step, nstep);

    if(m_name) LOG(info) << m_name 
                         << efficiency->brief(" efficiency")
                         << reflectivity->brief(" reflectivity")
                         ;


    unsigned ndim = a->getDimensions() ;
    assert(ndim == 5);
    float* values = a->getValues() + offset ;

    unsigned nl = a->getShape(3);
    unsigned nm_ = a->getShape(4);  // 4 corresponding to float4 of props used in tex 
    assert( nl == nstep );
    assert( nm_ == 4 );

    bool sensor = !efficiency->isZero() ;


    // compare with GSurfaceLib::createStandardSurface
    // default is to be all zeroes

    GProperty<double>* zero = NULL ; 

    GProperty<double>* _detect = NULL ; 
    GProperty<double>* _absorb = NULL ; 
    GProperty<double>* _specular = NULL ; 
    GProperty<double>* _diffuse = NULL ; 

    if(sensor)
    {
         _detect = efficiency ; 
         _absorb = GProperty<double>::make_one_minus( _detect );
         _specular = zero ; 
         _diffuse = zero ; 
    }
    else
    {
         if(specular)
         {
              _detect = zero ; 
              _absorb  = GProperty<double>::make_one_minus(reflectivity);
              _specular = reflectivity ;
              _diffuse = zero ; 
         }
         else
         {
              _detect = zero ; 
              _absorb  = GProperty<double>::make_one_minus(reflectivity);
              _specular = zero ; 
              _diffuse = reflectivity ; 
         }
    }



    for(unsigned l=0 ; l < nstep ; l++)
    {
        for(unsigned m=0 ; m < nm_ ; m++)
        {
            float value = 0 ; 
            switch(m)
            { 
                case 0:  value = _detect   ? _detect->getValue(l) : 0.f ; break ; 
                case 1:  value = _absorb   ? _absorb->getValue(l) : 0.f ; break ;
                case 2:  value = _specular ? _specular->getValue(l) : 0.f ; break ;
                case 3:  value = _diffuse  ? _diffuse->getValue(l) : 0.f ; break ;
            }
            *(values + l*nm_ + m) = value ;
        }
    }
}

