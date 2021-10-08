/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "CFG4_BODY.hh"
#include <cassert>
#include <cstring>
#include <sstream>
#include <iomanip>

//#include <boost/algorithm/string.hpp>
#include "BStr.hh"
#include "SDigest.hh"

#include "G4Version.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "X4MaterialPropertiesTable.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4MaterialPropertyVector.hh"
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


void CMPT::Dump_OLD(G4MaterialPropertiesTable* mpt, const char* msg)
{
    LOG(info) << msg ; 
    LOG(info) << "digest : " << CMPT::Digest(mpt) ; 

#if G4VERSION_NUMBER < 110
    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = mpt->GetPropertiesMap() ;

    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4MaterialPropertyVector* pvec = it->second ;
        G4MaterialPropertyVector* pvec2 = mpt->GetProperty(pname.c_str()) ;
        assert( pvec == pvec2 ) ;   
 
        LOG(info) << pname << "\n" << *pvec ; 
    }   

#else
   LOG(info) << "1100 drops G4MaterialPropertiesTable::GetPropertiesMap" ; 
#endif


}


void CMPT::Dump(G4MaterialPropertiesTable* mpt, const char* msg)
{
    LOG(info) << msg ; 
    LOG(info) << "digest : " << CMPT::Digest(mpt) ; 

    typedef G4MaterialPropertyVector MPV ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(debug) << " pns " << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& pname = pns[i]; 
        G4int pidx = mpt->GetPropertyIndex(pname); 
        assert( pidx > -1 );  
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx);  
        MPV* pvec2 = mpt->GetProperty(pname.c_str()) ;
        assert( pvec == pvec2 ) ;   
        if(pvec == NULL) continue ; 

        LOG(info) << pname << "\n" << *pvec ; 
    }   
}


bool CMPT::hasProperty(const char* pname) const
{
    G4MaterialPropertyVector* pvec = m_mpt->GetProperty(pname);
    return pvec != NULL ; 
}


bool CMPT::HasProperty(const G4MaterialPropertiesTable* mpt_, const char* pname) 
{    
    G4MaterialPropertiesTable* mpt = const_cast<G4MaterialPropertiesTable*>(mpt_); 
    G4MaterialPropertyVector* pvec = mpt->GetProperty(pname);
    return pvec != NULL ; 
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

std::vector<std::string> CMPT::getPropertyKeys() const 
{
    std::vector<std::string> keys ; 

    typedef G4MaterialPropertyVector MPV ; 
    std::vector<G4String> pns = m_mpt->GetMaterialPropertyNames() ;
    LOG(debug) << " pns " << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& pname = pns[i]; 
        G4int pidx = m_mpt->GetPropertyIndex(pname); 
        assert( pidx > -1 );  
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(m_mpt)->GetProperty(pidx);  
        if(pvec == NULL) continue ;    

        keys.push_back(pname); 
    }
    return keys ; 
}


void CMPT::addDummyProperty(const char* lkey, unsigned nval)
{
    AddDummyProperty(m_mpt, lkey, nval );
}

void CMPT::AddDummyProperty(G4MaterialPropertiesTable* mpt, const char* lkey, unsigned nval)
{
    G4double* ddom = new G4double[nval] ;   
    G4double* dval = new G4double[nval] ;   
    for(unsigned int j=0 ; j < nval ; j++)
    {   
        ddom[nval-1-j] = j*100. ; 
        dval[nval-1-j] = j*1000. ; 
    }   

#if G4VERSION_NUMBER < 1100
    G4MaterialPropertyVector* mpv = mpt->AddProperty(lkey, ddom, dval, nval); 
#else
    bool exists = X4MaterialPropertiesTable::PropertyExists(mpt, lkey); 
    G4bool createNewKey = exists == false ; 
    G4MaterialPropertyVector* mpv = mpt->AddProperty(lkey, ddom, dval, nval, createNewKey); 
#endif
    assert( mpv ); 

    delete [] ddom ;
    delete [] dval ; 
}


void CMPT::AddConstProperty(G4MaterialPropertiesTable* mpt, const char* lkey, G4double pval)
{
#if G4VERSION_NUMBER < 1100
    mpt->AddConstProperty(lkey, pval); 
#else
    G4String skey(lkey); 

    // 1st try: nope throws exception for non-existing key 
    //G4int keyIdx = mpt->GetConstPropertyIndex(skey);  
    //G4bool createNewKey = keyIdx == -1 ; 

    // 2nd try: nope again throws exception from GetConstPropertyIndex just like above
    //G4bool exists = mpt->ConstPropertyExists(lkey); 
    //G4bool createNewKey = !exists ; 
   
    // 3rd try 
    bool exists = X4MaterialPropertiesTable::ConstPropertyExists(mpt, lkey); 
    G4bool createNewKey = !exists ; 

    mpt->AddConstProperty(lkey, pval, createNewKey); 
#endif
}





CMPT* CMPT::MakeDummy()
{
    G4MaterialPropertiesTable* _mpt = new G4MaterialPropertiesTable(); 
    CMPT* mpt = new CMPT(_mpt) ; 

    mpt->addDummyProperty("A", 5 ); 
    mpt->addDummyProperty("B", 10 ); 

    return mpt ; 
}

std::string CMPT::Digest(G4MaterialPropertiesTable* mpt)  
{
    if(!mpt) return "" ; 

    SDigest dig ;
    typedef G4MaterialPropertyVector MPV ;

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(debug) << " pns " << pns.size() ;
    for( unsigned i=0 ; i < pns.size() ; i++)
    {
        const std::string& n = pns[i];
        G4int pidx = mpt->GetPropertyIndex(n);
        assert( pidx > -1 );
        MPV* v = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx);
        if(v == NULL) continue ;

        std::string vs = CVec::Digest(v) ; 
        dig.update( const_cast<char*>(n.data()),  n.size() );
        dig.update( const_cast<char*>(vs.data()), vs.size() );
    }
    return dig.finalize();
}







std::string CMPT::digest() const 
{
    return Digest(m_mpt) ; 
}


std::vector<std::string> CMPT::getPropertyDesc() const
{
    std::vector<std::string> desc ; 

    typedef G4MaterialPropertyVector MPV ; 
    std::vector<G4String> pns = m_mpt->GetMaterialPropertyNames() ;
    LOG(debug) << " pns " << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& pname = pns[i]; 
        G4int pidx = m_mpt->GetPropertyIndex(pname); 
        assert( pidx > -1 );  
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(m_mpt)->GetProperty(pidx);  
        if(pvec == NULL) continue ;    
 
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





std::vector<std::string> CMPT::getConstPropertyKeys() const 
{
    std::vector<std::string> keys ; 

    std::vector<G4String> cpns = m_mpt->GetMaterialConstPropertyNames() ;
    LOG(debug) << " cpns " << cpns.size() ; 

    for( unsigned i=0 ; i < cpns.size() ; i++)
    {   
        const std::string& pname = cpns[i]; 
        G4bool exists = m_mpt->ConstPropertyExists( pname.c_str() ) ; 
        if(!exists) continue ; 
        keys.push_back(pname) ; 
    }
    return keys ; 
}

 
std::vector<double> CMPT::getConstPropertyValues() const 
{
    std::vector<double> vals ; 

    std::vector<G4String> cpns = m_mpt->GetMaterialConstPropertyNames() ;
    LOG(debug) << " cpns " << cpns.size() ; 

    for( unsigned i=0 ; i < cpns.size() ; i++)
    {   
        const std::string& pname = cpns[i]; 
        G4bool exists = m_mpt->ConstPropertyExists( pname.c_str() ) ; 
        if(!exists) continue ; 

        G4int pidx = m_mpt->GetConstPropertyIndex(pname); 
        G4double pval = m_mpt->GetConstProperty(pidx);  

        vals.push_back(pval); 
    }

    return vals ; 
}







/**
CMPT::addProperty
------------------

Adds an Opticks GProperty to a Geant4 MPT doing the 
wavelength to energy swap. 

Spline argument is requires to be false, 
see issue/optical_local_time_goes_backward.rst

**/

void CMPT::addProperty(const char* lkey,  GProperty<double>* prop, bool spline)
{
    assert( spline == false ); 
    unsigned int nval  = prop->getLength();

    LOG(debug) << "CMPT::addProperty" 
               << " lkey " << std::setw(40) << lkey
               << " nval " << std::setw(10) << nval
               ;   

    G4double* ddom = new G4double[nval] ;
    G4double* dval = new G4double[nval] ;

    for(unsigned int j=0 ; j < nval ; j++)
    {
        double fnm = prop->getDomain(j) ;
        double fval = prop->getValue(j) ; 

        // cf CInputPhotonSource::convertPhoton

        G4double wavelength = G4double(fnm)*nm ; 
        G4double energy = h_Planck*c_light/wavelength ;

        G4double value = G4double(fval) ;

        ddom[nval-1-j] = G4double(energy) ;   // reverse wavelength order to give increasing energy order
        dval[nval-1-j] = G4double(value) ;
    }

    G4MaterialPropertyVector* mpv = m_mpt->AddProperty(lkey, ddom, dval, nval);
    assert( mpv ); 

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

    std::vector<G4MaterialPropertyVector*> vecs ; 
    for(unsigned i=0 ; i < nkey ; i++ )
    {
        const char* key = keys[i].c_str(); 
        G4MaterialPropertyVector* mpv = m_mpt->GetProperty(key); 
        G4MaterialPropertyVector* pofv = static_cast<G4MaterialPropertyVector*>(mpv);
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
              G4MaterialPropertyVector* pofv = vecs[i] ;
              G4double value = pofv->Value( photonMomentum );
              std::cout  << std::setw(15) << value ;
         }
         std::cout << std::endl ; 
    }
}


unsigned CMPT::getVecLength(const char* _keys)
{
    // asserts that all the keys vectors are of the same length 

    std::vector<std::string> keys ; 
    unsigned nkey = splitKeys(keys, _keys);
    unsigned vlen(0) ; 

    for(unsigned i=0 ; i < nkey ; i++ )
    {
        const char* key = keys[i].c_str(); 
        G4MaterialPropertyVector* v = getVec(key);

        if(vlen == 0)
        {
             vlen = v->GetVectorLength() ;
        }
        else
        {
             assert(vlen == v->GetVectorLength());
        } 
    } 
    return vlen ; 
}


unsigned CMPT::splitKeys(std::vector<std::string>& keys, const char* _keys)
{
    BStr::split(keys, _keys, ',');
    return keys.size();
}


NPY<double>* CMPT::makeArray(const char* _keys, bool reverse)
{
    unsigned vlen = getVecLength(_keys);
    std::vector<std::string> keys ; 
    unsigned nkey = splitKeys(keys, _keys);
 
    NPY<double>* vals = NPY<double>::make(nkey, vlen, 4);
    vals->zero();

    for(unsigned i=0 ; i < nkey ; i++ )
    {
        const char* key = keys[i].c_str(); 
        G4MaterialPropertyVector* v = getVec(key);
        assert( v->GetVectorLength() == vlen );

        for(unsigned j=0 ; j < vlen ; j++)
        {
            unsigned jj = reverse ? vlen - 1 - j : j ; 
            G4double en = v->Energy(jj);
            G4double wl = h_Planck*c_light/en ;
            G4double vl =  (*v)[jj] ;

            double x = en/eV ; 
            double y = wl/nm ;
            double z = vl ; 
            double w = 0.f ; 

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
        G4MaterialPropertyVector* v = getVec(key);

        unsigned vlen = v->GetVectorLength() ;

        LOG(info) << std::setw(15) << key 
                  << " MinValue " << v->GetMinValue()
                  << " MaxValue " << v->GetMaxValue()
                  << " MaxLowEdgeEnergy " << v->Energy(v->GetVectorLength()-1)
                  << " MinLowEdgeEnergy " << v->Energy(0)
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


/**
CMPT::getVec
-------------

Formerly this tickled a 91072 bug, see G4MaterialPropertiesTableTest::test_GetProperty_NonExisting 
Avoid that issue by checking the property index. 

Subsequently Geant4 changes G4MaterialPropertiesTable to throw exceptions for non-existing keys

**/

G4MaterialPropertyVector* CMPT::getVec(const char* key) const 
{
    int index = X4MaterialPropertiesTable::GetPropertyIndex(m_mpt, key) ;   // this avoids ticking 91072 bug when the key is non-existing
    G4MaterialPropertyVector* mpv = index < 0 ? nullptr : m_mpt->GetProperty(index); 
    return mpv ; 
}

CVec* CMPT::getCVec(const char* lkey) const
{
    G4MaterialPropertyVector* vec = getVec(lkey) ;
    return new CVec(vec);
}







void CMPT::sample(NPY<double>* a, unsigned offset, const char* _keys, double low, double step, unsigned nstep )
{
   // CAUTION: used by cfg4/tests/CInterpolationTest.cc 

    std::vector<std::string> keys ; 
    BStr::split(keys, _keys, ',') ; 

    unsigned nkey = keys.size();
    
    unsigned ndim = a->getDimensions() ;
    assert(ndim == 5);
    unsigned nl = a->getShape(3);
    unsigned nm_ = a->getShape(4);  // 4 corresponding to double4 of props used in tex 

    double* values = a->getValues() + offset ;


    assert( nl == nstep );

    if( nm_ != nkey )
    {
        LOG(fatal) << " unexpected _keys " << _keys 
                   << " nkey " << nkey 
                   << " nm_ " << nm_
                   << " a " << a->getShapeString()
                   ;
    }
    assert( nm_ == nkey );

    for(unsigned l=0 ; l < nl ; l++)
    {   
        G4double wavelength = (low + l*step)*CLHEP::nm ;
        G4double photonMomentum = h_Planck*c_light/wavelength ; 

        for(unsigned m=0 ; m < nm_ ; m++)
        {
            const char* key = keys[m].c_str(); 
            G4MaterialPropertyVector* pofv = getVec(key);
            G4double value = pofv ? pofv->Value( photonMomentum ) : 0.f ;
            *(values + l*nm_ + m) = value ;
        }
    }   
}


/**
 Mapping from G4 props to Opticks detect/absorb/reflect_specular/reflect_diffuse is non-trivial
**/

GProperty<double>* CMPT::makeProperty(const char* key, double low, double step, unsigned nstep)
{
    G4MaterialPropertyVector* vec = key == NULL ? NULL : getVec(key);

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



void CMPT::sampleSurf(NPY<double>* a, unsigned offset, double low, double step, unsigned nstep, bool specular)
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
    double* values = a->getValues() + offset ;

    unsigned nl = a->getShape(3);
    unsigned nm_ = a->getShape(4);  // 4 corresponding to double4 of props used in tex 
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
            double value = 0 ; 
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



