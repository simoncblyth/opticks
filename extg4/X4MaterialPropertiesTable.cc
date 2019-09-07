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


#include "G4MaterialPropertiesTable.hh"

#include "X4MaterialPropertiesTable.hh"
#include "X4PhysicsVector.hh"

#include "SDigest.hh"
#include "GPropertyMap.hh"
#include "GProperty.hh"
#include "PLOG.hh"


void X4MaterialPropertiesTable::Convert( GPropertyMap<float>* pmap,  const G4MaterialPropertiesTable* const mpt )
{
    X4MaterialPropertiesTable xtab(pmap, mpt);
}

X4MaterialPropertiesTable::X4MaterialPropertiesTable( GPropertyMap<float>* pmap,  const G4MaterialPropertiesTable* const mpt )
    :
    m_pmap(pmap),
    m_mpt(mpt)
{
    init();
}

void X4MaterialPropertiesTable::init()
{ 
    AddProperties( m_pmap, m_mpt );    
}


void X4MaterialPropertiesTable::AddProperties(GPropertyMap<float>* pmap, const G4MaterialPropertiesTable* const mpt)   // static
{
    typedef G4MaterialPropertyVector MPV ; 
    G4bool warning ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(debug) << " pns " << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& pname = pns[i]; 
        G4int pidx = mpt->GetPropertyIndex(pname, warning=true); 
        assert( pidx > -1 );  
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning=false );  
        if(pvec == NULL) continue ; 

        LOG(debug)
            << " pname : " 
            << std::setw(30) << pname  
            << " pidx : " 
            << std::setw(5) << pidx 
            << " pvec : "
            << std::setw(16) << pvec 
            ;   

        GProperty<float>* prop = X4PhysicsVector<float>::Convert(pvec) ; 
        pmap->addPropertyStandardized( pname.c_str(), prop );  // interpolates onto standard domain 
    }


    std::vector<G4String> cpns = mpt->GetMaterialConstPropertyNames() ;
    LOG(debug) << " cpns " << cpns.size() ; 

    for( unsigned i=0 ; i < cpns.size() ; i++)
    {   
        const std::string& pname = cpns[i]; 
        G4bool exists = mpt->ConstPropertyExists( pname.c_str() ) ;
        if(!exists) continue ; 

        G4int pidx = mpt->GetConstPropertyIndex(pname, warning=true); 
        assert( pidx > -1 );  
        G4double pval = mpt->GetConstProperty(pidx);  

        LOG(debug)
            << " pname : " 
            << std::setw(30) << pname  
            << " pidx : " 
            << std::setw(5) << pidx 
            << " pval : "
            << std::setw(16) << pval 
            ;   

        pmap->addConstantProperty( pname.c_str(), pval );   // asserts without standard domain
    }
}



std::string X4MaterialPropertiesTable::Digest(const G4MaterialPropertiesTable* mpt)  // static
{
    if(!mpt) return "" ; 

    SDigest dig ;

    typedef G4MaterialPropertyVector MPV ; 
    G4bool warning ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(debug) << " pns " << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& n = pns[i]; 
        G4int pidx = mpt->GetPropertyIndex(n, warning=true); 
        assert( pidx > -1 );  
        MPV* v = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning=false );  
        if(v == NULL) continue ; 

        std::string vs = X4PhysicsVector<float>::Digest(v) ; 
        dig.update( const_cast<char*>(n.data()),  n.size() );  
        dig.update( const_cast<char*>(vs.data()), vs.size() );  
    }

    std::vector<G4String> cpns = mpt->GetMaterialConstPropertyNames() ;
    LOG(debug) << " cpns " << cpns.size() ; 

    for( unsigned i=0 ; i < cpns.size() ; i++)
    {   
        const std::string& n = cpns[i]; 
        G4bool exists = mpt->ConstPropertyExists( n.c_str() ) ;
        if(!exists) continue ; 

        G4int pidx = mpt->GetConstPropertyIndex(n, warning=true); 
        assert( pidx > -1 );  
        G4double pvalue = mpt->GetConstProperty(pidx);  

        dig.update( const_cast<char*>(n.data()), n.size() );  
        dig.update( reinterpret_cast<char*>(&pvalue), sizeof(double) );  
    }
    return dig.finalize();
}



void X4MaterialPropertiesTable::AddProperties_OLD(GPropertyMap<float>* pmap, const G4MaterialPropertiesTable* const mpt_)   // static
{
    G4MaterialPropertiesTable* mpt = const_cast<G4MaterialPropertiesTable*>(mpt_);   // needed with 10.4.2

    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = mpt->GetPropertiesMap() ;

    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)
    {   
        G4String pname = it->first ;

        G4MaterialPropertyVector* pvec = it->second ;  
        // G4MaterialPropertyVector is typedef to G4PhysicsOrderedFreeVector with most of imp in G4PhysicsVector

        GProperty<float>* prop = X4PhysicsVector<float>::Convert(pvec) ; 


      //   pmap->addProperty( pname.c_str(), prop );  // non-interpolating collection
        pmap->addPropertyStandardized( pname.c_str(), prop );  // interpolates onto standard domain 
    }  

    typedef const std::map< G4String, G4double, std::less<G4String> > CKP ; 
    CKP* cm = mpt->GetPropertiesCMap();

    for(CKP::const_iterator it=cm->begin() ; it != cm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4double pvalue = it->second ;  
        float value = pvalue ; 

        pmap->addConstantProperty( pname.c_str(), value );   // asserts without standard domain
    }     
}



std::string X4MaterialPropertiesTable::Digest_OLD(const G4MaterialPropertiesTable* mpt_)  // static
{
    G4MaterialPropertiesTable* mpt = const_cast<G4MaterialPropertiesTable*>(mpt_);   // needed with 10.4.2
    if(!mpt) return "" ; 

    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = mpt->GetPropertiesMap() ;

    SDigest dig ;
    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)  
    {   
        const std::string&  n = it->first ;
        G4MaterialPropertyVector* v = it->second ; 

        std::string vs = X4PhysicsVector<float>::Digest(v) ; 
        dig.update( const_cast<char*>(n.data()),  n.size() );  
        dig.update( const_cast<char*>(vs.data()), vs.size() );  
    }   

    typedef const std::map< G4String, G4double, std::less<G4String> > CKP ; 
    CKP* cm = mpt->GetPropertiesCMap();

    for(CKP::const_iterator it=cm->begin() ; it != cm->end() ; it++)
    {   
        const std::string& n = it->first ;
        double pvalue = it->second ;  

        dig.update( const_cast<char*>(n.data()), n.size() );  
        dig.update( reinterpret_cast<char*>(&pvalue), sizeof(double) );  
    }  
    return dig.finalize();
}


