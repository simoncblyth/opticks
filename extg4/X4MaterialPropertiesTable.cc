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

const plog::Severity X4MaterialPropertiesTable::LEVEL = PLOG::EnvLevel("X4MaterialPropertiesTable", "DEBUG"); 


void X4MaterialPropertiesTable::Convert( GPropertyMap<double>* pmap,  const G4MaterialPropertiesTable* const mpt, bool standardized )
{
    if(mpt == NULL) 
      LOG(fatal) << "cannot convert a null G4MaterialPropertiesTable : this usually means you have omitted to setup any properties for a surface or material" ;  
    assert( mpt ); 
    X4MaterialPropertiesTable xtab(pmap, mpt, standardized);
}

X4MaterialPropertiesTable::X4MaterialPropertiesTable( GPropertyMap<double>* pmap,  const G4MaterialPropertiesTable* const mpt, bool standardized )
    :
    m_pmap(pmap),
    m_mpt(mpt),
    m_standardized(standardized)
{
    init();
}

void X4MaterialPropertiesTable::init()
{ 
    AddProperties( m_pmap, m_mpt, m_standardized );    
}

/**
X4MaterialPropertiesTable::AddProperties
-------------------------------------------

Used from X4Material::Convert/X4Material::init


**/

void X4MaterialPropertiesTable::AddProperties(GPropertyMap<double>* pmap, const G4MaterialPropertiesTable* const mpt, bool standardized )   // static
{
    typedef G4MaterialPropertyVector MPV ; 
    G4bool warning ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;

    unsigned pns_null = 0 ; 

    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& pname = pns[i]; 
        G4int pidx = mpt->GetPropertyIndex(pname, warning=true); 
        assert( pidx > -1 );  
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning=false );  
        LOG(LEVEL)
            << " pname : " 
            << std::setw(30) << pname  
            << " pidx : " 
            << std::setw(5) << pidx 
            << " pvec : "
            << std::setw(16) << pvec 
            ;   

        if(pvec == NULL) 
        {
            pns_null += 1 ;  
            continue ; 
        }

        GProperty<double>* prop = X4PhysicsVector<double>::Convert(pvec) ; 

        if(strcmp(pname.c_str(), "EFFICIENCY") == 0)
        {
            LOG(LEVEL) << prop->brief("X4MaterialPropertiesTable::AddProperties.EFFICIENCY"); 
        }

        if( standardized )
        {
            pmap->addPropertyStandardized( pname.c_str(), prop );  // interpolates onto standard domain 
        }
        else
        {
            pmap->addPropertyAsis( pname.c_str(), prop );     // for raw materials, needed for scintillator props
        }

    }
    LOG(LEVEL) 
        << " pns " << pns.size()
        << " pns_null " << pns_null
         ; 


    std::vector<G4String> cpns = mpt->GetMaterialConstPropertyNames() ;

    unsigned cpns_null = 0 ; 

    for( unsigned i=0 ; i < cpns.size() ; i++)
    {   
        const std::string& pname = cpns[i]; 
        G4bool exists = mpt->ConstPropertyExists( pname.c_str() ) ;
        if(!exists)
        { 
            cpns_null += 1 ; 
            continue ; 
        } 
 
        G4int pidx = mpt->GetConstPropertyIndex(pname, warning=true); 
        assert( pidx > -1 );  
        G4double pval = mpt->GetConstProperty(pidx);  

        LOG(LEVEL)
            << " pname : " 
            << std::setw(30) << pname  
            << " pidx : " 
            << std::setw(5) << pidx 
            << " pval : "
            << std::setw(16) << pval 
            ;   

        pmap->addConstantProperty( pname.c_str(), pval );   // asserts without standard domain
    }

    LOG(LEVEL) 
        << " cpns " << cpns.size()
        << " cpns_null " << cpns_null
         ; 




}



std::string X4MaterialPropertiesTable::Digest(const G4MaterialPropertiesTable* mpt)  // static
{
    if(!mpt) return "" ; 

    SDigest dig ;

    typedef G4MaterialPropertyVector MPV ; 
    G4bool warning ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(LEVEL) << " NumProp " << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& n = pns[i]; 
        G4int pidx = mpt->GetPropertyIndex(n, warning=true); 
        assert( pidx > -1 );  
        MPV* v = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning=false );  
        if(v == NULL) continue ; 

        std::string vs = X4PhysicsVector<double>::Digest(v) ; 
        dig.update( const_cast<char*>(n.data()),  n.size() );  
        dig.update( const_cast<char*>(vs.data()), vs.size() );  
    }

    std::vector<G4String> cpns = mpt->GetMaterialConstPropertyNames() ;
    LOG(LEVEL) << " NumPropConst " << cpns.size() ; 

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



