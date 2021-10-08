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
#include "GDomain.hh"
#include "GProperty.hh"
#include "PLOG.hh"

const plog::Severity X4MaterialPropertiesTable::LEVEL = PLOG::EnvLevel("X4MaterialPropertiesTable", "DEBUG"); 

/**
X4MaterialPropertiesTable::GetPropertyIndex
--------------------------------------------

This provides a workaround for the removal of property discovery functionality 
from Geant4 1100 G4MaterialPropertiesTable::GetPropertyIndex, 
as that now throws fatal exceptions for non-existing keys.

Also bizarrely there is no *PropertyExists* method but there is *ConstPropertyExists*, 
hence in order to discover if a property exists it is necessary to 
GetMaterialPropertyNames (bizarrely by value) and then check within that 
vector of strings. This static method does this, reproducing the old behavior.

**/


int X4MaterialPropertiesTable::GetPropertyIndex( const G4MaterialPropertiesTable* mpt, const char* key ) // static
{
    const std::vector<G4String> names = mpt->GetMaterialPropertyNames() ;
    return GetIndex(names, key); 
}

/**
X4MaterialPropertiesTable::GetConstPropertyIndex
--------------------------------------------------

Suspect flawed implementation in G4 1100 with non-existing keys, so role own. 

**/

int X4MaterialPropertiesTable::GetConstPropertyIndex( const G4MaterialPropertiesTable* mpt, const char* key ) // static
{
    const std::vector<G4String> constPropNames = mpt->GetMaterialConstPropertyNames() ;
    return GetIndex(constPropNames, key); 
}

int X4MaterialPropertiesTable::GetIndex(const std::vector<G4String>& nn, const char* key ) // static
{
    G4String k(key); 
    typedef std::vector<G4String> VS ; 
    typedef VS::const_iterator   VSI ; 
    VSI b = nn.begin() ; 
    VSI e = nn.end() ; 
    VSI p = std::find(b, e, k ); 
    return p == e ? -1 : std::distance(b, p) ; 
}



void X4MaterialPropertiesTable::Convert( GPropertyMap<double>* pmap,  const G4MaterialPropertiesTable* const mpt, char mode )
{
    if(mpt == NULL) 
      LOG(fatal) << "cannot convert a null G4MaterialPropertiesTable : this usually means you have omitted to setup any properties for a surface or material" ;  
    assert( mpt ); 
    X4MaterialPropertiesTable xtab(pmap, mpt, mode);
}

X4MaterialPropertiesTable::X4MaterialPropertiesTable( GPropertyMap<double>* pmap,  const G4MaterialPropertiesTable* const mpt, char mode )
    :
    m_pmap(pmap),
    m_mpt(mpt),
    m_mode(mode)
{
    init();
}

void X4MaterialPropertiesTable::init()
{ 
    AddProperties( m_pmap, m_mpt, m_mode );    
}

/**
X4MaterialPropertiesTable::AddProperties
-------------------------------------------

Used from X4Material::Convert/X4Material::init

**/

void X4MaterialPropertiesTable::AddProperties(GPropertyMap<double>* pmap, const G4MaterialPropertiesTable* const mpt, char mode )   //  static
{
    typedef G4MaterialPropertyVector MPV ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(LEVEL) << " MaterialPropertyNames pns.size " << pns.size() ; 
        
    GDomain<double>* dom = GDomain<double>::GetDefaultDomain(); 
    unsigned pns_null = 0 ; 

    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& pname = pns[i]; 
        G4int pidx = X4MaterialPropertiesTable::GetPropertyIndex(mpt, pname.c_str()); 
        assert( pidx > -1 );  
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx);  
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

        GProperty<double>* prop = nullptr ;        

        if( mode == 'G' )           // Geant4 src interpolation onto the domain 
        {
            prop = X4PhysicsVector<double>::Interpolate(pvec, dom) ; 
            pmap->addPropertyAsis( pname.c_str(), prop );     
        }
        else if( mode == 'S' )      // Opticks pmap interpolation onto standard domain   
        {
            bool nm_domain = true ;  
            prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ; 
            pmap->addPropertyStandardized( pname.c_str(), prop );  
        }
        else if( mode == 'A' )      //  asis : no interpolation, but converted to nm  
        {
            bool nm_domain = true ;  
            prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ; 
            pmap->addPropertyAsis( pname.c_str(), prop );     
        }
        else if( mode == 'E' )      //  asis : no interpolation, NOT converted to nm : Energy domain 
        {
            bool nm_domain = false ;  
            prop = X4PhysicsVector<double>::Convert(pvec, nm_domain ) ; 
            pmap->addPropertyAsis( pname.c_str(), prop );     
        }
        else
        {
            LOG(fatal) << " mode must be one of G/S/A/E " ; 
            assert(0); 
        }

        if(strcmp(pname.c_str(), "EFFICIENCY") == 0)
        {
            LOG(LEVEL) << prop->brief("X4MaterialPropertiesTable::AddProperties.EFFICIENCY"); 
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

        G4int pidx = mpt->GetConstPropertyIndex(pname); 
        //assert( pidx > -1 );  // comment assert to investigate behavior change with 91702(aka 1100)
        G4double pval = pidx > -1 ? mpt->GetConstProperty(pidx) : -1. ;  

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

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(LEVEL) << " NumProp " << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {   
        const std::string& n = pns[i]; 
        int pidx = X4MaterialPropertiesTable::GetPropertyIndex(mpt, n.c_str()); 
        assert( pidx > -1 );  
        MPV* v = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx);  
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
        int pidx = X4MaterialPropertiesTable::GetConstPropertyIndex(mpt, n.c_str());  
        if(pidx == -1) continue ; 

        G4double pvalue = mpt->GetConstProperty(pidx);  

        dig.update( const_cast<char*>(n.data()), n.size() );  
        dig.update( reinterpret_cast<char*>(&pvalue), sizeof(double) );  
    }
    return dig.finalize();
}



