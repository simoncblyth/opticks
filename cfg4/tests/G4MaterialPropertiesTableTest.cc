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

// TEST=G4MaterialPropertiesTableTest om-t

#include <iomanip>
#include "G4MaterialPropertiesTable.hh"

#include "CMPT.hh"
#include "OPTICKS_LOG.hh"

void test_MPT(G4MaterialPropertiesTable* mpt)
{
    typedef G4MaterialPropertyVector MPV ; 
    G4bool warning ; 
    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
    LOG(info) << "pns:" << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {
        const std::string& pn = pns[i]; 
        G4int idx = mpt->GetPropertyIndex(pn, warning=true); 
        assert( idx > -1 );  
        MPV* mpv = mpt->GetProperty(idx, warning=false ); 

        std::cout 
            << " pn : " 
            << std::setw(30) << pn 
            << " idx : " 
            << std::setw(5) << idx 
            << " mpv : "
            << std::setw(16) << mpv 
            << std::endl
            ;  
    } 
}


void test_MPTConst(G4MaterialPropertiesTable* mpt)
{
    G4bool warning ; 
    std::vector<G4String> pns = mpt->GetMaterialConstPropertyNames() ;
    LOG(info) << "pns:" << pns.size() ; 
    for( unsigned i=0 ; i < pns.size() ; i++)
    {
        const std::string& pn = pns[i]; 

        G4bool exists = mpt->ConstPropertyExists( pn.c_str() ) ; 

        G4int idx = mpt->GetConstPropertyIndex(pn, warning=true); 
        assert( idx > -1 );  
        G4double pval = exists ? mpt->GetConstProperty(idx) : 0. ; 

        std::cout 
            << " pn : " 
            << std::setw(30) << pn 
            << " exists : " 
            << std::setw(3) << exists 

            << " idx : " 
            << std::setw(5) << idx 
            << " pval : "
            << std::setw(16) << pval 
            << std::endl
            ;  
    } 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG__(argc, argv);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable(); 

    LOG(info) << " mpt " << mpt ;  

    CMPT::AddDummyProperty( mpt, "A", 5 ); 
    CMPT::AddDummyProperty( mpt, "B", 10 ); 
    CMPT::AddDummyProperty( mpt, "C", 10 ); 

    CMPT::AddConstProperty( mpt, "AA", 5 ); 
    CMPT::AddConstProperty( mpt, "BB", 10 ); 
    CMPT::AddConstProperty( mpt, "CC", 10 ); 


    CMPT::Dump( mpt ) ;    

    test_MPT(mpt); 
    test_MPTConst(mpt); 


    return 0 ; 
}



