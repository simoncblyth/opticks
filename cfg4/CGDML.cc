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


#include "SGDML.hh"
#include "NMeta.hpp"
#include "CGDML.hh"
#include "BFile.hh"
#include "G4GDMLParser.hh"

#include "PLOG.hh"

const plog::Severity CGDML::LEVEL = PLOG::EnvLevel("CGDML", "DEBUG"); 

G4VPhysicalVolume* CGDML::Parse(const char* path) // static 
{
    CGDML cg(path); 
    return cg.getWorldVolume() ;
}
G4VPhysicalVolume* CGDML::Parse(const char* path, NMeta** meta) // static 
{
    CGDML cg(path); 
    *meta = cg.getAuxMeta();
    return cg.getWorldVolume() ;
}

G4GDMLParser* CGDML::InitParser(const char* path)  // static 
{
    LOG(LEVEL) << "path " << path ; 
    bool validate = false ; 
    bool trimPtr = false ; 
    G4GDMLParser* parser = new G4GDMLParser ;
    parser->SetStripFlag(trimPtr);
    parser->Read(path, validate);
    return parser ; 
}

CGDML::CGDML(const char* path)
    :
    m_parser(path ? InitParser(path) : NULL)
{
}

G4VPhysicalVolume* CGDML::getWorldVolume() const 
{
    return m_parser ? m_parser->GetWorldVolume() : NULL ; 
}


/**
CGDML::getAuxMeta
--------------------

Due to current keyed only (no lists) limitations of NMeta interface 
to the underlying json implementation this does not completely capture the aux info. 
But enough to be useful. eg::

    {
        "/dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d980x3ee9e20": {
            "SensDet": "SD0"
        },
        "/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400": {
            "SensDet": "SD0"
        }
    }


GDML Aux example, plant "auxiliary" element inside "volume" (LV) elements::

    <volume name="/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400">
      <materialref ref="/dd/Materials/Bialkali0x3e5d3e0"/>
      <solidref ref="pmt-hemi-cathode0xc2f1ce80x3e842d0"/>
      <auxiliary auxtype="SensDet" auxvalue="SD0"/>
    </volume>


To find the LV name that corresponds to an all_volume node index 
use geocache/GNodeLib/all_volume_LVNames.txt 
(remembering vim line numbers start from 1, not zero)

The names will of course often appear more than once in the all_volume_LVName.txt list.


G4GDMLAux::

    typedef std::map<G4LogicalVolume*,G4GDMLAuxListType> G4GDMLAuxMapType;
    typedef std::vector<G4GDMLAuxStructType> G4GDMLAuxListType;

         42 struct G4GDMLAuxStructType
         43 {
         44    G4String type;
         45    G4String value;
         46    G4String unit;
         47    std::vector<G4GDMLAuxStructType>* auxList;
         48 };

**/


NMeta* CGDML::getAuxMeta() const 
{
    const G4GDMLAuxMapType* auxmap = m_parser->GetAuxMap();
    if( auxmap->size() == 0 ) return NULL ; 

    NMeta* top = new NMeta ; 

    typedef G4GDMLAuxMapType::const_iterator MIT ;  
    typedef G4GDMLAuxListType::const_iterator VIT ; 

    for (MIT mit = auxmap->begin(); mit != auxmap->end(); mit++) 
    {
        G4LogicalVolume* lv = mit->first ; 
        G4GDMLAuxListType ls = mit->second ;      
        const G4String& lvname = lv->GetName();  

        NMeta* lvmeta = new NMeta ; 
        //lvmeta->set<std::string>("lvname", lvname ) ; 

        for (VIT vit = ls.begin(); vit != ls.end(); vit++) 
        {
            const G4GDMLAuxStructType& aux = *vit ;  
            lvmeta->set<std::string>(aux.type, aux.value) ; 
        }   
        top->setObj(lvname, lvmeta);  
    }
    return top ; 
}


void CGDML::dumpAux(const char* msg) const
{
    const G4GDMLAuxMapType* auxmap = m_parser->GetAuxMap();
    LOG(info) << msg 
              << " auxmap.size " << auxmap->size()
              << " (volumes with aux info) " 
              ;

    typedef G4GDMLAuxMapType::const_iterator MIT ;  
    typedef G4GDMLAuxListType::const_iterator VIT ; 


    for (MIT mit = auxmap->begin(); mit != auxmap->end(); mit++) 
    {
        G4LogicalVolume* lv = mit->first ; 
        G4GDMLAuxListType ls = mit->second ;      
          
        std::cout 
             << "LV " << lv->GetName()
             << " has the following list of auxiliary information: "
             << std::endl 
             ;

        for (VIT vit = ls.begin(); vit != ls.end(); vit++) 
        {
            const G4GDMLAuxStructType& aux = *vit ;  
            std::cout 
                   << " aux.type [" << aux.type << "]"
                   << " aux.value ["   << aux.value << "]"
                   << " aux.unit [" << aux.unit << "]"
                   << std::endl 
                   ;
        }   
    }   
}











void CGDML::Export(const char* dir, const char* name, const G4VPhysicalVolume* const world )
{
    std::string path = BFile::FormPath(dir, name);
    CGDML::Export( path.c_str(), world ); 
}

void CGDML::Export(const char* path, const G4VPhysicalVolume* const world )
{
    assert( world );

    bool exists = BFile::ExistsFile( path ); 

    // cannot skip and reuse existing despite it having the same digest 
    // as the pointer locations will differ so all the names will be different
    // relative to those in lv2sd for example
    if(exists) 
    {
        BFile::RemoveFile( path ) ; 
    }

    bool create = true ; 
    BFile::preparePath( path, create ) ;   

    LOG(info) << "export to " << path ; 

    G4GDMLParser* gdml = new G4GDMLParser ;
    G4bool refs = true ;
    G4String schemaLocation = "" ; 

    gdml->Write(path, world, refs, schemaLocation );
}


// based on G4GDMLWrite::GenerateName 
std::string CGDML::GenerateName(const char* name, const void* const ptr, bool addPointerToName )
{
    return SGDML::GenerateName(name, ptr, addPointerToName );
}



