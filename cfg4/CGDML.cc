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


#include "SPath.hh"
#include "SGDML.hh"
#include "BMeta.hh"
#include "BFile.hh"

#include "CGDML.hh"
#include "G4GDMLParser.hh"

#include "PLOG.hh"

const plog::Severity CGDML::LEVEL = PLOG::EnvLevel("CGDML", "DEBUG"); 

G4VPhysicalVolume* CGDML::Parse(const char* path) // static 
{
    CGDML cg; 
    cg.read(path);  
    return cg.getWorldVolume() ;
}
G4VPhysicalVolume* CGDML::Parse(const char* path, BMeta** meta) // static 
{
    CGDML cg ;
    cg.read(path);  
    *meta = cg.getMeta();
    return cg.getWorldVolume() ;
}
G4VPhysicalVolume* CGDML::Parse(const char* dir, const char* name) // static 
{
    std::string path = BFile::FormPath(dir, name);
    return CGDML::Parse(path.c_str());
}
G4VPhysicalVolume* CGDML::Parse(const char* dir, const char* name, BMeta** meta) // static 
{
    std::string path = BFile::FormPath(dir, name);
    return CGDML::Parse(path.c_str(), meta);
}


CGDML::CGDML()
    :
    m_parser(new G4GDMLParser),
    m_write_refs(true),
    m_write_schema_location(""),
    m_read_validate(false),
    m_read_trimPtr(false)
{
}

void CGDML::read(const char* path_)
{
    const char* path = SPath::Resolve(path_, false); 
    LOG(info) << " resolved path_ " << path_ << " as path " << path ;   
    m_parser->SetStripFlag(m_read_trimPtr),
    m_parser->Read(path, m_read_validate);
}

G4VPhysicalVolume* CGDML::getWorldVolume() const 
{
    const G4String setupName = "Default" ; 
    return m_parser->GetWorldVolume(setupName) ; 
}


/**
CGDML::getLVMeta
--------------------

Due to current keyed only (no lists) limitations of BMeta interface 
to the underlying json implementation this does not completely capture the aux info. 
But enough to be useful. eg::

    epsilon:cfg4 blyth$ CGDMLTest /tmp/v1.gdml
    G4GDML: Reading '/tmp/v1.gdml'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/v1.gdml' done!
    2020-12-08 18:48:40.448 INFO  [3590231] [*CGDML::getUserMeta@245] auxlist 0x7ff7e9726438
    2020-12-08 18:48:40.449 INFO  [3590231] [BMeta::dump@199] CGDMLTest::test_Parse
    {
        "lvmeta": {
            "/dd/Geometry/AD/lvADE0xc2a78c00x3ef9140": {
                "label": "target",
                "lvname": "/dd/Geometry/AD/lvADE0xc2a78c00x3ef9140"
            },
            "/dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d980x3ee9e20": {
                "SensDet": "SD0",
                "lvname": "/dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d980x3ee9e20"
            },
            "/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400": {
                "SensDet": "SD0",
                "lvname": "/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400"
            }
        },
        "usermeta": {
            "opticks_blue": "3",
            "opticks_green": "2",
            "opticks_red": "1"
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

::

    1 <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    2 <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="">
    3 
    4   <userinfo>
    5       <auxiliary auxtype="opticks_red" auxvalue="1"/>
    6       <auxiliary auxtype="opticks_green" auxvalue="2"/>
    7       <auxiliary auxtype="opticks_blue" auxvalue="3"/>
    8   </userinfo>

**/


BMeta* CGDML::getLVMeta() const 
{
    const G4GDMLAuxMapType* auxmap = m_parser->GetAuxMap();
    if( auxmap->size() == 0 ) return NULL ; 

    BMeta* lvmeta = new BMeta ; 

    typedef G4GDMLAuxMapType::const_iterator MIT ;  
    typedef G4GDMLAuxListType::const_iterator VIT ; 

    for (MIT mit = auxmap->begin(); mit != auxmap->end(); mit++) 
    {
        G4LogicalVolume* lv = mit->first ; 
        G4GDMLAuxListType ls = mit->second ;      
        const G4String& lvname = lv->GetName();  

        BMeta* one = new BMeta ; 
        one->set<std::string>("lvname", lvname ) ;  

        // although duplicating the higher level key, its convenient
        // for sub-objects to have this too

        for (VIT vit = ls.begin(); vit != ls.end(); vit++) 
        {
            const G4GDMLAuxStructType& aux = *vit ;  
            one->set<std::string>(aux.type, aux.value) ; 
        }   
        lvmeta->setObj(lvname, one);  
    }
    return lvmeta ; 
}


void CGDML::addLVMeta(const BMeta* lvmeta)
{
    LOG(info) << " NOT IMPLEMENTED " << lvmeta ; 
}


const char* CGDML::LVMETA = "lvmeta" ; 
const char* CGDML::USERMETA = "usermeta" ; 

BMeta* CGDML::getMeta() const 
{
    BMeta* meta = new BMeta ; 
    
    BMeta* lv = getLVMeta(); 
    BMeta* user = getUserMeta(); 

    if(lv)   meta->setObj(LVMETA,  lv) ; 
    if(user) meta->setObj(USERMETA,  user) ; 

    return meta ; 
}

void CGDML::addMeta(const BMeta* meta)
{
    BMeta* lv   = meta->getObj(LVMETA);   
    BMeta* user = meta->getObj(USERMETA);   
    
    addLVMeta(lv); 
    addUserMeta(user); 
}



void CGDML::dumpLVMeta(const char* msg) const
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





void CGDML::dumpUserMeta(const char* msg) const
{
    LOG(LEVEL) << msg ; 
    const G4GDMLAuxListType* auxlist = m_parser->GetAuxList() ; 

    LOG(info) << msg 
              << " auxlist " << auxlist 
              << " auxlist.size " << ( auxlist ? auxlist->size() : -1 )
              << " (userinfo/auxiliary info) " 
              ;

    if(!auxlist) return ; 

    typedef G4GDMLAuxListType::const_iterator VIT ; 

    for (VIT vit = auxlist->begin(); vit != auxlist->end(); vit++) 
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


/**
CGDML::getUserMeta
--------------------

Simplifying assumption of unique keys.

If that is not the case in general, can restrict collection to 
only metadata with aux.type keys starting with "opticks_" prefix. 

**/

BMeta* CGDML::getUserMeta() const 
{
    const G4GDMLAuxListType* auxlist = m_parser->GetAuxList() ; 
    LOG(info) << "auxlist " << auxlist ;  
    if(!auxlist) return NULL ;
    if(auxlist->size() == 0 ) return NULL ; 
 
    typedef G4GDMLAuxListType::const_iterator VIT ; 
    BMeta* user = new BMeta ; 

    for (VIT vit = auxlist->begin(); vit != auxlist->end(); vit++) 
    {
        const G4GDMLAuxStructType& aux = *vit ;  
        user->set<std::string>(aux.type, aux.value) ; 
    } 
    return user ; 
}

void CGDML::addUserMeta(const BMeta* user)
{
    unsigned nk = user->getNumKeys(); 
    std::string k ; 
    std::string v ; 
    for(unsigned i=0 ; i < nk ; i++)
    {
        user->getKV(i, k, v );     // unordered      
        G4GDMLAuxStructType aux ;
        aux.type = k ; 
        aux.value = v ; 
        aux.unit = "" ; 
        aux.auxList = NULL ; 
        m_parser->AddAuxiliary(aux);
    }
}

void CGDML::Export(const char* dir, const char* name, const G4VPhysicalVolume* const world, const BMeta* meta)
{
    std::string path = BFile::FormPath(dir, name);
    CGDML::Export( path.c_str(), world, meta ); 
}

void CGDML::Export(const char* path, const G4VPhysicalVolume* const world, const BMeta* meta)
{
    assert( world );
    CGDML cg ; 
    if(meta) cg.addMeta(meta);  
    cg.write(path, world, meta ); 
}

void CGDML::write( const char* path,  const G4VPhysicalVolume* const world, const BMeta* meta )
{
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
    LOG(info) << "write to " << path ; 

    m_parser->Write(path, world, m_write_refs, m_write_schema_location ); 
}


// based on G4GDMLWrite::GenerateName 
std::string CGDML::GenerateName(const char* name, const void* const ptr, bool addPointerToName )
{
    return SGDML::GenerateName(name, ptr, addPointerToName );
}



