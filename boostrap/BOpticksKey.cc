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

#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <csignal>
#include <vector>

#include "SAr.hh"
#include "SSys.hh"
#include "SDigest.hh"
#include "BStr.hh"
#include "BOpticksKey.hh"

#include "PLOG.hh"

const plog::Severity BOpticksKey::LEVEL = PLOG::EnvLevel("BOpticksKey", "DEBUG") ; 

BOpticksKey* BOpticksKey::fKey = NULL ; 

const char* BOpticksKey::G4LIVE = "g4live" ; 
const char* BOpticksKey::IDSTEM = "g4ok" ; 
const char* BOpticksKey::IDFILE = "g4ok.gltf" ; 
const char* BOpticksKey::IDSUBD = "g4ok_gltf" ; 
int         BOpticksKey::LAYOUT = 1 ; 


bool BOpticksKey::IsSet()  // static
{
    return fKey != NULL ; 
}

BOpticksKey* BOpticksKey::GetKey()
{
    // invoked by BOpticksResource::BOpticksResource at Opticks instanciation
    return fKey ; 
}

const char* BOpticksKey::StemName( const char* ext, const char* sep )
{
    return BStr::concat(IDSTEM, sep, ext );
}


bool BOpticksKey::SetKey(const char* spec)
{
    if(BOpticksKey::IsSet())
    {
        LOG(error) << "key is already set, ignoring update with spec " << spec ;
        return true ;  
    }

    if(spec == NULL)
    {
        spec = SSys::getenvvar("OPTICKS_KEY");  
        LOG(LEVEL) << "from OPTICKS_KEY envvar " << spec ; 
    } 

    LOG(info) << " spec " << spec ; 

    fKey = spec ? new BOpticksKey(spec) : NULL  ; 

    if(fKey) LOG(LEVEL) << std::endl << fKey->desc() ; 

    return true ; 
}

void BOpticksKey::Desc()
{
    if(fKey) LOG(info) << std::endl << fKey->desc() ; 
}




bool BOpticksKey::isKeySource() const  // current executable is geocache creator 
{
    return m_current_exename && m_exename && strcmp(m_current_exename, m_exename) == 0 ; 
}

BOpticksKey::BOpticksKey(const char* spec)
    :
    m_spec( spec ? strdup(spec) : NULL ),
    m_exename( NULL ),
    m_class( NULL ),
    m_volname( NULL ),
    m_digest( NULL ),
    m_idname( NULL ),
    m_idfile( StemName("gltf", ".") ),
    m_idgdml( StemName("gdml", ".") ),
    m_idsubd( IDSUBD ),
    m_layout( LAYOUT ),
    m_current_exename( SAr::Instance ? SAr::Instance->exename() : "OpticksEmbedded" )
{
    std::vector<std::string> elem ; 
    BStr::split(elem, spec, '.' ); 

    bool four = elem.size() == 4  ;
    if(!four) LOG(fatal) << " expecting 4 element spec delimited by dot " << spec ;  
    assert( four ); 
    
    m_exename = strdup(elem[0].c_str()); 
    m_class = strdup(elem[1].c_str()); 
    m_volname   = strdup(elem[2].c_str()); 
    m_digest = strdup(elem[3].c_str()); 

    assert( SDigest::IsDigest(m_digest) ); 

    std::stringstream ss ; 
    ss 
        << m_exename 
        << "_"
        << m_volname 
        << "_"
        << G4LIVE 
        ;

    std::string idname = ss.str();

    m_idname = strdup(idname.c_str()); 
}


const char* BOpticksKey::getSpec() const 
{
    return m_spec ;  
}
const char* BOpticksKey::getExename() const 
{
    return m_exename ;  
}
const char* BOpticksKey::getClass() const 
{
    return m_class ;  
}
const char* BOpticksKey::getVolname() const 
{
    return m_volname ;  
}
const char* BOpticksKey::getDigest() const 
{
    return m_digest ;  
}


const char* BOpticksKey::getIdname() const 
{
    return m_idname ;  
}

const char* BOpticksKey::getIdfile() const 
{
    return m_idfile ;  
}
const char* BOpticksKey::getIdGDML() const 
{
    return m_idgdml ;  
}


const char* BOpticksKey::getIdsubd() const 
{
    return m_idsubd ;  
}


int BOpticksKey::getLayout() const 
{
    return m_layout ;  
}


std::string BOpticksKey::desc() const 
{
    std::stringstream ss ; 
    ss 
        << std::setw(25) << " BOpticksKey " << " : " << ( isKeySource() ? "KEYSOURCE" : " " ) << std::endl 
        << std::setw(25) << " spec (OPTICKS_KEY) " << " : " << m_spec    << std::endl 
        << std::setw(25) << " exename " << " : " << m_exename << std::endl 
        << std::setw(25) << " current_exename " << " : " << m_current_exename << std::endl 
        << std::setw(25) << " class "   << " : " << m_class   << std::endl 
        << std::setw(25) << " volname " << " : " << m_volname << std::endl 
        << std::setw(25) << " digest "  << " : " << m_digest  << std::endl 
        << std::setw(25) << " idname "  << " : " << m_idname  << std::endl 
        << std::setw(25) << " idfile "  << " : " << m_idfile  << std::endl 
        << std::setw(25) << " idgdml "  << " : " << m_idgdml  << std::endl 
        << std::setw(25) << " layout "  << " : " << m_layout  << std::endl 
        ;
    return ss.str(); 
}


