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

#include <cstring>
#include <string>
#include <sstream>

#include "BStr.hh"
#include "BOpticksEvent.hh"

#include "NLoad.hpp"
#include "NPY.hpp"

#include "OpticksEventSpec.hh"

#include "PLOG.hh"

const char* OpticksEventSpec::G4_ = "G4" ; 
const char* OpticksEventSpec::OK_ = "OK" ; 
const char* OpticksEventSpec::NO_ = "NO" ; 

const plog::Severity OpticksEventSpec::LEVEL = PLOG::EnvLevel("OpticksEventSpec", "DEBUG") ; 

OpticksEventSpec::OpticksEventSpec(OpticksEventSpec* spec) 
    :
    m_pfx(strdup(spec->getPfx())),
    m_typ(strdup(spec->getTyp())),
    m_tag(strdup(spec->getTag())),
    m_det(strdup(spec->getDet())),
    m_cat(spec->getCat() ? strdup(spec->getCat()) : NULL),
    m_udet(spec->getUDet() ? strdup(spec->getUDet()) : NULL),
    m_dir(NULL),
    m_reldir(NULL),
    m_fold(NULL),
    m_itag(spec->getITag())
{
}

OpticksEventSpec::OpticksEventSpec(const char* pfx, const char* typ, const char* tag, const char* det, const char* cat) 
    :
    m_pfx(strdup(pfx)),
    m_typ(strdup(typ)),
    m_tag(strdup(tag)),
    m_det(strdup(det)),
    m_cat(cat ? strdup(cat) : NULL),
    m_udet(cat && strlen(cat) > 0 ? strdup(cat) : strdup(det)),
    m_dir(NULL),
    m_reldir(NULL),
    m_fold(NULL),
    m_itag(BStr::atoi(m_tag, 0))
{
}

OpticksEventSpec::~OpticksEventSpec()
{
    //LOG(info); 
    free((char*)m_pfx);
    free((char*)m_typ);
    free((char*)m_tag);
    free((char*)m_det);
    free((char*)m_cat);
    free((char*)m_udet);
    free((char*)m_dir);
    free((char*)m_reldir);
    free((char*)m_fold);
}

/**
OpticksEventSpec::getOffsetTag
-------------------------------

**/

const char* OpticksEventSpec::getOffsetTag(unsigned tagoffset) const
{
    int ntag = getOffsetTagInteger(tagoffset); 
    const char* tag = BStr::itoa( ntag );
    return tag ; 
}

/**
OpticksEventSpec::getOffsetTagInteger
---------------------------------------

The ITag of the event is constrained to being a non-zero integer.

ITag +ve (Opticks events)
    tag integer returned is itag+tagoffet

ITag -ve (Geant4 events)
    tag integer returned is itag-tagoffet

This system is used to manage and identify paired events 

**/

int OpticksEventSpec::getOffsetTagInteger(unsigned tagoffset) const
{
    int itag = getITag();
    bool iszero = itag == 0 ; 
    if( iszero )
    { 
        LOG(fatal) 
            << " iszero itag "
            << brief()
            ; 
    }  

    assert( !iszero && "--tag 0 NOT ALLOWED : AS USING G4 NEGATED CONVENTION " );
    int ntag = itag > 0 ? itag + tagoffset : itag - tagoffset ; 
    return ntag ; 
}





OpticksEventSpec* OpticksEventSpec::clone(unsigned tagoffset) const 
{
    const char* tag = getOffsetTag(tagoffset);  
    return new OpticksEventSpec( getPfx(), getTyp(), tag, getDet(), getCat() );
}

int OpticksEventSpec::getITag() const 
{
    return m_itag ; 
}
bool OpticksEventSpec::isG4() const
{
    return m_itag < 0 ;     
}
bool OpticksEventSpec::isOK() const
{
    return m_itag > 0 ;     
}

const char* OpticksEventSpec::getEngine() const
{
    const char* engine = NO_ ; 
    if(     isOK())  engine = OK_ ; 
    else if(isG4())  engine = G4_ ; 
    return engine ; 
}



const char* OpticksEventSpec::getPfx() const 
{
    return m_pfx ; 
}
const char* OpticksEventSpec::getTyp() const 
{
    return m_typ ; 
}

const char* OpticksEventSpec::getTag() const 
{
    return m_tag ; 
}
const char* OpticksEventSpec::getDet() const 
{
    return m_det ; 
}
const char* OpticksEventSpec::getCat() const 
{
    return m_cat ; 
}
const char* OpticksEventSpec::getUDet() const
{
    return m_udet ; 
}





const char* OpticksEventSpec::formDir() const  
{
    const char* pfx = m_pfx ; 
    const char* top = m_udet ; 
    const char* sub = m_typ ; 
    const char* tag = m_tag ; 
    const char* anno = NULL ; 
    std::string dir = BOpticksEvent::directory(pfx, top, sub, tag, anno);    

    LOG(LEVEL) 
        << " pfx " << pfx
        << " top " << top
        << " sub " << sub
        << " tag " << tag
        << " dir " << dir
        ;

    return strdup(dir.c_str()) ; 
}
const char* OpticksEventSpec::formRelDir() const  
{
    const char* pfx = m_pfx ; 
    const char* top = m_udet ; 
    const char* sub = m_typ ; 
    const char* tag = m_tag ; 
    std::string dir = BOpticksEvent::reldir(pfx, top, sub, tag);    
    return strdup(dir.c_str()) ; 
}
const char* OpticksEventSpec::formFold() const  
{
    const char* pfx = m_pfx ; 
    const char* top = m_udet ; 
    const char* sub = m_typ ; 
    const char* tag = NULL ; 
    const char* anno = NULL ; 
    std::string dir = BOpticksEvent::directory(pfx, top, sub, tag, anno);    

    LOG(LEVEL) 
        << " pfx " << pfx
        << " top " << top
        << " sub " << sub
        << " dir " << dir
        ;

    return strdup(dir.c_str()) ; 
}


const char* OpticksEventSpec::getDir() 
{
    if(m_dir == NULL) m_dir = formDir(); 
    return m_dir ; 
}
const char* OpticksEventSpec::getRelDir() 
{
    if(m_reldir == NULL) m_reldir = formRelDir(); 
    return m_reldir ; 
}
const char* OpticksEventSpec::getFold() 
{
    if(m_fold == NULL) m_fold = formFold(); 
    return m_fold ; 
}


std::string OpticksEventSpec::brief() const 
{
    std::stringstream ss ; 
    ss 
       << " pfx " << m_pfx
       << " typ " << m_typ
       << " tag " << m_tag
       << " itag " << getITag()
       << " det " << m_det
       << " cat " << ( m_cat ? m_cat : "-" )
       << " eng " << getEngine()
       ;

    return ss.str();
}

void OpticksEventSpec::Summary(const char* msg) const
{
    LOG(info) << msg 
              << " " 
              << brief() 
              ;
}
