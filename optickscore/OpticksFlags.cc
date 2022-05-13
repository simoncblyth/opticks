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


#include <map>
#include <vector>
#include <string>

#include "PLOG.hh"
#include "BStr.hh"
#include "SBit.hh"

#include "BRegex.hh"
#include "BMeta.hh"

#include "Index.hpp"

#include "OpticksGenstep.hh"
#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"
#include "OpticksFlags.hh"


const plog::Severity OpticksFlags::LEVEL = PLOG::EnvLevel("OpticksFlags", "DEBUG") ; 


const char* OpticksFlags::ABBREV_META_NAME = "OpticksFlagsAbbrevMeta.json" ;
const char* OpticksFlags::ENUM_HEADER_PATH = "$OPTICKS_INSTALL_PREFIX/include/SysRap/OpticksPhoton.h" ;
//  envvar OPTICKS_INSTALL_PREFIX is set internally by OpticksResource based on cmake config 


/**
OpticksFlags::MakeAbbrevMeta
------------------------------

Mapping from flag name string to flag abbreviation string. 

**/

BMeta* OpticksFlags::MakeAbbrevMeta()  // static 
{
    BMeta* m = new BMeta ; 

    typedef std::pair<const char*, const char*> KV ;
    std::vector<KV> pairs ;
    OpticksPhoton::FlagAbbrevPairs(pairs);

    for(unsigned i=0 ; i < pairs.size() ; i++)
    {   
        const KV& kv = pairs[i] ; 
        const char* flag = kv.first ; 
        const char* abbr = kv.second ; 
        m->set<std::string>(flag, abbr); 
    }
    return m ; 
}

BMeta* OpticksFlags::MakeFlag2ColorMeta()  // static 
{
    BMeta* m = BMeta::FromTxt(OpticksPhoton::flag2color); 
    return m ; 
}


const char* OpticksFlags::SourceType( int code )
{
    return OpticksGenstep_::Name(code) ; 
}

unsigned int OpticksFlags::SourceCode(const char* type)
{
    return OpticksGenstep_::Type(type); 
}


Index* OpticksFlags::getIndex()     const { return m_index ;  } 
BMeta* OpticksFlags::getAbbrevMeta() const { return m_abbrev_meta ;  } 
BMeta* OpticksFlags::getColorMeta() const { return m_color_meta ;  } 

OpticksFlags::OpticksFlags(const char* path) 
    :
    m_index(parseFlags(path)),
    m_abbrev_meta(MakeAbbrevMeta()),
    m_color_meta(MakeFlag2ColorMeta())
{
    LOG(LEVEL) << " path " << path ; 
}


void OpticksFlags::save(const char* installcachedir)
{
    LOG(info) << installcachedir ; 
    m_index->setExt(".ini"); 
    m_index->save(installcachedir);
    m_abbrev_meta->save( installcachedir, ABBREV_META_NAME ); 
}

Index* OpticksFlags::parseFlags(const char* path)
{
    typedef std::pair<unsigned, std::string>  upair_t ;
    typedef std::vector<upair_t>              upairs_t ;
    upairs_t ups ;
    BRegex::enum_regexsearch( ups, path ); 

    const char* reldir = NULL ; 
    Index* index = new Index("GFlags", reldir);
    for(unsigned i=0 ; i < ups.size() ; i++)
    {
        upair_t p = ups[i];
        unsigned mask = p.first ;
        unsigned bitpos = SBit::ffs(mask);  // first set bit, 1-based bit position
        unsigned xmask = 1 << (bitpos-1) ; 
        assert( mask == xmask);

        const char* key = p.second.c_str() ;

        LOG(debug) << " key " << std::setw(20) << key 
                   << " bitpos " << bitpos 
                   ;

        index->add( key, bitpos );
   
        //  OpticksFlagsTest --OKCORE debug 
    }

    unsigned int num_flags = index->getNumItems() ;
    if(num_flags == 0)
    { 
        LOG(fatal)
             << " path " << path 
             << " num_flags " << num_flags 
             << " " << ( index ? index->description() : "NULL index" )
             ;
    }
    assert(num_flags > 0 && "missing flags header ? : you need to update OpticksFlags::ENUM_HEADER_PATH ");

    return index ; 
}



