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

#include "BStr.hh"
#include "BFile.hh"
#include "BPath.hh"
#include "SDigest.hh"

#include "PLOG.hh"

const char* BPath::getIdPath() const 
{
    return m_idpath ; 
} 
const char* BPath::getIdFile() const  
{
    return m_idfile ; 
}
const char* BPath::getSrcDigest() const  
{
    return m_srcdigest ; 
}
const char* BPath::getIdName() const  
{
    return m_idname ; 
}
const char* BPath::getGeoBase() const  
{
    return m_geobase ; 
}
const char* BPath::getPrefix() const  
{
    return m_prefix ; 
}
const char* BPath::getSrcPath() const  
{
    return m_srcpath ; 
}
int         BPath::getLayout() const  
{
    return m_layout ; 
}


BPath::BPath(const char* idpath)
    :
    m_idpath(idpath ? strdup(idpath) : NULL ),
    m_triple(false),
    m_idfile(NULL),
    m_srcdigest(NULL),
    m_idname(NULL),
    m_prefix(NULL),
    m_srcpath(NULL),
    m_layout(-1)
{
    init();
}

void BPath::init()
{
    if(!m_idpath) return ; 
    BFile::SplitPath(m_elem, m_idpath );
    parseLayout();
}


bool BPath::isTriple(const char* triple ) const 
{
    if(!triple) return false ; 
    std::vector<std::string> bits ; 
    BStr::split(bits, triple, '.' );  
    return bits.size() == 3 ; 
}

bool BPath::isDigest(const char* last ) const 
{
    return SDigest::IsDigest(last) ;  
}


bool BPath::parseTriple(const char* triple ) 
{
    if(!triple) return false ; 

    std::vector<std::string> bits ; 
    BStr::split(bits, triple, '.' );  

    if( bits.size() != 3 )
    { 
        LOG(debug) << "not a triple" << triple ; 
        return false   ; 
    }

    const std::string& a = bits[0] ; 
    const std::string& b = bits[1] ; 
    const std::string& c = bits[2] ; 

    m_idfile = BStr::concat<const char*>( a.c_str(), ".", c.c_str() );    
    m_srcdigest = strdup(b.c_str()) ;
    assert( strlen(m_srcdigest) == 32 ); 

    return true ; 
}
 
const char* BPath::getElem(int idx) const 
{
    int nelem = m_elem.size(); 
    if( idx < 0 ) idx += nelem ; 
    assert( idx >= 0 && idx < nelem );
    return strdup(m_elem[idx].c_str()) ; 
}




void BPath::parseLayout() 
{
    /*

    layout 0
         /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae

         has dotted triple as last element: g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
         which yields idfile (eg g4_00.dae) and digest (96ff965744a2f6b78c24e33c80d3a4cd) 

    layout > 0
         /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 

          last elem is the layout integer

    */

    const char* last = getElem(-1) ; 
    LOG(info) << " last " << last ;  

    if(isTriple(last) )
    {
        m_triple = parseTriple(last) ;
        m_layout = m_triple ? 0 : -1 ; 
        assert( m_triple ) ;   
        m_idname = getElem(-2) ; 

        unsigned i0 = 0 ; 
        unsigned i1 = m_elem.size() - 2 ; 
        std::string geobase = BFile::FormPath( m_elem, i0, i1 ) ; 
        m_geobase = strdup(geobase.c_str()); 

        unsigned j1 = m_elem.size() - 4 ; 
        std::string prefix = BFile::FormPath( m_elem, i0, j1 ) ; 
        m_prefix = strdup(prefix.c_str()); 
    }
    else
    {
        m_layout = atoi(last) ; 
        assert( m_layout > 0 );
        m_srcdigest = getElem(-2 ); 
        m_idfile = getElem(-3); 
        m_idname = getElem(-4); 

        unsigned i0 = 0 ; 
        unsigned i1 = m_elem.size() - 4 ; 
        std::string geobase = BFile::FormPath( m_elem, i0, i1 ) ; 
        m_geobase = strdup(geobase.c_str()); 

        unsigned j1 = m_elem.size() - 5 ; 
        std::string prefix = BFile::FormPath( m_elem, i0, j1 ) ; 
        m_prefix = strdup(prefix.c_str()); 
    }

    assert( strlen(m_srcdigest) == 32 ); 
    assert( strlen(m_idfile) > 3 ); 
    assert( strlen(m_idname) > 3 ); 

    std::string srcpath = BFile::FormPath( m_prefix, "opticksdata", "export", m_idname, m_idfile ) ; 
    m_srcpath = strdup(srcpath.c_str()); 
}



std::string BPath::desc() const 
{
    std::stringstream ss ;

    unsigned w = 25 ; 

    ss << "BPath" 
       << std::endl 
       << std::setw(w) << " idpath " << ( m_idpath ? m_idpath : "-" ) 
       << std::endl 
       << std::setw(w) << " elem " << m_elem.size() 
       << std::endl 
       << std::setw(w) << " layout " << m_layout 
       << std::endl 
       << std::setw(w) << " idfile " << ( m_idfile ? m_idfile  : "-" )
       << std::endl 
       << std::setw(w) << " srcdigest " << ( m_srcdigest ? m_srcdigest : "-" )
       << std::endl 
       << std::setw(w) << " idname " << ( m_idname ? m_idname : "-" )
       << std::endl 
       << std::setw(w) << " geobase " << ( m_geobase ? m_geobase : "-" )
       << std::endl 
       << std::setw(w) << " prefix " << ( m_prefix ? m_prefix : "-" )
       << std::endl 
       << std::setw(w) << " srcpath " << ( m_srcpath ? m_srcpath : "-" )
       << std::endl 
       ;

    return ss.str();
}



