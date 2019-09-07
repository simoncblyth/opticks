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


#include "BLog.hh"
#include "BTxt.hh"
#include "BStr.hh"
#include "SVec.hh"
#include "PLOG.hh"

#include <cmath>
#include <cstdlib>
#include "PLOG.hh"

const plog::Severity BLog::LEVEL = PLOG::EnvLevel("BLog", "DEBUG"); 

const double BLog::TOLERANCE = 1e-6 ; 

const char* BLog::VALUE = " u_" ; 
const char* BLog::CUT = " c_" ; 
const char* BLog::NOTE = " n_" ; 

const char* BLog::DELIM = ":" ; 
const char* BLog::END = " " ; 


BLog* BLog::Load(const char* path)
{
    BTxt* txt = BTxt::Load(path); 
    BLog* log = new BLog ; 

    log->addNote("checknote",101); 

    LOG(LEVEL) 
        << " path " << path 
        << " txt " << txt->desc()
        ;  
    unsigned ni = txt->getNumLines(); 
    for(unsigned i=0 ; i < ni ; i++)
    {
        const std::string& line = txt->getString(i); 
        LOG(LEVEL) << line ; 

        std::string uk, uv ;  
        int urc = ParseKV(line, VALUE, DELIM, END, uk, uv ) ; 

        if(urc == 0)
        { 
            log->addValue( uk.c_str(),  BStr::atod(uv.c_str(), -1. )); 
        
            std::string ck, cv ;  
            int crc = ParseKV(line, CUT, DELIM, END, ck, cv ) ; 
            if(crc == 0)
            {
                log->addCut( ck.c_str(), BStr::atod(cv.c_str(), -1. )); 
            }

            std::string nk, nv ;  
            int nrc = ParseKV(line, NOTE, DELIM, END, nk, nv ) ; 
            if(nrc == 0)
            {
                log->addNote( nk.c_str(), BStr::atoi(nv.c_str(), -1 )); 
            }

        }
    }
    return log ; 
}

int BLog::ParseKV( const std::string& line,  const char* start, const char* delim, const char* end, std::string& k, std::string& v )
{
    std::size_t ps = line.find(start) ; 
    if( ps == std::string::npos ) return 1  ;
    ps += strlen(start) ;     

    std::size_t pd = line.find(delim, ps) ; 
    if( pd == std::string::npos ) return 2 ;   
    pd += 1 ;  

    std::size_t pe = line.find(end, pd) ; 
    if( pe == std::string::npos ) return 3 ;   

    k = line.substr(ps,pd-ps-1); 
    v = line.substr(pd,pe-pd); 

    return 0 ;  
}



BLog::BLog()
    :
    m_sequence(NULL)
{
}
void BLog::setSequence(const std::vector<double>*  sequence)
{
    m_sequence = sequence ;
}

void BLog::addValue(const char* key, double value )
{
    m_keys.push_back(key); 
    m_values.push_back(value); 
}

int BLog::getIdx() const   // returns -1, before adding any keys
{
    return m_keys.size() - 1 ; 
}

void BLog::addCut( const char* ckey, double cvalue )
{
    int idx = getIdx(); 
    m_ckeys.push_back( std::pair<int, std::string>(idx, ckey ) ); 
    m_cvalues.push_back( std::pair<int, double>(idx, cvalue ) ); 
}

void BLog::addNote( const char* nkey, int nvalue )
{
    int idx = getIdx(); 
    m_nkeys.push_back( std::pair<int, std::string>(idx, nkey ) ); 
    m_nvalues.push_back( std::pair<int, int>(idx, nvalue ) ); 
}

unsigned BLog::getNumKeys() const 
{
    return m_keys.size();  
}
const char* BLog::getKey(unsigned i) const 
{
    return m_keys[i].c_str(); 
}
double BLog::getValue(unsigned i) const 
{
    return i < m_values.size() ? m_values[i] : -1. ; 
}

int BLog::getSequenceIndex(unsigned i) const 
{
    double s = getValue(i); 
    return m_sequence ? SVec<double>::FindIndexOfValue( *m_sequence, s, TOLERANCE) : -1 ;  
}


const std::vector<double>& BLog::getValues() const 
{
    return m_values ; 
}

void BLog::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    assert( m_keys.size() == m_values.size() ) ; 

    std::cerr 
         << makeNoteString(-1) 
         << std::endl 
         ;

    for( unsigned i=0 ; i < m_keys.size() ; i++ ) 
    {
        int idx = getSequenceIndex(i) ;
   
        std::cerr 
             << std::setw(4) << idx
             << makeValueString(i, true) 
             << makeCutString(i) 
             << makeNoteString(i) 
             << std::endl 
             ;
    }
}

std::string BLog::makeValueString(unsigned i, bool present) const 
{
    std::stringstream ss ; 
    ss << VALUE
       << std::setw(present ? 40 : 0)  
       << m_keys[i]  
       << DELIM
       << std::setprecision(9) << m_values[i]
       << END
       ; 
    return ss.str(); 
}

std::string BLog::makeCutString(unsigned i, bool ) const 
{
    std::stringstream ss ; 
    assert( m_ckeys.size() == m_cvalues.size() ) ;
    for( unsigned j=0 ; j < m_ckeys.size() ; j++)
    {
        const PIS_t& ick = m_ckeys[j]; 
        const PID_t& icv = m_cvalues[j]; 

        int idx = ick.first ; 
        assert( icv.first == idx );  

        if( idx != i ) continue ;  

        const std::string& ck = ick.second ;    
        const double&      cv = icv.second ;    

        ss << CUT
           << ck  
           << DELIM
           << std::setprecision(9) << cv
           << END
           ; 
    }
    return ss.str(); 
}

std::string BLog::makeNoteString(unsigned i, bool ) const 
{
    std::stringstream ss ; 
    assert( m_nkeys.size() == m_nvalues.size() ) ;
    for( unsigned j=0 ; j < m_nkeys.size() ; j++)
    {
        const PIS_t& ink = m_nkeys[j]; 
        const PII_t& inv = m_nvalues[j]; 

        int idx = ink.first ; 
        assert( inv.first == idx );  

        if( idx != i ) continue ;  

        const std::string& nk = ink.second ;    
        const int&         nv = inv.second ;    

        ss << NOTE
           << nk  
           << DELIM
           << nv
           << END
           ; 
    }
    return ss.str(); 
}


std::string BLog::makeLine(unsigned i) const
{
    std::stringstream ss ; 
    ss << makeValueString(i); 
    ss << makeCutString(i); 
    ss << makeNoteString(i); 
    return ss.str(); 
}


BTxt* BLog::makeTxt() const 
{
    BTxt* txt = new BTxt ; 
    for( unsigned i=0 ; i < m_keys.size() ; i++ ) 
    {
        std::string line = makeLine(i); 
        txt->addLine(line); 
    }
    return txt ; 
}

void BLog::write(const char* path) const 
{
    BTxt* t = makeTxt(); 
    t->write(path); 
}



int BLog::Compare( const BLog* a , const BLog* b )
{
    unsigned ai = a->getNumKeys() ; 
    unsigned bi = b->getNumKeys() ; 
    unsigned ni = std::max( ai, bi ); 

    int RC = 0 ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
         int rc = 0 ;  

         const char* ak = i < ai ? a->getKey(i) : NULL ; 
         const char* bk = i < bi ? b->getKey(i) : NULL ; 
         bool mk = ak && bk && strcmp(ak, bk) == 0 ; 
         if( !mk ) rc |= 0x1 ;   


         double av      = i < ai ? a->getValue(i) : -1. ; 
         double bv      = i < bi ? b->getValue(i) : -1. ; 
         double dv      = av - bv ; 
         bool mv = std::fabs(dv) < TOLERANCE ; 
         if( !mv ) rc |= 0x10 ;   

         int ax = a->getSequenceIndex(i) ;   
         int bx = b->getSequenceIndex(i) ;   

         const char* marker = rc == 0 ? " " : "*" ; 

         std::cerr
              << " i " << std::setw(4) << i 
              << " rc " << std::setw(4) << std::hex << rc << std::dec 
              << " ak/bk " 
              << std::setw(40) << std::right << ak 
              << "/"
              << std::setw(40) << std::left << bk << std::right 
              << "  "
              << std::setw(1) << marker 
              << "  "
              << " ax/bx " 
              << std::setw(2)  << ax
              << "/"
              << std::setw(2)  << bx
              << "   " 
              << " av/bv " 
              << std::setw(12) << std::setprecision(10) << av
              << "/"
              << std::setw(12) << std::setprecision(10) << bv
              << "   " 
              << " dv " << std::setw(13) << std::setprecision(10) << dv
              << std::endl 
              ; 

         RC |= rc ;  
    }

    LOG(info) 
        << " ai " << ai  
        << " bi " << bi 
        << " RC " << RC 
        << " tol " << std::setw(13) << std::setprecision(10) << TOLERANCE
        ;

    return RC ; 
}



