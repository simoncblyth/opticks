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

#include "Opticks.hh"
#include "OpticksPhoton.hh"
#include "OpticksFlags.hh"

#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"
#include "OpticksEventStat.hh"
#include "OpticksEventCompare.hh"

#include "PLOG.hh"

OpticksEventCompare::OpticksEventCompare(OpticksEvent* a, OpticksEvent* b)
    :
    m_ok(a->getOpticks()),
    m_dbgseqhis(m_ok->getDbgSeqhis()),
    m_dbgseqmat(m_ok->getDbgSeqmat()),
    m_a(a),
    m_b(b),
    m_as(new OpticksEventStat(a,0)),   
    m_bs(new OpticksEventStat(b,0)),   
    m_ad(new OpticksEventDump(a)),   
    m_bd(new OpticksEventDump(b))   
{
}


void OpticksEventCompare::dump(const char* msg) const 
{
    LOG(info) << msg ;

    m_as->dump("A");
    m_bs->dump("B");

    dumpMatchedSeqHis() ;
}


void OpticksEventCompare::dumpMatchedSeqHis() const 
{
    unsigned na = m_a->getNumPhotons() ; 
    unsigned nb = m_b->getNumPhotons() ; 
    assert( na == nb );

    unsigned ab_count = 0 ; 
    unsigned a_count = 0 ; 
    unsigned b_count = 0 ; 

    for(unsigned i=0 ; i < na ; i++)
    {  
        unsigned long long a_seqhis = m_a->getSeqHis(i);
        unsigned long long b_seqhis = m_b->getSeqHis(i);

        if( a_seqhis == b_seqhis && a_seqhis == m_dbgseqhis )
        {    
            ab_count++ ; 
            if(ab_count < 10 )
            {
                LOG(info) << "OpticksEventCompare::dumpMatchedSeqHis AB " << ab_count ; 
                m_ad->dump(i);
                m_bd->dump(i);
            }
        }
        else if( a_seqhis == m_dbgseqhis )
        {
            a_count++ ; 
            if(a_count < 10 )
            {
                LOG(info) << "OpticksEventCompare::dumpMatchedSeqHis A " << a_count ; 
                m_ad->dump(i);
            }
        }
        else if( b_seqhis == m_dbgseqhis )
        {
            b_count++ ; 
            if(b_count < 10 )
            {
                LOG(info) << "OpticksEventCompare::dumpMatchedSeqHis B " << b_count ; 
                m_bd->dump(i);
            }
        }
    }

    LOG(info) << "OpticksEventCompare::dumpMatchedSeqHis"
              << " pho_num " << m_a->getNumPhotons()
              << " dbgseqhis " << std::hex << m_dbgseqhis << std::dec 
              << " dbgseqhis " << OpticksPhoton::FlagSequence( m_dbgseqhis, true )
              << " ab_count " << ab_count 
              << " a_count " << a_count 
              << " b_count " << b_count 
              ;

       
}




