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


#include "BStr.hh"

#include "OpStatus.hh"
#include "OpticksFlags.hh"
#include "Opticks.hh"

#include "CCtx.hh"
#include "CPhoton.hh"
#include "CRecorder.hh"
#include "CRec.hh"
#include "CMaterialBridge.hh"
#include "CDebug.hh"
#include "Format.hh"

#include "PLOG.hh"


CDebug::CDebug(CCtx& ctx, const CPhoton& photon, CRecorder* recorder)
    :
    m_ctx(ctx),
    m_ok(m_ctx.getOpticks()),
    m_verbosity(m_ok->getVerbosity()),
    m_recorder(recorder),
    m_crec(recorder->getCRec()),
    m_material_bridge(NULL),
    m_photon(photon),
    m_seqhis_select(0x8bd),
    m_seqmat_select(0),
    m_dbgseqhis(m_ok->getDbgSeqhis()),
    m_dbgseqmat(m_ok->getDbgSeqmat()),
    m_dbgflags(m_ok->hasOpt("dbgflags")),

    m_posttrack_dbgzero(false),
    m_posttrack_dbgseqhis(false),
    m_posttrack_dbgseqmat(false)
{
}

void CDebug::setMaterialBridge(const CMaterialBridge* material_bridge) 
{
    m_material_bridge = material_bridge ; 
}


bool CDebug::isHistorySelected()
{
   return m_seqhis_select == m_photon._seqhis ; 
}
bool CDebug::isMaterialSelected()
{
   return m_seqmat_select == m_photon._seqmat ; 
}
bool CDebug::isSelected()
{
   return isHistorySelected() || isMaterialSelected() ;
}


std::string CDebug::desc() const   // reason for the dump
{
    std::stringstream ss ; 
    ss << "CDebug"
       << " " << ( m_posttrack_dump ? "posttrack_dump" : " nodump " )
       << " " << ( m_posttrack_dbgzero ? "posttrack_dbgzero" : " - " )
       << " " << ( m_posttrack_dbgseqhis ? "posttrack_dbgseqhis" : " - " )
       << " " << ( m_posttrack_dbgseqmat ? "posttrack_dbgseqmat" : " - " )
       << " " << ( m_photon._badflag > 0 ? " badflag " : " - " )
       << " " << ( m_ctx._debug > 0 ? " _debug " : " - " )
       << " " << ( m_ctx._other > 0 ? " _other " : " - " )
       << " " << ( m_verbosity > 0 ? " verbosity " : " - " )
       ;
    return ss.str();
}


void CDebug::postTrack()
{
    if(m_photon._badflag > 0) addDebugPhoton(m_ctx._record_id);  

    m_posttrack_dbgzero = m_photon._seqhis == 0 || m_photon._seqmat == 0 ; 
    m_posttrack_dbgseqhis = m_dbgseqhis == m_photon._seqhis ; 
    m_posttrack_dbgseqmat = m_dbgseqmat == m_photon._seqmat ; 

    m_posttrack_dump = m_verbosity > 0 || 
                       m_posttrack_dbgzero || 
                       m_posttrack_dbgseqhis || 
                       m_posttrack_dbgseqmat || 
                       m_ctx._other || 
                       m_ctx._debug || 
                       m_photon._badflag > 0  ;


    //LOG(info) << "CDebug::posttrack " << desc() ;  

    if(m_posttrack_dump) dump("CDebug::postTrack");
}






#ifdef USE_CUSTOM_BOUNDARY
void CDebug::Collect(const G4StepPoint* point, Ds::DsG4OpBoundaryProcessStatus boundary_status, const CPhoton& photon )
#else
void CDebug::Collect(const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, const CPhoton& photon )
#endif
{
    double time = point->GetGlobalTime();

    m_points.push_back(new G4StepPoint(*point));
    m_flags.push_back(photon._flag);
    m_materials.push_back(photon._material);
    m_bndstats.push_back(boundary_status);  // will duplicate the status for the last step
    m_mskhis_dbg.push_back(photon._mskhis);
    m_seqhis_dbg.push_back(photon._seqhis);
    m_seqmat_dbg.push_back(photon._seqmat);
    m_times.push_back(time);
}


void CDebug::Clear()
{
    for(unsigned int i=0 ; i < m_points.size() ; i++) delete m_points[i] ;
    m_points.clear();
    m_flags.clear();
    m_materials.clear();
    m_bndstats.clear();
    m_seqhis_dbg.clear();
    m_seqmat_dbg.clear();
    m_mskhis_dbg.clear();
    m_times.clear();
}



bool CDebug::hasIssue()
{
    unsigned int npoints = m_points.size() ;
    assert(m_flags.size() == npoints);
    assert(m_materials.size() == npoints);
    assert(m_bndstats.size() == npoints);

    bool issue = false ; 
    for(unsigned int i=0 ; i < npoints ; i++) 
    {
       if(m_flags[i] == 0 || m_flags[i] == NAN_ABORT) issue = true ; 
    }
    return issue ; 
}



void CDebug::dump(const char* msg)
{
    LOG(info) << msg ; 

    m_crec->dump("CDebug::dump");

    dump_brief("CRecorder::dump_brief");

    //if(m_ctx._debug || m_ctx._other ) 
    {
        dump_sequence("CDebug::dump_sequence");
        dump_points("CDeug::dump_points");
    }
}



void CDebug::dump_brief(const char* msg)
{
    LOG(info) << msg 
              << " m_ctx._record_id " << std::setw(8) << m_ctx._record_id 
              << " m_photon._badflag " << std::setw(5) << m_photon._badflag 
              << (m_ctx._debug ? " --dindex " : "" )
              << (m_ctx._other ? " --oindex " : "" )
              << (m_dbgseqhis == m_photon._seqhis ? " --dbgseqhis " : "" )
              << (m_dbgseqmat == m_photon._seqmat ? " --dbgseqmat " : "" )
              << " sas: " << m_recorder->getStepActionString()  // huh : why the live step action ?
              ;
    LOG(info) 
              << " seqhis " << std::setw(16) << std::hex << m_photon._seqhis << std::dec 
              << "    " << OpticksFlags::FlagSequence(m_photon._seqhis, true) 
              ;

    LOG(info) 
              << " mskhis " << std::setw(16) << std::hex << m_photon._mskhis << std::dec 
              << "    " << OpticksFlags::FlagMask(m_photon._mskhis, true) 
              ;

    LOG(info) 
              << " seqmat " << std::setw(16) << std::hex << m_photon._seqmat << std::dec 
              << "    " << m_material_bridge->MaterialSequence(m_photon._seqmat) 
              ;
}

void CDebug::dump_sequence(const char* msg)
{
    LOG(info) << msg ; 
    unsigned npoints = m_points.size() ;
    for(unsigned int i=0 ; i<npoints ; i++) 
        std::cout << std::setw(4) << i << " "
                  << std::setw(16) << std::hex << m_seqhis_dbg[i] << std::dec 
                  << " " << OpticksFlags::FlagSequence(m_seqhis_dbg[i], true)
                  << std::endl 
                  ;

    for(unsigned int i=0 ; i<npoints ; i++) 
        std::cout << std::setw(4) << i << " "
                  << std::setw(16) << std::hex << m_mskhis_dbg[i] << std::dec 
                  << " " << OpticksFlags::FlagMask(m_mskhis_dbg[i], true)
                  << std::endl 
                  ;

    for(unsigned int i=0 ; i<npoints ; i++) 
        std::cout << std::setw(4) << i << " "
                  << std::setw(16) << std::hex << m_seqmat_dbg[i] << std::dec 
                  << " " << m_material_bridge->MaterialSequence(m_seqmat_dbg[i]) 
                  << std::endl 
                  ;
}

void CDebug::dump_points(const char* msg)
{
    LOG(info) << msg ; 
    G4ThreeVector origin ;
    unsigned npoints = m_points.size() ;
    if(npoints > 0) origin = m_points[0]->GetPosition();

    for(unsigned int i=0 ; i<npoints ; i++) 
    {
        unsigned mat = m_materials[i] ;
        const char* matname = ( mat == 0 ? "-" : m_material_bridge->getMaterialName(mat-1)  ) ;
        dump_point(origin, i, m_points[i], m_bndstats[i], m_flags[i], matname );
    }
}


#ifdef USE_CUSTOM_BOUNDARY
void CDebug::dump_point(const G4ThreeVector& origin, unsigned index, const G4StepPoint* point, Ds::DsG4OpBoundaryProcessStatus boundary_status, unsigned flag, const char* matname )
#else
void CDebug::dump_point(const G4ThreeVector& origin, unsigned index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, unsigned flag, const char* matname )
#endif
{
    std::string bs = OpStatus::OpBoundaryAbbrevString(boundary_status) ;
    const char* flg = OpticksFlags::Abbrev(flag) ;
    std::cout << std::setw(3) << flg << std::setw(7) << index << " " << std::setw(18) << matname << " " << Format(point, origin, bs.c_str()) << std::endl ;
}



void CDebug::dumpStepVelocity(const char* msg )
{
    // try to understand GlobalTime calc from G4Transportation::AlongStepDoIt by duping attempt
    // issue is what velocity it gets to use, and the updating of that 

    G4Track* track = m_ctx._step->GetTrack() ;
    G4double trackStepLength = track->GetStepLength();
    G4double trackGlobalTime = track->GetGlobalTime() ;
    G4double trackVelocity = track->GetVelocity() ;

    const G4StepPoint* pre  = m_ctx._step->GetPreStepPoint() ; 
    const G4StepPoint* post = m_ctx._step->GetPostStepPoint() ; 


    G4double preDeltaTime = 0.0 ; 
    G4double preVelocity = pre->GetVelocity();
    if ( preVelocity > 0.0 )  { preDeltaTime = trackStepLength/preVelocity; }

    G4double postDeltaTime = 0.0 ; 
    G4double postVelocity = post->GetVelocity();
    if ( postVelocity > 0.0 )  { postDeltaTime = trackStepLength/postVelocity; }


    LOG(info) << msg
              << " trackStepLength " << std::setw(10) << trackStepLength 
              << " trackGlobalTime " << std::setw(10) << trackGlobalTime
              << " trackVelocity " << std::setw(10) << trackVelocity
              << " preVelocity " << std::setw(10) << preVelocity
              << " postVelocity " << std::setw(10) << postVelocity
              << " preDeltaTime " << std::setw(10) << preDeltaTime
              << " postDeltaTime " << std::setw(10) << postDeltaTime
              ;

}







void CDebug::addSeqhisMismatch(unsigned long long rdr, unsigned long long rec)
{
    m_seqhis_mismatch.push_back(std::pair<unsigned long long, unsigned long long>(rdr, rec));
}
void CDebug::addSeqmatMismatch(unsigned long long rdr, unsigned long long rec)
{
    m_seqmat_mismatch.push_back(std::pair<unsigned long long, unsigned long long>(rdr, rec));
}
void CDebug::addDebugPhoton(int record_id)
{
    m_debug_photon.push_back(record_id);
}




void CDebug::report(const char* msg)
{
     LOG(info) << msg ;
     unsigned cut = 50 ; 

     typedef std::vector<std::pair<unsigned long long, unsigned long long> >  VUU ; 
   
     unsigned nhis = m_seqhis_mismatch.size() ;
     unsigned ihis(0); 
     LOG(info) << " seqhis_mismatch " << nhis ;
     for(VUU::const_iterator it=m_seqhis_mismatch.begin() ; it != m_seqhis_mismatch.end() ; it++)
     { 
          ihis++ ;
          if(ihis < cut || ihis > nhis - cut )
          {
              unsigned long long rdr = it->first ;
              unsigned long long rec = it->second ;
              std::cout 
                        << " ihis " << std::setw(10) << ihis
                        << " rdr " << std::setw(16) << std::hex << rdr << std::dec
                        << " rec " << std::setw(16) << std::hex << rec << std::dec
                    //    << " rdr " << std::setw(50) << OpticksFlags::FlagSequence(rdr)
                    //    << " rec " << std::setw(50) << OpticksFlags::FlagSequence(rec)
                        << std::endl ; 
          }
          else if(ihis == cut)
          {
                std::cout << " ... " << std::endl ; 
          }
     }

     unsigned nmat = m_seqmat_mismatch.size() ;
     unsigned imat(0); 
     LOG(info) << " seqmat_mismatch " << nmat ;
     for(VUU::const_iterator it=m_seqmat_mismatch.begin() ; it != m_seqmat_mismatch.end() ; it++)
     {
          imat++ ; 
          if(imat < cut || imat > nmat - cut)
          {
              unsigned long long rdr = it->first ;
              unsigned long long rec = it->second ;
              std::cout 
                        << " imat " << std::setw(10) << imat
                        << " rdr " << std::setw(16) << std::hex << rdr << std::dec
                        << " rec " << std::setw(16) << std::hex << rec << std::dec
                        << " rdr " << std::setw(50) << m_material_bridge->MaterialSequence(rdr)
                        << " rec " << std::setw(50) << m_material_bridge->MaterialSequence(rec)
                        << std::endl ; 
           } 
           else if(imat == cut)
           {
                std::cout << " ... " << std::endl ; 
           }
     }


     unsigned ndbg = m_debug_photon.size() ;
     LOG(info) << " debug_photon " << ndbg << " (photon_id) " ; 
     typedef std::vector<int> VI ; 
     if(ndbg < 100) 
     for(VI::const_iterator it=m_debug_photon.begin() ; it != m_debug_photon.end() ; it++) std::cout << std::setw(8) << *it << std::endl ; 

     LOG(info) << "TO DEBUG THESE USE:  --dindex=" << BStr::ijoin(m_debug_photon, ',') ;

}




