#pragma once
/**
STrackInfo.h (formerly U4TrackInfo.h)
========================================

Formerly uses templated STrackInfo but as
doing dynamic cast on such a type is not possible
that is dangerous as must rely on no other track
info subclasses being in use.

Required methods for spho::

   std::string desc() const ;
   spho spho::Placeholder() ;
   spho spho::Fabricate(int id) ;


Users::

    epsilon:opticks blyth$ opticks-fl STrackInfo
    ./sysrap/CMakeLists.txt
    ./sysrap/STrackInfo.h
    ./sysrap/SFastSimOpticalModel.hh

    ./u4/U4Track.h

    ./u4/tests/U4TrackTest.cc
    ./u4/tests/U4TrackInfoTest.cc

    ./u4/U4Recorder.cc
         vital part of U4Recorder::PreUserTrackingAction_Optical PostUserTrackingAction_Optical

    ./u4/InstrumentedG4OpBoundaryProcess.cc
         not fully impl, seems informational only

    ./u4/U4.cc
         setting photon labels at generation


::

    epsilon:PMTFastSim blyth$ grep STrackInfo.h *.*
    junoPMTOpticalModel.cc:#include "STrackInfo.h"
    junoPMTOpticalModel.rst:* instead replaced with passing the FastSim status via trackinfo with sysrap/STrackInfo.h
    junoPMTOpticalModelSimple.cc:#include "STrackInfo.h"
    epsilon:PMTFastSim blyth$


**/

#include <string>
#include "G4Track.hh"
#include "G4VUserTrackInformation.hh"

#include "spho.h"

struct STrackInfo : public G4VUserTrackInformation
{
    spho label  ;

    STrackInfo(const spho& label);
    std::string desc() const ;

    static STrackInfo* GetTrackInfo_UNDEFINED(const G4Track* track);
    static STrackInfo* GetTrackInfo(          const G4Track* track);
    static bool Exists(const G4Track* track);
    static spho  Get(   const G4Track* track);   // by value
    static spho* GetRef(const G4Track* track);   // by reference, allowing inplace changes
    static std::string Desc(const G4Track* track);

    static void Set(G4Track* track, const spho& label );
};

inline STrackInfo::STrackInfo(const spho& _label )
    :
    G4VUserTrackInformation("STrackInfo"),
    label(_label)
{
}

inline std::string STrackInfo::desc() const
{
    std::stringstream ss ;
    ss << *pType << " " << label.desc() ;
    std::string s = ss.str();
    return s ;
}

/**
STrackInfo::GetTrackInfo_UNDEFINED
-----------------------------------

With U4PhotonInfo the ancestor of STrackInfo was using dynamic_cast
without issue. After moving to the templated STrackInfo the
dynamic cast always giving nullptr. So switched to static_cast.

**/

inline STrackInfo* STrackInfo::GetTrackInfo_UNDEFINED(const G4Track* track) // static, label by value
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    STrackInfo* trackinfo = ui ? static_cast<STrackInfo*>(ui) : nullptr ;
    return trackinfo ;
}

inline STrackInfo* STrackInfo::GetTrackInfo(const G4Track* track) // static, label by value
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    STrackInfo* trackinfo = ui ? dynamic_cast<STrackInfo*>(ui) : nullptr ;
    return trackinfo ;
}




inline bool STrackInfo::Exists(const G4Track* track) // static
{
    STrackInfo* trackinfo = GetTrackInfo(track);
    return trackinfo != nullptr ;
}

inline spho STrackInfo::Get(const G4Track* track) // static, label by value
{
    STrackInfo* trackinfo = GetTrackInfo(track);
    return trackinfo ? trackinfo->label : spho::Placeholder() ;
}

inline spho* STrackInfo::GetRef(const G4Track* track) // static, label reference
{
    STrackInfo* trackinfo = GetTrackInfo(track);
    return trackinfo ? &(trackinfo->label) : nullptr ;
}

inline std::string STrackInfo::Desc(const G4Track* track)
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;

    STrackInfo* trackinfo = GetTrackInfo(track);
    STrackInfo* trackinfo_static = GetTrackInfo_UNDEFINED(track);

    std::stringstream ss ;
    ss << "STrackInfo::Desc"
       << std::endl
       << " track " << track
       << " track.GetUserInformation " << ui
       << std::endl
       << " trackinfo " << trackinfo
       << " trackinfo_static " << trackinfo_static
       << std::endl
       << " trackinfo.desc " << ( trackinfo ? trackinfo->desc() : "-" )
       << std::endl
       << " trackinfo_static.desc " << ( trackinfo_static ? trackinfo_static->desc() : "-" )
       ;
    std::string s = ss.str();
    return s ;
}


inline void STrackInfo::Set(G4Track* track, const spho& _label )  // static
{
    spho* label = GetRef(track);
    if(label == nullptr)
    {
        track->SetUserInformation(new STrackInfo(_label));
    }
    else
    {
        *label = _label ;
    }
}

