#pragma once
/**
STrackInfo.h (formerly U4TrackInfo.h)
========================================

Required methods for T::

   std::string desc() const ;  
   T T::Placeholder() ; 
   T T::Fabricate(int id) ; 


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

template<typename T>
struct STrackInfo : public G4VUserTrackInformation
{
    T label  ; 

    STrackInfo(const T& label); 
    std::string desc() const ; 

    static STrackInfo<T>* GetTrackInfo(const G4Track* track); 
    static STrackInfo<T>* GetTrackInfo_dynamic(const G4Track* track); 
    static bool Exists(const G4Track* track); 
    static T  Get(   const G4Track* track);   // by value 
    static T* GetRef(const G4Track* track);   // by reference, allowing inplace changes
    static std::string Desc(const G4Track* track); 

    static void Set(G4Track* track, const T& label ); 
};

template<typename T>
inline STrackInfo<T>::STrackInfo(const T& _label )
    :   
    G4VUserTrackInformation("STrackInfo"),
    label(_label)
{
}
 
template<typename T>
inline std::string STrackInfo<T>::desc() const 
{
    std::stringstream ss ; 
    ss << *pType << " " << label.desc() ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
STrackInfo::GetTrackInfo
--------------------------

With U4PhotonInfo the ancestor of STrackInfo was using dynamic_cast 
without issue. After moving to the templated STrackInfo the 
dynamic cast always giving nullptr. So switched to static_cast. 

**/

template<typename T>
inline STrackInfo<T>* STrackInfo<T>::GetTrackInfo(const G4Track* track) // static, label by value 
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    STrackInfo<T>* trackinfo = ui ? static_cast<STrackInfo<T>*>(ui) : nullptr ;
    return trackinfo ; 
}

template<typename T>
inline STrackInfo<T>* STrackInfo<T>::GetTrackInfo_dynamic(const G4Track* track) // static, label by value 
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    STrackInfo<T>* trackinfo = ui ? dynamic_cast<STrackInfo<T>*>(ui) : nullptr ;
    return trackinfo ; 
}




template<typename T>
inline bool STrackInfo<T>::Exists(const G4Track* track) // static
{
    STrackInfo<T>* trackinfo = GetTrackInfo(track); 
    return trackinfo != nullptr ; 
}

template<typename T>
inline T STrackInfo<T>::Get(const G4Track* track) // static, label by value 
{
    STrackInfo<T>* trackinfo = GetTrackInfo(track); 
    return trackinfo ? trackinfo->label : T::Placeholder() ; 
}

template<typename T>
inline T* STrackInfo<T>::GetRef(const G4Track* track) // static, label reference 
{
    STrackInfo<T>* trackinfo = GetTrackInfo(track); 
    return trackinfo ? &(trackinfo->label) : nullptr ; 
}

template<typename T>
inline std::string STrackInfo<T>::Desc(const G4Track* track)
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;

    STrackInfo<T>* trackinfo = GetTrackInfo(track); 
    STrackInfo<T>* trackinfo_dyn = GetTrackInfo_dynamic(track); 

    std::stringstream ss ; 
    ss << "STrackInfo::Desc" 
       << std::endl 
       << " track " << track 
       << " track.GetUserInformation " << ui
       << std::endl 
       << " trackinfo " << trackinfo 
       << " trackinfo_dyn " << trackinfo_dyn 
       << std::endl 
       << " trackinfo.desc " << ( trackinfo ? trackinfo->desc() : "-" )
       << std::endl 
       << " trackinfo_dyn.desc " << ( trackinfo_dyn ? trackinfo_dyn->desc() : "-" )
       ; 
    std::string s = ss.str(); 
    return s ; 
}  


template<typename T>
inline void STrackInfo<T>::Set(G4Track* track, const T& _label )  // static 
{
    T* label = GetRef(track); 
    if(label == nullptr)
    {
        track->SetUserInformation(new STrackInfo<T>(_label)); 
    }
    else
    {
        *label = _label ; 
    }
}

