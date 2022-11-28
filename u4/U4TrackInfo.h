#pragma once
/**
U4TrackInfo.h
================

Required methods for T::

   std::string desc() const ;  
   T T::Placeholder() ; 

**/

#include <string>
#include "G4Track.hh"
#include "G4VUserTrackInformation.hh"

template<typename T>
struct U4TrackInfo : public G4VUserTrackInformation
{
    T label  ; 

    U4TrackInfo(const T& label); 
    std::string desc() const ; 

    static bool Exists(const G4Track* track); 
    static T  Get(   const G4Track* track);   // by value 
    static T* GetRef(const G4Track* track);   // by reference, allowing inplace changes

    //static int GetIndex(const  G4Track* track);
    static void Set(G4Track* track, const T& label ); 
};

template<typename T>
inline U4TrackInfo<T>::U4TrackInfo(const T& _label )
    :   
    G4VUserTrackInformation("U4TrackInfo"),
    label(_label)
{
}
 
template<typename T>
inline std::string U4TrackInfo<T>::desc() const 
{
    std::stringstream ss ; 
    ss << *pType << " " << label.desc() ; 
    std::string s = ss.str(); 
    return s ; 
}

template<typename T>
inline bool U4TrackInfo<T>::Exists(const G4Track* track) // static
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    U4TrackInfo<T>* pin = ui ? dynamic_cast<U4TrackInfo<T>*>(ui) : nullptr ;
    return pin != nullptr ; 
}

template<typename T>
inline T U4TrackInfo<T>::Get(const G4Track* track) // static, label by value 
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    U4TrackInfo<T>* pin = ui ? dynamic_cast<U4TrackInfo<T>*>(ui) : nullptr ;
    return pin ? pin->label : T::Placeholder() ; 
}

template<typename T>
inline T* U4TrackInfo<T>::GetRef(const G4Track* track) // static, label reference 
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    U4TrackInfo<T>* pin = ui ? dynamic_cast<U4TrackInfo<T>*>(ui) : nullptr ;
    return pin ? &(pin->label) : nullptr ; 
}

/*
template<typename T>
inline int U4TrackInfo<T>::GetIndex(const G4Track* track)
{
    T label = Get(track); 
    return label.id ;    // a bit much requiring id field
}
*/

template<typename T>
inline void U4TrackInfo<T>::Set(G4Track* track, const T& _label )  // static 
{
    T* label = GetRef(track); 

    if(label) std::cout << " init: label.desc " << label->desc() << std::endl ; 
  
    if(label == nullptr)
    {
        track->SetUserInformation(new U4TrackInfo<T>(_label)); 

        T* label1 = GetRef(track);  
        if(label1) std::cout << " initial set : label1.desc " << label1->desc() << std::endl ; 

    }
    else
    {
        *label = _label ; 
        std::cout << " change : label.desc " << label->desc() << std::endl ; 
    }
}

