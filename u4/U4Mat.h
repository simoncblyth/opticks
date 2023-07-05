#pragma once
/**
U4Mat.h
========

TODO: change U4Material.hh into header only and integrate with this

**/

class G4Material ; 

struct U4Mat
{
    static const G4MaterialPropertyVector* GetRINDEX(   const G4Material* mt ); 
    static const G4MaterialPropertyVector* GetProperty( const G4Material* mt, int index ); 
}; 

inline const G4MaterialPropertyVector* U4Mat::GetRINDEX(  const G4Material* mt ) // static
{
    return GetProperty(mt, kRINDEX ); 
}
inline const G4MaterialPropertyVector* U4Mat::GetProperty(const G4Material* mt, int index ) // static
{
    G4MaterialPropertiesTable* mpt = mt ? mt->GetMaterialPropertiesTable() : nullptr ;
    const G4MaterialPropertyVector* mpv = mpt ? mpt->GetProperty(index) : nullptr ;    
    return mpv ; 
}

