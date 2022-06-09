#pragma once
/**
spho.h : photon labelling used by genstep collection
========================================================

After cfg4/CPho

isSameLineage 
    does not require the same reemission generation

isIdentical
    requires isSameLineage and same reemission generation

* HMM: spho lacks gentype, to get that must reference corresponding sgs struct using the gs index 

**/

#include <string>

struct spho
{
    int gs ; // 0-based genstep index within the event
    int ix ; // 0-based photon index within the genstep
    int id ; // 0-based photon identity index within the event 
    int gn ; // 0-based reemission index incremented at each reemission 

    static spho MakePho(int gs_, int ix_, int id_, int gn_); 
    static spho Fabricate(int track_id); 
    static spho Placeholder() ; 

    bool isSameLineage(const spho& other) const { return gs == other.gs && ix == other.ix && id == other.id ; }
    bool isIdentical(const spho& other) const { return isSameLineage(other) && gn == other.gn ; }

    bool isPlaceholder() const { return gs == -1 ; }
    bool isDefined() const { return gs != -1 ; }

    spho make_reemit() const ; 
    std::string desc() const ;
};


#include <cassert>
#include <sstream>
#include <iomanip>




inline spho spho::MakePho(int gs_, int ix_, int id_, int gn_) // static
{
    spho ph = {gs_, ix_, id_, gn_} ; 
    return ph ;   
}
/**
spho::Fabricate
---------------

*Fabricate* is not normally used, as C+S photons are always 
labelled at generation by U4::GenPhotonEnd

However as a workaround for torch/input photons that lack labels
this method is used from U4Recorder::PreUserTrackingAction_Optical
to provide a standin label based only on a 0-based track_id. 

**/
inline spho spho::Fabricate(int track_id) // static
{
    assert( track_id >= 0 ); 
    spho fab = {0, track_id, track_id, 0 };
    return fab ;
}
inline spho spho::Placeholder() // static
{
    spho inv = {-1, -1, -1, -1 };
    return inv ;
}
inline spho spho::make_reemit() const
{
    spho reemit = {gs, ix, id, gn+1 } ;
    return reemit ;
}

inline std::string spho::desc() const
{
    std::stringstream ss ;
    ss << "spho" ;
    if(isPlaceholder())  ss << " isPlaceholder " ; 
    else ss << " ( gs ix id gn " 
            << std::setw(3) << gs 
            << std::setw(4) << ix 
            << std::setw(5) << id 
            << std::setw(2) << gn
            << " ) "
            ;
    std::string s = ss.str();
    return s ;
}

