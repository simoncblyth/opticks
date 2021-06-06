#include <cassert>
#include <sstream>
#include "CPho.hh"

const unsigned CPho::MISSING = ~0u ; 

CPho::CPho()
    :
    gs(MISSING),
    ix(MISSING),
    id(MISSING),
    gn(MISSING)
{
}

CPho::CPho( unsigned gs_, unsigned ix_, unsigned id_, unsigned gn_ )
    :
    gs(gs_),
    ix(ix_),
    id(id_),
    gn(gn_)
{
}

/**
CPho::FabricateTrackIdPhoton
------------------------------

This is used from CPhotonInfo::Get for optical photon tracks that 
are not labelled.  As S+C photon tracks should always be labelled 
this should only be relevant to input "primary" photons.
The input photons are artificially contructed torch 'T' photons 
used for debugging.

**/

CPho CPho::FabricateTrackIdPhoton(unsigned track_id) // static
{
    CPho fab(0, track_id, track_id, 0) ; 
    return fab ; 
}



bool CPho::is_missing() const { return gs == MISSING ; }

int CPho::get_gs() const { return gs == MISSING ? -1 : gs ; }
int CPho::get_ix() const { return ix == MISSING ? -1 : ix ; }
int CPho::get_id() const { return id == MISSING ? -1 : id ; }
int CPho::get_gn() const { return gn == MISSING ? -1 : gn ; }

std::string CPho::desc() const 
{ 
    std::stringstream ss ; 
    ss << "CPho" ; 
    if(is_missing())
    {
        ss << " (missing) " ; 
    }
    else
    {
        ss << " gs " << gs << " ix " << ix << " id " << id << " gn " << gn ; 
    }
    std::string s = ss.str(); 
    return s ; 
}


CPho CPho::make_reemit() const
{
    assert(!is_missing());
    CPho reemit(gs, ix, id, gn+1) ; 
    return reemit ; 
}




