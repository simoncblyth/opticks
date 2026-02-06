#pragma once
/**
SSimtrace : aiming to replace X4Simtrace
=============================================

SSimtrace.h is very local centric it populates a default sframe with ce
from the G4VSolid.

This is used from U4Tree::simtrace_scan (u4/test/U4SimtraceTest.sh)
for all distinct solids in a geometry saving the simtrace SEvt
with reldir for each solid name.

In addition stree::save_trs is used to save the placement transforms


Quick Check of simtrace positions
----------------------------------

For a quick check of simtrace positions, run the below from an SEvt folder::

    ~/opticks/sysrap/tests/SSimtrace_check.sh

That runs something like the below with ipython::

    #!/usr/bin/env python
    import os, numpy as np, matplotlib.pyplot as plt
    SIZE = np.array([1280, 720])

    title = os.getcwd()
    s = np.load("simtrace.npy")
    lpos = s[:,1,:3]

    fig,ax = plt.subplots(figsize=SIZE/100)
    ax.set_aspect('equal')
    fig.suptitle(title)
    ax.scatter( lpos[:,0], lpos[:,2] )
    fig.show()

**/
#include <cstring>
#include "plog/Severity.h"

#ifdef WITH_OLD_FRAME
#include "sframe.h"
#else
#include "sfr.h"
#endif

class G4VSolid ;
struct SEvt ;

struct SSimtrace
{
    static constexpr const plog::Severity LEVEL = info ;
    static void Scan( const G4VSolid* solid );

    const G4VSolid* solid ;
    SEvt* sev ;
#ifdef WITH_OLD_FRAME
    sframe frame = {} ;
#else
    sfr fr = {} ;
#endif

    SSimtrace();
    ~SSimtrace();

    void setSolid(const G4VSolid* solid);
    void simtrace();
};

#include "SEvt.hh"
#include "SEventConfig.hh"
#include "ssolid.h"
#include "SLOG.hh"
#include "NPFold.h"

inline void SSimtrace::Scan(const G4VSolid* solid )
{
    G4String soname_ = solid->GetName();
    const char* soname = soname_.c_str() ;
    //LOG(LEVEL) << "[ " << soname ;

    SEventConfig::SetEventReldir(soname);

    SSimtrace t ;
    t.setSolid(solid);
    t.simtrace();

    //LOG(LEVEL) << "] " << soname ;
}

inline SSimtrace::SSimtrace()
    :
    solid(nullptr),
    sev(nullptr)
{
}

inline SSimtrace::~SSimtrace()
{
    delete sev ;
}

inline void SSimtrace::setSolid(const G4VSolid* solid_)
{
    solid = solid_ ;

    float4 ce = {} ;

    ssolid::GetCenterExtent(ce, solid );

#ifdef WITH_OLD_FRAME
    frame.ce = ce ;
    LOG(LEVEL) << " frame.ce.w " << frame.ce.w ;
#else
    fr.set_ce( &ce.x );
    LOG(LEVEL) << " fr.ce.w " << fr.ce.w ;
#endif

}

/**
SSimtrace::simtrace
---------------------

SEvt::setFrame


1. creates gensteps with SFrameGenstep::MakeCenterExtentGensteps and adds them to SEvt
2. as frame.is_hostside_simtrace also generates simtrace "photons"


SEvt::beginOfEvent invokes SEvt::addFrameGenstep
which in RGModeSimtrace adds simtrace gensteps configured via envvars
especially::

CEGS : CenterExtentGensteps
    eg 16:0:9:5000 : specifies grid and photons per grid point

Because SSimtrace::simtrace gets called for each solid with U4SimtraceTest.sh
the instanciation of SEvt here in SSimtrace::simtrace is unusual,

With simulate running SEvt is usually only ever instanciated once.

**/

inline void SSimtrace::simtrace()
{
    SEventConfig::SetRGModeSimtrace();


#ifdef WITH_OLD_FRAME
    frame.set_hostside_simtrace();
#else
    fr.set_hostside_simtrace();
#endif

    // set_hostside_simtrace into frame which
    // influences the action of SEvt::setFrame
    // especially SEvt::setFrame_HostsideSimtrace which
    // generates simtrace photons

    sev = SEvt::Create_ECPU() ;

#ifdef WITH_OLD_FRAME
    sev->setFrame(frame);
#else
    sev->setFr(fr);
#endif

    int eventID = 0 ;

    sev->beginOfEvent(eventID);   // invokes SEvt::addFrameGenstep for RGModeSimtrace

    unsigned num_simtrace = sev->simtrace.size() ;
    LOG(LEVEL) << " num_simtrace " << num_simtrace ;

    bool dump = false ;
    for(unsigned i=0 ; i < num_simtrace ; i++)
    {
        quad4& p = sev->simtrace[i] ;
        ssolid::Simtrace(p, solid, dump);
    }

    sev->gather();
    sev->topfold->concat();
    sev->topfold->clear_subfold();

    sev->endOfEvent(eventID);
}


