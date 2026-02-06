#include "SLOG.hh"
#include "SEventConfig.hh"
#include "SSys.hh"
#include "SEvt.hh"
#include "SSim.hh"
#include "CSGFoundry.h"
#include "CSGSimtrace.hh"
#include "CSGQuery.h"
#include "CSGDraw.h"
#include "NP.hh"
#include "NPFold.h"

const plog::Severity CSGSimtrace::LEVEL = SLOG::EnvLevel("CSGSimtrace", "DEBUG");

int CSGSimtrace::Preinit()    // static
{
    SEventConfig::SetRGModeSimtrace();
    return 0 ;
}

CSGSimtrace::CSGSimtrace()
    :
    prc(Preinit()),
    geom(SSys::getenvvar("GEOM", "nmskSolidMaskTail")),
    sim(SSim::Create()),
    fd(CSGFoundry::Load()),
    sev(SEvt::Create_ECPU()),
    outdir(sev->getDir()),
    q(new CSGQuery(fd)),
    d(new CSGDraw(q,'Z')),
    SELECTION(getenv("SELECTION")),
    selection(SSys::getenvintvec("SELECTION",',')),  // when no envvar gives nullptr
    num_selection(selection && selection->size() > 0 ? selection->size() : 0 ),
    selection_simtrace(num_selection > 0 ? NP::Make<float>(num_selection, 4, 4) : nullptr ),
    qss(selection_simtrace ? (quad4*)selection_simtrace->bytes() : nullptr)
{
    init();
}

void CSGSimtrace::init()
{
    LOG(LEVEL) << d->desc();

    float4 ce = q->select_prim_ce ;
    LOG(LEVEL) << " ce " << ce  ;

#ifdef WITH_OLD_FRAME
    frame.set_hostside_simtrace();
    frame.ce = ce ;
    sev->setFrame(frame);
#else
    fr.set_hostside_simtrace();
    fr.set_ce( &ce.x );
    sev->setFr(fr);
#endif

    LOG(LEVEL) << " SELECTION " << SELECTION << " num_selection " << num_selection << " outdir " << outdir ;
    if(selection_simtrace)
    {
        selection_simtrace->set_meta<std::string>("SELECTION", SELECTION) ;
    }

}

/**
CSGSimtrace::simtrace
----------------------

Follow CPU side event handling from U4Recorder::BeginOfEventAction_ U4Recorder::EndOfEventAction_

**/

int CSGSimtrace::simtrace()
{
    int eventID = 0 ;

    sev->beginOfEvent(eventID);

    int num_intersect = qss ? simtrace_selection() : simtrace_all() ;

    sev->gather();
    sev->topfold->concat();
    sev->topfold->clear_subfold();

    sev->endOfEvent(eventID);   // save done in here

    LOG(LEVEL)
         << " outdir [" << ( outdir ? outdir : "-" ) << "]"
         << " num_selection " << num_selection
         << " selection_simtrace.sstr " << ( selection_simtrace ? selection_simtrace->sstr() : "-" )
         ;

    if(num_selection > 0)
    {
        selection_simtrace->save(outdir, "simtrace_selection.npy") ;
    }
    q->post(outdir);  // only DEBUG_CYLINDER

    return num_intersect ;

}

int CSGSimtrace::simtrace_all()
{
    int num_simtrace = sev->simtrace.size() ;
    int num_intersect = 0 ;
    for(int i=0 ; i < num_simtrace ; i++)
    {
        quad4& p = sev->simtrace[i] ;
        bool valid_intersect = q->simtrace(p);
        if(valid_intersect) num_intersect += 1 ;
    }
    LOG(LEVEL)
        << " num_simtrace " << num_simtrace
        << " num_intersect " << num_intersect
        ;
    return num_intersect ;
}

int CSGSimtrace::simtrace_selection()
{
    int num_intersect = 0 ;
    for(int i=0 ; i < num_selection ; i++)
    {
        int j = (*selection)[i] ;
        const quad4& p0 = sev->simtrace[j] ;
        quad4& p = qss[i] ;
        p = p0 ;
        bool valid_intersect = q->simtrace(p);
        if(valid_intersect) num_intersect += 1 ;
    }
    LOG(LEVEL)
        << " num_selection " << num_selection
        << " num_intersect " << num_intersect
        ;
    return num_intersect ;
}


