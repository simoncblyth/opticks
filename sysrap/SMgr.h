#pragma once
/**
SMgr.h
=======

UNcompleted thoughts on evt level concurrency, from a top down viewpoint.

* actually top-down thinking is not very productive so far.
* bottom-up in tests/SPM_test.sh is much more tractable
  as can immediately test : pattern is of spreading use of
  streams and returning future objects

* eventually will need top-down and well as bottom up, but
  there is lots more threading of streams thru lots of code
  and lots of waiting on future objects to sort out before
  need top down coordination



**/


#include <vector>
#include "SEvt.hh"

struct SLaunchContext
{
    cudaStream_t   stream           = nullptr;
    cudaEvent_t    photons_done     = nullptr;   // after last propagation launch
    cudaEvent_t    hits_merged_done = nullptr;   // after final merge
    int            qevt_slot        = -1;        // index in the pool (for debugging)
    bool           in_use           = false;

    bool is_inited() const;
    void init();
};

inline bool SLaunchContext::is_inited() const
{
    return stream != nullptr ;
}

inline void SLaunchContext::init()
{
    // create per-context resources (only once per context)
    if(is_inited()) return ;
    cudaStreamCreate(&stream);
    cudaEventCreate(&photons_done);
    cudaEventCreate(&hits_merged_done);
}






struct SMgr
{
    static SMgr*  INSTANCE ;
    static SMgr*  Get();

    SMgr();

    // ideas for evt level concurrency
    std::vector<SLaunchContext> contexts;             // size = max_concurrency (dynamic)
    int                         max_concurrent = 6;    // runtime adjustable
    int                         current_concurrency = 1;

    void                        configure_max_concurrency(int n);
    int                         choose_concurrency_level(size_t num_photons);
    SLaunchContext*             acquire_context(size_t num_photons);
    SLaunchContext*             get_context_for_current_evt();
    void                        initEvent();
    void                        finalizeEvent();

    SEvt*                       sev ;
    static cudaStream_t         Stream();
};


inline SMgr* SMgr::INSTANCE = nullptr ;
inline SMgr* SMgr::Get()
{
    if(!INSTANCE) new SMgr ;
    return INSTANCE ;
}

inline SMgr::SMgr()
{
    INSTANCE = this ;
}


inline void SMgr::configure_max_concurrency(int n)
{
    max_concurrent = n;
}


inline SLaunchContext* SMgr::get_context_for_current_evt()
{
    // HMM: which is current when multiple ?
    SEvt* sev = SEvt::Get_EGPU();
    return static_cast<SLaunchContext*>(sev->async_handle);
}

inline int SMgr::choose_concurrency_level(size_t num_photons)
{
    return 1 ;
}

inline SLaunchContext* SMgr::acquire_context(size_t num_photons)
{
    //int allowed = choose_concurrency_level(num_photons);
    SLaunchContext* ctx = nullptr;
    while (!ctx)
    {
        // if one not in_use set ctx
        for (auto& c : contexts)
        {
            if (!c.in_use) { ctx = &c; break; }
        }

        // add default context when less than max
        if (!ctx && contexts.size() < (size_t)max_concurrent)
        {
            contexts.emplace_back(); // add default context
            ctx = &contexts.back();
        }
        // if still no context sleep, before trying again
        if (!ctx)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    ctx->in_use = true;
    current_concurrency = std::max(current_concurrency, (int)contexts.size());
    return ctx;
}


inline void SMgr::initEvent()  // HMM having gensteps arg would be clearer
{
    sev = SEvt::Create_EGPU();

    size_t num_photons_this_event = 0 ; // placeholder

    SLaunchContext* ctx = acquire_context(num_photons_this_event);
    sev->async_handle = ctx;

    if(!ctx->is_inited()) ctx->init(); // create per-context resources (only once per context)
}

inline void SMgr::finalizeEvent()
{
    if (sev && sev->async_handle)
    {
        SLaunchContext* ctx = (SLaunchContext*)sev->async_handle;
        ctx->in_use = false;
        sev->async_handle = nullptr;
    }
    // more cleanup
}

inline cudaStream_t SMgr::Stream()  // static
{
    SMgr* mgr = SMgr::Get();
    if (!mgr) return 0;    // fallback for old paths
    SLaunchContext* ctx = (SLaunchContext*)mgr->sev->async_handle;
    return ctx ? ctx->stream : 0;
}


