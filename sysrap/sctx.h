#pragma once
/**
sctx.h : holding "thread local" state
=========================================

Q: why not keep such state in sevent.h *evt* ?
A: this state must be "thread" local, whereas the evt instance 
   is shared by all threads and always saves into (idx, bounce) 
   slotted locations   

This is aiming to avoid non-production instrumentation costing anything 
in production running by simply removing it from the context via the 
PRODUCTION macro. 
**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SCTX_METHOD __device__ __forceinline__
#else
#    define SCTX_METHOD inline 
#endif


struct sevent ; 
struct quad2 ; 
struct sphoton ; 
struct sstate ; 

#ifndef PRODUCTION
struct srec ; 
struct sseq ; 
struct stagr ;  
#endif

struct sctx
{
    sevent* evt ; 
    const quad2* prd ; 
    unsigned idx ; 

    sphoton p ; 
    sstate  s ; 

#ifndef PRODUCTION
    srec rec ; 
    sseq seq ; 
    stagr tagr ; 
#endif


#ifndef PRODUCTION
    SCTX_METHOD void point(int bounce); 
#endif
    SCTX_METHOD void end(int bounce); 
}; 


#ifndef PRODUCTION
SCTX_METHOD void sctx::point(int bounce)
{ 
    if(evt->record) evt->record[evt->max_record*idx+bounce] = p ;   
    if(evt->rec) evt->add_rec( rec, idx, bounce, p );  
    if(evt->seq) seq.add_nibble( bounce, p.flag(), p.boundary() );  
    if(evt->prd) evt->prd[evt->max_prd*idx+bounce] = *prd ; 
}
#endif

/**
sctx::end
------------

Note no prd saving here as that is unchanged by propagation

**/

SCTX_METHOD void sctx::end(int bounce)
{
#ifndef PRODUCTION
    if( evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;
    if( evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p );
    if( evt->seq    && bounce < evt->max_seq    ) seq.add_nibble(bounce, p.flag(), p.boundary() );

    if( evt->seq) evt->seq[idx] = seq ;
#ifdef DEBUG_TAG
    if(evt->tag)  evt->tag[idx]  = tagr.tag ;
    if(evt->flat) evt->flat[idx] = tagr.flat ;
#endif
#endif
    evt->photon[idx] = p ;
}


