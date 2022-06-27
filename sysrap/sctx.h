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

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SCTX_METHOD void zero(); 
#endif

#ifndef PRODUCTION
    SCTX_METHOD void point(int bounce); 
    SCTX_METHOD void trace(int bounce); 
    SCTX_METHOD void end(); 
#endif
}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
SCTX_METHOD void sctx::zero(){ *this = {} ; }
#endif


#ifndef PRODUCTION
/**
sctx::point : record current sphoton p into record/rec/seq 
---------------------------------------------------------------

As *prd* is updated by *trace* rather than *propagate* it is handled separately. 
Consider a history::

   TO->BT->BT->SC->AB

The *prd* corresponds to the arrows and corresponding trace that happens to get 
between the points. 

**/

SCTX_METHOD void sctx::point(int bounce)
{ 
    if(evt->record && bounce < evt->max_record) evt->record[evt->max_record*idx+bounce] = p ;   
    if(evt->rec    && bounce < evt->max_rec)    evt->add_rec( rec, idx, bounce, p );    // this copies into evt->rec array 
    if(evt->seq    && bounce < evt->max_seq)    seq.add_nibble( bounce, p.flag(), p.boundary() );  
}
SCTX_METHOD void sctx::trace(int bounce)
{
    if(evt->prd) evt->prd[evt->max_prd*idx+bounce] = *prd ; 
}
SCTX_METHOD void sctx::end()
{
    if( evt->seq) evt->seq[idx] = seq ; // Q: did I forget rec ? A: No. rec+record are added to evt->rec+record in sctx::point 
#ifdef DEBUG_TAG
    if(evt->tag)  evt->tag[idx]  = tagr.tag ;
    if(evt->flat) evt->flat[idx] = tagr.flat ;
#endif
}

#endif

