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


sctx::aux usually accessed via SEvt::current_aux 
-------------------------------------------------

TODO : document the current_aux content here 

+----+
|    |
+====+
| q0 |
+----+
| q1 |
+----+
| q2 |    
+----+
| q3 |
+----+

q2.i.z : formerly fakemask, now uc4packed from the spho label
q2.i.w : st 


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
struct quad4 ; 
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
    quad4 aux ; 
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
sctx::point : copy current sphoton p into (idx,bounce) entries of evt->record/rec/seq/aux 
-------------------------------------------------------------------------------------------

Notice this is NOT writing into evt->photon, that is done at SEvt::finalPhoton

**/

SCTX_METHOD void sctx::point(int bounce)
{ 
    if(evt->record && bounce < evt->max_record) evt->record[evt->max_record*idx+bounce] = p ;   
    if(evt->rec    && bounce < evt->max_rec)    evt->add_rec( rec, idx, bounce, p );    // this copies into evt->rec array 
    if(evt->seq    && bounce < evt->max_seq)    seq.add_nibble( bounce, p.flag(), p.boundary() );  
    if(evt->aux    && bounce < evt->max_aux)    evt->aux[evt->max_aux*idx+bounce] = aux ;   
}


/**
sctx::trace : copy current prd into (idx,bounce) entry of evt->prd
---------------------------------------------------------------------

As *prd* is updated by *trace* rather than *propagate* it is handled separately to sctx:point.
The *prd* corresponds to the arrows (and trace) that gets between the points, eg:: 

   TO->BT->BT->SC->AB

**/

SCTX_METHOD void sctx::trace(int bounce)
{
    if(evt->prd) evt->prd[evt->max_prd*idx+bounce] = *prd ; 
}

/**
sctx::end : copy current seq into idx entry of evt->seq
-----------------------------------------------------------

Q: did I forget rec ? 
A: No. rec+record are added bounce-by-bounce into evt->rec/record in sctx::point 

   * seq is different because it is added nibble by nibble into the big integer


Q: why not copy p into evt->photon[idx] here ?
A: unsure, currently thats done in SEvt::finalPhoton

**/

SCTX_METHOD void sctx::end()
{
    if(evt->seq)  evt->seq[idx] = seq ; 
#ifdef DEBUG_TAG
    if(evt->tag)  evt->tag[idx]  = tagr.tag ;
    if(evt->flat) evt->flat[idx] = tagr.flat ;
#endif
}

#endif

