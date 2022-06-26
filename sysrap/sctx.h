#pragma once
/**
sctx.h
==========

This is aiming to avoid non-production instrumentation costing anything 
in production running.  

**/

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
    void point(int bounce); 
#endif
    void end(int bounce); 
}; 


#ifndef PRODUCTION
void sctx::point(int bounce)
{ 
    if(evt->record) evt->record[evt->max_record*idx+bounce] = p ;   
    if(evt->rec) evt->add_rec( rec, idx, bounce, p );  
    if(evt->seq) seq.add_nibble( bounce, p.flag(), p.boundary() );  
    if(evt->prd) evt->prd[evt->max_prd*idx+bounce] = *prd ; 
}
#endif

void sctx::end(int bounce)
{
#ifndef PRODUCTION
    if( evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;
    if( evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p );
    if( evt->seq    && bounce < evt->max_seq    ) seq.add_nibble(bounce, p.flag(), p.boundary() );
    // no prd in the tail : as that is unchanged by propagation
    if(evt->seq) evt->seq[idx] = seq ;
#ifdef DEBUG_TAG
    if(evt->tag)  evt->tag[idx]  = tagr.tag ;
    if(evt->flat) evt->flat[idx] = tagr.flat ;
#endif
#endif
    evt->photon[idx] = p ;
}


