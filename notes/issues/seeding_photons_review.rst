Seeding Photons Review
=========================


using the seeds in generate.cu
--------------------------------


1. launch_index -> photon_id -> photon_offset
2. seed_buffer[photon_id] -> genstep_id  

   * WITH_SEED_BUFFER is current standard, previously grabbed genstep_id from photon_buffer

3. genstep_id -> genstep_offset -> gencode (1st 4 bytes of the genstep)


Adding EMIT gencode
~~~~~~~~~~~~~~~~~~~~

* only need one genstep containing the new EMIT gencode
* seed buffer can thus be full of zeros 


Overheads of treating CPU photons with current machinery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using CPU photons do not need 

* genstep buffer
* seed buffer

Instead need input_photon buffer ? Suspect best to keep that 
separate from the output photons buffer.


Avoid overheads with separate "propagate" entrypoint ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Splitting off propagation from generate.cu is a major undertaking 
from current state of cu/generate.cu 

Perhaps can aim in that direction, but 
too much global state buffers etc.. to pass into methods.




Generation mechanics
---------------------------

cu/generate.cu::

    317 RT_PROGRAM void generate()
    318 {
    319     unsigned long long photon_id = launch_index.x ;
    320     unsigned int photon_offset = photon_id*PNUMQUAD ;
    321 
    322 #ifdef WITH_SEED_BUFFER
    323     unsigned int genstep_id = seed_buffer[photon_id] ;
    324 #else
    325     union quad phead ;
    326     phead.f = photon_buffer[photon_offset+0] ;   // crazy values for this in interop mode on Linux, photon_buffer being overwritten ?? 
    327     unsigned int genstep_id = phead.u.x ;        // input "seed" pointing from photon to genstep (see seedPhotonsFromGensteps)
    328 #endif
    329     unsigned int genstep_offset = genstep_id*GNUMQUAD ;
    330 
    331     union quad ghead ;
    332     ghead.f = genstep_buffer[genstep_offset+0];
    333     int gencode = ghead.i.x ;
    ...
    344     curandState rng = rng_states[photon_id];
    ...
    348     State s ;
    349     Photon p ;
    ...
    352     if(gencode == CERENKOV)   // 1st 4 bytes, is enumeration distinguishing cerenkov/scintillation/torch/...
    353     {
    354         CerenkovStep cs ;
    355         csload(cs, genstep_buffer, genstep_offset, genstep_id);
    356 #ifdef DEBUG
    357         if(dbg) csdebug(cs);
    358 #endif
    359         generate_cerenkov_photon(p, cs, rng );
    360         s.flag = CERENKOV ;
    361     }
    362     else if(gencode == SCINTILLATION)
    363     {
    364         ScintillationStep ss ;
    365         ssload(ss, genstep_buffer, genstep_offset, genstep_id);
    366 #ifdef DEBUG
    367         if(dbg) ssdebug(ss);
    368 #endif
    369         generate_scintillation_photon(p, ss, rng );
    370         s.flag = SCINTILLATION ;
    371     }
    372     else if(gencode == TORCH)
    373     {







