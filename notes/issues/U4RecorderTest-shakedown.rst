U4RecorderTest-shakedown
===========================

With Gun : First 100 label id are zero ? FIXED 
------------------------------------------------

::

    In [25]: np.all( id_[100:] == np.arange(100,388, dtype=np.int32)  )
    Out[25]: True

    In [26]: np.all( id_[:100] == 0 )
    Out[26]: True

FIXED by commenting the SEvt::AddTorchGenstep when gun running::

    133 int main(int argc, char** argv)
    134 {    
    135     OPTICKS_LOG(argc, argv);
    136 
    137     unsigned max_bounce = 9 ;
    138     SEventConfig::SetMaxBounce(max_bounce);
    139     SEventConfig::SetMaxRecord(max_bounce+1);
    140     SEventConfig::SetMaxRec(max_bounce+1);
    141     SEventConfig::SetMaxSeq(max_bounce+1);
    142 
    143     SEvt evt ; 
    144     //SEvt::AddTorchGenstep();


With Gun : FIXED : Unexpected seq labels 
-----------------------------------------

* should be starting with SI or CK 

::

   0 : MI SD SD SD MI MI 
   1 : MI SD SD SD MI MI 
   2 : MI SD SD MI MI MI 
   3 : MI SD SD MI MI MI 
   4 : MI SC SD MI MI MI 
   5 : SI SC SD MI MI MI 
   6 : SI SC SD MI MI MI 
   7 : SI AB AB MI 
   8 : SI AB AB MI 


After zeroing seq and rec at SEvt::startPhoton the seq looks more reasonable::

   0 : CK AB AB 
   1 : CK AB SC AB MI 
   2 : CK AB 
   3 : CK MI 
   4 : CK AB 
   5 : SI AB 
   6 : SI SC MI MI MI MI 
   7 : SI AB 
   8 : SI AB AB MI 
   9 : SI MI 


With Gun : Not terminated at AB ? Probably reemision rejoin AB scrub not working yet ? YEP: FIXED
----------------------------------------------------------------------------------------------------

* actually did i implement that at all ? only did the flagmask not the seqhis ?

seqhis::

   0 : CK AB AB 
   1 : CK AB SC AB MI 
   2 : CK AB 
   3 : CK MI 
   4 : CK AB 
   5 : SI AB 
   6 : SI SC MI MI MI MI 
   7 : SI AB 
   8 : SI AB AB MI 
   9 : SI MI 

Implement GIDX control for debug running with single genstep.::

    bflagdesc_(r[0,j])
     idx(     0) prd(  0    0     0 0 ii:    0)  CK               CK  
     idx(     0) prd(  0    0     0 0 ii:    0)  AB            AB|CK  
     idx(     0) prd(  0    0     0 0 ii:    0)  AB         RE|AB|CK  


* FIXED : clear discrepancy between the flag+seqhis and the flagmask 

The current_photon flag gets seq.add_nibble by SEvt::pointPhoton::

    342 void SEvt::pointPhoton(const spho& label)
    343 {   
    344     assert( label.isSameLineage(current_pho) );
    345     unsigned idx = label.id ;
    346     int& bounce = slot[idx] ;
    347     
    348     const sphoton& p = current_photon ;
    349     srec& rec = current_rec ;
    350     sseq& seq = current_seq ;
    351     
    352     if( evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;
    353     if( evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p );  
    354     if( evt->seq    && bounce < evt->max_seq    ) seq.add_nibble(bounce, p.flag(), p.boundary() );
    355     
    356     bounce += 1 ;
    357 }

Fixed reemission bookkeeping by history rewrite.

SEvt::rjoinPhoton::


    331     if( evt->photon )
    332     {
    333        // HMM: could directly change photon[idx] via ref ? 
    334        // But are here taking a copy to current_photon
    335        // and relying on copyback at SEvt::endPhoton
    336 
    337         current_photon = photon[idx] ;
    338         assert( current_photon.flag() == BULK_ABSORB );
    339         assert( current_photon.flagmask & BULK_ABSORB  );   // all continuePhoton should have BULK_ABSORB in flagmask
    340 
    341         current_photon.flagmask &= ~BULK_ABSORB  ; // scrub BULK_ABSORB from flagmask
    342         current_photon.set_flag(BULK_REEMIT) ;     // gets OR-ed into flagmask 
    343     }
    344 
    345     if( evt->seq )
    346     {
    347         current_seq = seq[idx] ;
    348         unsigned seq_flag = current_seq.get_flag(prior);
    349         assert( seq_flag == BULK_ABSORB );
    350         current_seq.set_flag(prior, BULK_REEMIT);
    351     }
    352 
    353     if( evt->record )
    354     {
    355         sphoton& rjoin_record = evt->record[evt->max_record*idx+prior]  ;
    356         unsigned rjoin_flag = rjoin_record.flag() ;
    357 
    358         LOG(info) << " rjoin.flag "  << OpticksPhoton::Flag(rjoin_flag)  ;
    359         assert( rjoin_flag == BULK_ABSORB );
    360         assert( rjoin_record.flagmask & BULK_ABSORB );
    361 
    362         rjoin_record.flagmask &= ~BULK_ABSORB ; // scrub BULK_ABSORB from flagmask  
    363         rjoin_record.set_flag(BULK_REEMIT) ;
    364     }


GIDX selection beyond the first is asserting : FIXED 
--------------------------------------------------------

::

    2022-06-09 16:52:41.855 INFO  [19428647] [U4Recorder::BeginOfRunAction@38] 
    2022-06-09 16:52:41.855 INFO  [19428647] [U4Recorder::BeginOfEventAction@40] 
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   0 pho     5 off      0 typ G4Cerenkov_modified gidx 0 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   0 pho     1 off      0 typ DsG4Scintillation_r4695 gidx 1 enabled 1
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::setNumPhoton@210]  numphoton 1
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 2 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 3 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 4 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::beginPhoton@269] 
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::beginPhoton@270] spho ( gs ix id gn   1   0    1 0 ) 
    2022-06-09 16:52:41.856 ERROR [19428647] [SEvt::beginPhoton@275]  not in_range  idx 1 pho.size  1 label spho ( gs ix id gn   1   0    1 0 ) 
    Assertion failed: (in_range), function beginPhoton, file /Users/blyth/opticks/sysrap/SEvt.cc, line 281.
    ./U4RecorderTest.sh: line 43: 73818 Abort trap: 6           U4RecorderTest
    === ./U4RecorderTest.sh : logdir /tmp/blyth/opticks/U4RecorderTest
    epsilon:tests blyth$ 



The sgs genstep labelling is using an offset that does not account for enabled gensteps presumably::

     56 inline spho sgs::MakePho(unsigned idx, const spho& ancestor)
     57 {
     58     return ancestor.isDefined() ? ancestor.make_reemit() : spho::MakePho(index, idx, offset + idx, 0) ;
     59 }


FIXED this by simplifying genstep disabling to simply set the numphotons of disabled gensteps to zero, 
without any change to the collection machinery.  As genstep disabling is purely for debugging this is acceptable. 




