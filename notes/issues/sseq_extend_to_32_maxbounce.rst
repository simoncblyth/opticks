sseq_extend_to_32_maxbounce
==============================

Unexpected NA in history::

    280 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   729 seq.desc_seqhis     8acbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SA
     281 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   728 seq.desc_seqhis              8ccd nib  4 TO BT BT SA
     282 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   727 seq.desc_seqhis      8ccccaaccccd nib 12 TO BT BT BT BT SR SR BT BT BT BT SA
     283 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   726 seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
     284 U4Recorder::PostUserTrackingAction_Optical@330:  label.id   725 seq.desc_seqhis              8ccd nib  4 TO BT BT SA


    U4Recorder::BeginOfEventAction@93: 
    U4Recorder::PostUserTrackingAction_Optical@330:  label.id   726 seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    U4Recorder::EndOfEventAction@94: 




Dumping the seqhis point by point with the rerun shows its a wraparound effect::

    epsilon:tests blyth$ grep SEvt::pointPhoton *.log
    SEvt::pointPhoton@1269:  label.id   726 bounce  0 ctx.p.flag TO seq.desc_seqhis                 0 nib  0  
    SEvt::pointPhoton@1269:  label.id   726 bounce  1 ctx.p.flag BT seq.desc_seqhis                 d nib  1 TO
    SEvt::pointPhoton@1269:  label.id   726 bounce  2 ctx.p.flag BT seq.desc_seqhis                cd nib  2 TO BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  3 ctx.p.flag BT seq.desc_seqhis               ccd nib  3 TO BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  4 ctx.p.flag BT seq.desc_seqhis              cccd nib  4 TO BT BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  5 ctx.p.flag SR seq.desc_seqhis             ccccd nib  5 TO BT BT BT BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  6 ctx.p.flag SR seq.desc_seqhis            accccd nib  6 TO BT BT BT BT SR
    SEvt::pointPhoton@1269:  label.id   726 bounce  7 ctx.p.flag BT seq.desc_seqhis           aaccccd nib  7 TO BT BT BT BT SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce  8 ctx.p.flag BR seq.desc_seqhis          caaccccd nib  8 TO BT BT BT BT SR SR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce  9 ctx.p.flag BR seq.desc_seqhis         bcaaccccd nib  9 TO BT BT BT BT SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 10 ctx.p.flag BT seq.desc_seqhis        bbcaaccccd nib 10 TO BT BT BT BT SR SR BT BR BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 11 ctx.p.flag SR seq.desc_seqhis       cbbcaaccccd nib 11 TO BT BT BT BT SR SR BT BR BR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce 12 ctx.p.flag SR seq.desc_seqhis      acbbcaaccccd nib 12 TO BT BT BT BT SR SR BT BR BR BT SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 13 ctx.p.flag SR seq.desc_seqhis     aacbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 14 ctx.p.flag BT seq.desc_seqhis    aaacbbcaaccccd nib 14 TO BT BT BT BT SR SR BT BR BR BT SR SR SR
    SEvt::pointPhoton@1269:  label.id   726 bounce 15 ctx.p.flag BR seq.desc_seqhis   caaacbbcaaccccd nib 15 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT
    SEvt::pointPhoton@1269:  label.id   726 bounce 16 ctx.p.flag BT seq.desc_seqhis  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 17 ctx.p.flag SR seq.desc_seqhis  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 18 ctx.p.flag BT seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1269:  label.id   726 bounce 19 ctx.p.flag SA seq.desc_seqhis  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    epsilon:tests blyth$ 


Also looks like getting repeated flag at FastSim/SlowSim transitions ? 
NO its not, its just the BT across the fake boundary leading to more. 

Reproduce the misbehavior in sseq_test::

    epsilon:tests blyth$ name=sseq_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name
                   TORCH :                 d nib  1 TO
       BOUNDARY_TRANSMIT :                cd nib  2 TO BT
       BOUNDARY_TRANSMIT :               ccd nib  3 TO BT BT
       BOUNDARY_TRANSMIT :              cccd nib  4 TO BT BT BT
       BOUNDARY_TRANSMIT :             ccccd nib  5 TO BT BT BT BT
        SURFACE_SREFLECT :            accccd nib  6 TO BT BT BT BT SR
        SURFACE_SREFLECT :           aaccccd nib  7 TO BT BT BT BT SR SR
       BOUNDARY_TRANSMIT :          caaccccd nib  8 TO BT BT BT BT SR SR BT
        BOUNDARY_REFLECT :         bcaaccccd nib  9 TO BT BT BT BT SR SR BT BR
        BOUNDARY_REFLECT :        bbcaaccccd nib 10 TO BT BT BT BT SR SR BT BR BR
       BOUNDARY_TRANSMIT :       cbbcaaccccd nib 11 TO BT BT BT BT SR SR BT BR BR BT
        SURFACE_SREFLECT :      acbbcaaccccd nib 12 TO BT BT BT BT SR SR BT BR BR BT SR
        SURFACE_SREFLECT :     aacbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SR
        SURFACE_SREFLECT :    aaacbbcaaccccd nib 14 TO BT BT BT BT SR SR BT BR BR BT SR SR SR
       BOUNDARY_TRANSMIT :   caaacbbcaaccccd nib 15 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT
        BOUNDARY_REFLECT :  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
       BOUNDARY_TRANSMIT :  bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
        SURFACE_SREFLECT :  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
       BOUNDARY_TRANSMIT :  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
          SURFACE_ABSORB :  bcaaacbbcaaccced nib 16 TO NA BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    epsilon:tests blyth$ 



* The wraparound is from shifting beyond the width of the type. 
* And getting NA arises from OR-ing of different flags together. 

  
Need to widen sseq storage adopting the techniques used in stag.h 
to write the nibbles. 
   
Writing for both GPU and CPU is done via::

    076 SCTX_METHOD void sctx::point(int bounce)
     77 {
     78     if(evt->record && bounce < evt->max_record) evt->record[evt->max_record*idx+bounce] = p ;
     79     if(evt->rec    && bounce < evt->max_rec)    evt->add_rec( rec, idx, bounce, p );    // this copies into evt->rec array 
     80     if(evt->seq    && bounce < evt->max_seq)    seq.add_nibble( bounce, p.flag(), p.boundary() );
     81 }


    114 SSEQ_METHOD void sseq::add_nibble(unsigned slot, unsigned flag, unsigned boundary )
    115 {
    116     seqhis |=  (( FFS(flag) & 0xfull ) << 4*slot );
    117     seqbnd |=  (( boundary  & 0xfull ) << 4*slot );
    118     // 0xfull is needed to avoid all bits above 32 getting set
    119     // NB: nibble restriction of each "slot" means there is absolute no need for FFSLL
    120 }


Reworked sseq.h to hold NSEQ elements following stag.h example.

This fixes overwriting, increasing sseq recording to not overwrite up to maxbounce 32::

    epsilon:tests blyth$ name=sseq_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name
    test_desc_seqhis_1
                   TORCH :                 0                d nib  1 TO
       BOUNDARY_TRANSMIT :                 0               cd nib  2 TO BT
       BOUNDARY_TRANSMIT :                 0              ccd nib  3 TO BT BT
       BOUNDARY_TRANSMIT :                 0             cccd nib  4 TO BT BT BT
       BOUNDARY_TRANSMIT :                 0            ccccd nib  5 TO BT BT BT BT
        SURFACE_SREFLECT :                 0           accccd nib  6 TO BT BT BT BT SR
        SURFACE_SREFLECT :                 0          aaccccd nib  7 TO BT BT BT BT SR SR
       BOUNDARY_TRANSMIT :                 0         caaccccd nib  8 TO BT BT BT BT SR SR BT
        BOUNDARY_REFLECT :                 0        bcaaccccd nib  9 TO BT BT BT BT SR SR BT BR
        BOUNDARY_REFLECT :                 0       bbcaaccccd nib 10 TO BT BT BT BT SR SR BT BR BR
       BOUNDARY_TRANSMIT :                 0      cbbcaaccccd nib 11 TO BT BT BT BT SR SR BT BR BR BT
        SURFACE_SREFLECT :                 0     acbbcaaccccd nib 12 TO BT BT BT BT SR SR BT BR BR BT SR
        SURFACE_SREFLECT :                 0    aacbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SR
        SURFACE_SREFLECT :                 0   aaacbbcaaccccd nib 14 TO BT BT BT BT SR SR BT BR BR BT SR SR SR
       BOUNDARY_TRANSMIT :                 0  caaacbbcaaccccd nib 15 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT
        BOUNDARY_REFLECT :                 0 bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
       BOUNDARY_TRANSMIT :                 c bcaaacbbcaaccccd nib 17 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT
        SURFACE_SREFLECT :                ac bcaaacbbcaaccccd nib 18 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR
       BOUNDARY_TRANSMIT :               cac bcaaacbbcaaccccd nib 19 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR BT
          SURFACE_ABSORB :              8cac bcaaacbbcaaccccd nib 20 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR BT SA
    epsilon:tests blyth$ 



Back to U4PMTFastSimTest.sh::

    epsilon:tests blyth$ grep SEvt::pointPhoton *.log
    SEvt::pointPhoton@1274: (  726, 0) TO                0                d nib  1 TO
    SEvt::pointPhoton@1274: (  726, 1) BT                0               cd nib  2 TO BT
    SEvt::pointPhoton@1274: (  726, 2) BT                0              ccd nib  3 TO BT BT
    SEvt::pointPhoton@1274: (  726, 3) BT                0             cccd nib  4 TO BT BT BT
    SEvt::pointPhoton@1274: (  726, 4) BT                0            ccccd nib  5 TO BT BT BT BT
    SEvt::pointPhoton@1274: (  726, 5) SR                0           accccd nib  6 TO BT BT BT BT SR
    SEvt::pointPhoton@1274: (  726, 6) SR                0          aaccccd nib  7 TO BT BT BT BT SR SR
    SEvt::pointPhoton@1274: (  726, 7) BT                0         caaccccd nib  8 TO BT BT BT BT SR SR BT
    SEvt::pointPhoton@1274: (  726, 8) BR                0        bcaaccccd nib  9 TO BT BT BT BT SR SR BT BR
    SEvt::pointPhoton@1274: (  726, 9) BR                0       bbcaaccccd nib 10 TO BT BT BT BT SR SR BT BR BR
    SEvt::pointPhoton@1274: (  726,10) BT                0      cbbcaaccccd nib 11 TO BT BT BT BT SR SR BT BR BR BT
    SEvt::pointPhoton@1274: (  726,11) SR                0     acbbcaaccccd nib 12 TO BT BT BT BT SR SR BT BR BR BT SR
    SEvt::pointPhoton@1274: (  726,12) SR                0    aacbbcaaccccd nib 13 TO BT BT BT BT SR SR BT BR BR BT SR SR
    SEvt::pointPhoton@1274: (  726,13) SR                0   aaacbbcaaccccd nib 14 TO BT BT BT BT SR SR BT BR BR BT SR SR SR
    SEvt::pointPhoton@1274: (  726,14) BT                0  caaacbbcaaccccd nib 15 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT
    SEvt::pointPhoton@1274: (  726,15) BR                0 bcaaacbbcaaccccd nib 16 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR
    SEvt::pointPhoton@1274: (  726,16) BT                c bcaaacbbcaaccccd nib 17 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT
    SEvt::pointPhoton@1274: (  726,17) SR               ac bcaaacbbcaaccccd nib 18 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR
    SEvt::pointPhoton@1274: (  726,18) BT              cac bcaaacbbcaaccccd nib 19 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR BT
    SEvt::pointPhoton@1274: (  726,19) SA             8cac bcaaacbbcaaccccd nib 20 TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR BT SR BT SA
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 


::

    epsilon:tests blyth$ ./U4PMTFastSimTest.sh ana

    In [7]: t.seq[726,0]
    Out[7]: array([13594868347730447565,                36012], dtype=uint64)

    In [8]: seqhis_(t.seq[726,0])
    Out[8]: ['TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR', 'BT SR BT SA']




