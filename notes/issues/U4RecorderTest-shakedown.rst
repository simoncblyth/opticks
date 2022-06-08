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


With Gun : Not terminated at AB ? Probably reemision rejoin AB scrub not working yet ?
-----------------------------------------------------------------------------------------

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









