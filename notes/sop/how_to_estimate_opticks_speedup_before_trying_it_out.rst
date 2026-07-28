how_to_estimate_opticks_speedup_before_trying_it_out
=====================================================


Using Amdahls Law to estimate maximal overall speedup
---------------------------------------------------------

To estimate potential overall speedup, compare your Geant4 simulation time
with and without optical photons enabled to give the parallelizable fraction p.
As the optical speedup n is expected to be large you can then estimate your
maximal limit on the overall speedup as,  1/(1-p)
Amdahl overall speedup S as a function of parallel fraction p and parallel speedup n::

                     1
        S  =   -----------------
               ( 1 - p ) + p/n

The maximal limits for the type of events you are intersested in can
inform you of whether pursuing Opticks or other GPU approaches will be worthwhile.


What optical speedup *n* is needed to achieve a fraction *k* of maximal overall speedup
-----------------------------------------------------------------------------------------

Expressing S as a fraction k of this maximum::

      k               1
   --------  =   ------------------
    1 - p        ( 1 - p ) + p/n


Inverting and rearranging that::

   (1-p) [ 1/k - 1 ]  = p/n


So for parallel fraction *p* the parallel speedup *n* required
to reach a fraction *k* of the overall speed limit::

                     p             k
           n =   ---------   .  --------
                 ( 1 - p )      ( 1 - k )


The k/(1-k) factor expresses dimininishing returns, as k->1 you need infinite parallel speedup *n*.

But for  p = 0.99 k = 0.5  you get to half the maximum overall speedup with only n = 99::

             0.99        0.5
      n =   -------  .   ---- =    99
             0.01        0.5


And you get to 90%(95%) with n = 891(1881)::

              0.99      0.9
      n  =   -----  .  -----   =   891  [easily possible with older NVIDIA GPUs]
              0.01      0.1

              0.99      0.95
      n  =   -----  .  -----   =   99*19 = 1881    [perfectly possible with recent NVIDIA RTX GPUs]
              0.01      0.05







