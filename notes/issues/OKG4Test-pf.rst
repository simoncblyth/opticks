OKG4Test-pf
===============

::

   OKG4Test --target 62590 --generateoverride -1 --rngmax 3 --xanalytic --cvd 1 --rtx 1 --save

       ### takes lots of minutes, for Geant4 initialization ...

   OKG4Test --target 62590 --generateoverride -1 --rngmax 3 --xanalytic --cvd 1 --rtx 1 --load
   OKTest   --target 62590 --generateoverride -3 --rngmax 3 --xanalytic --cvd 1 --rtx 1 --load

       ## few seconds, just loads 

   OKG4Test --target 62590 --generateoverride -1 --rngmax 3 --xanalytic --cvd 1 --rtx 1 --load --vizg4




