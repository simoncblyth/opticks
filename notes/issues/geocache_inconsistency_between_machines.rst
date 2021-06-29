geocache_inconsistency_between_machines
==========================================

geocache PE_PA inconsistency 
-------------------------------

* HMM : dont see any "PE_PA" in --dbgseqhis output 
* perhaps the epsilon geocache is out of sync with the precision one : which could mess up material names among other things 

  * YEP, using kcd shows that  
  * that means the "PE_PA" actually means "Vacuum", but the boundaries could be messed too 

* getting a full geometry digest to work from experience is too much effort, but 
  a simple geocache summary digest would be much less effort and would catch 
  most normal cases of forgetting to update all geocache after geometry changes 

* after writing the geocache could create a digest from the usual files eg in GItemList  
  and store that together with event metadata 

* then analysis can check against the digest for the geocache in use 

* need a digest that is quick and easy to obtain in bash/python/C++ 
  given a list of file paths to consume 

* want geocache created from a GDML file and the live one to match, GDML 0x postfix poses a problem for this 


Differing materials between kcd on different machines
---------------------------------------------------------

::

    O[blyth@localhost GItemList]$ cat GMaterialLib.txt
    LS
    Steel
    Tyvek
    Air
    Scintillator
    TiO2Coating
    Adhesive
    Aluminium
    Rock
    LatticedShellSteel
    Acrylic
    Vacuum
    Pyrex
    Water
    vetoWater
    Galactic
    O[blyth@localhost GItemList]$ 


    epsilon:GItemList blyth$ cat GMaterialLib.txt 
    LS
    Steel
    Tyvek
    Air
    Scintillator
    TiO2Coating
    Adhesive
    Aluminium
    Rock
    LatticedShellSteel
    Acrylic
    **PE_PA**
    Vacuum
    Pyrex
    Water
    vetoWater
    Galactic
    epsilon:GItemList blyth$ 




