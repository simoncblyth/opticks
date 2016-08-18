Event Layout Awkward
=====================

Persisted evts are too spread out and mixed up with others.
Need better locality.

New Layout Idea
----------------

::

      BoxInBox/torch/1/fdom.npy
      BoxInBox/torch/1/idom.npy
      BoxInBox/torch/1/no.npy
      BoxInBox/torch/1/rx.npy
      BoxInBox/torch/1/parameters.json
      
      BoxInBox/torch/1/20160813_111151/parameters.json 

Directory Levels det/src/tag/:

* det : BoxInBox,PmtInBox,dayabay,prism,reflect  (corresponds to a geometry)
* src : torch|cerenkov|scintillation  (photon source)
* tag : 1/2/..   (eg differernt or S and P pol)

* constituent arrays: ox.npy gs.npy ... parameters.json
* dated folders : for metadata saved from prior runs


All NPY evt paths coming via::

     BOpticksEvent::path(typ, tag, det)

Questions
-----------

* compute and interop co-located ?


Current Layout
------------------
::

    simon:evt blyth$ find . -name 1.npy

    ./BoxInBox/fdomtorch/1.npy
    ./BoxInBox/idomtorch/1.npy
    ./BoxInBox/notorch/1.npy
    ./BoxInBox/oxtorch/1.npy
    ./BoxInBox/phtorch/1.npy
    ./BoxInBox/pstorch/1.npy
    ./BoxInBox/rstorch/1.npy
    ./BoxInBox/rxtorch/1.npy

    ./dayabay/fdomcerenkov/1.npy
    ./dayabay/fdomtorch/1.npy
    ./dayabay/gscerenkov/1.npy
    ./dayabay/gstorch/1.npy
    ./dayabay/idomcerenkov/1.npy
    ./dayabay/idomtorch/1.npy
    ./dayabay/nocerenkov/1.npy
    ./dayabay/notorch/1.npy

    simon:evt blyth$ find . -name parameters.json
    ./BoxInBox/mdtorch/1/20160816_131052/parameters.json
    ./BoxInBox/mdtorch/1/parameters.json
    ./dayabay/mdcerenkov/1/20160816_184156/parameters.json
    ./dayabay/mdcerenkov/1/parameters.json
    ./dayabay/mdtorch/1/20160813_111151/parameters.json
    ./dayabay/mdtorch/1/20160813_150950/parameters.json
    ./dayabay/mdtorch/1/20160813_174811/parameters.json
    ./dayabay/mdtorch/1/20160814_122440/parameters.json
    ./dayabay/mdtorch/1/20160816_114131/parameters.json
    ./dayabay/mdtorch/1/20160816_162450/parameters.json



