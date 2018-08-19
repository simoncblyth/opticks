rng_aligned_cerenkov_generation
=================================

* Recall gensteps can be regarded as a copy of the "stack" 
  from just before the photon generation loop, with values that 
  include the number of photons to generate, positions etc..

* By input gensteps : I mean Opticks runs with input gensteps, Geant4 
  runs as normal from primaries : so it is consuming RNGs before get to the 
  photon generation, to for example decide how many photons.  

* contrast with input photons where all RNG consumption on CPU/GPU sides 
  is matched as the bifurcation between the simulations was immediate.

* With input gensteps the bifurcation is in the middle of G4Cerenkov
  at the generation loop : so need to switch the RNG stream from 
  a normal one to an aligned one at each spin of that loop, 
  with the photon index to pick the sub-stream 

* Can use photon index -1 to mean ordinary non-aligned RNG stream 


Attempt to get iteration started without alignment : using direct key geocache + primary persisting
----------------------------------------------------------------------------------------------------

::

    ckm--(){ ckm-cd ; ./go.sh ; } 
        ## Cerenkov minimal sets up geometry primaries then persists to key geocache+gdml+primaries 

    ckm-okg4()
    {
        OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey --embedded --save
    }
       ## okg4 picks up geocache+GDML+primaries from the key geocache and proceeds to bi-simulate in compute mode

    ckm-okg4-load()
    {
        OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --load --envkey --embedded
    }
       ## subsequent load event and geometry for visualization 



Immediate issues::

    epsilon:torch blyth$ np.py 
    /private/tmp/blyth/opticks/evt/g4live/torch
           ./Opticks.npy : (33, 1, 4) 

         ./-1/report.txt : 31 
           ./-1/idom.npy : (1, 1, 4) 
           ./-1/fdom.npy : (3, 1, 4) 
             ./-1/gs.npy : (5, 6, 4) 
             ./-1/no.npy : (152, 4, 4) 

             ./-1/rx.npy : (76, 10, 2, 4) 
             ./-1/ox.npy : (76, 4, 4) 
             ./-1/ph.npy : (76, 1, 2) 

           ## NO HITS FROM G4 : SD-LV ASSOCIATION DIDNT SURVIVE THE CACHE/GDML ??

           ## phosel and recsel fail to get created <-??-> CPU side indexing aborted 


    ./-1/20180819_214856/report.txt : 31 

          ./1/report.txt : 38 
            ./1/idom.npy : (1, 1, 4) 
            ./1/fdom.npy : (3, 1, 4) 
              ./1/gs.npy : (5, 6, 4) 
              ./1/no.npy : (152, 4, 4) 

              ./1/rx.npy : (76, 10, 2, 4) 
              ./1/ox.npy : (76, 4, 4) 
              ./1/ph.npy : (76, 1, 2) 

              ./1/ps.npy : (76, 1, 4) 
              ./1/rs.npy : (76, 10, 1, 4)     
              ./1/ht.npy : (9, 4, 4) 

    ./1/20180819_214856/report.txt : 38 
    epsilon:torch blyth$ 


Lots of bad flags : skipped some asserts in CRecorder/CPhoton




