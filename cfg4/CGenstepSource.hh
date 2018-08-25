#pragma once

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"
#include <vector>

class G4Event ; 
class G4PrimaryVertex ;
class G4PrimaryParticle ;
class G4VParticleChange ; 

class Opticks ; 
class OpticksGenstep ; 
template <typename T> class NPY ; 

/**
CGenstepSource
==================

Instanciated by CGenerator::initSource like all other CSource subclasses.


Note that Opticks has been running gensteps from the beginning, the
difference with this class is that it allows Geant4 for do that too, by 
moving a copy of the photon generation loop from Cerenkov or Scintillation 
PostStepDoIt into the primary stage.


Recall that gensteps can yield many millions of photons : 
how best to arrange those into G4Event/G4PrimaryVertex/G4PrimaryParticle ?

Putting them all into a single event will probably fail, putting 
a single photon into each event would probably be horribly slow.

As gensteps are the input its natural to split along genstep lines, 
putting a configurable number of gensteps into each event, except
the last the gets the remainder.
  
With photon sources, I restricted to 10000(?) photons per event 
so perhaps 100 gensteps per event. 

:: 

    epsilon:tests blyth$ OpticksGenstepTest 
    PLOG::PLOG  instance 0x7fe7bf403490 this 0x7fe7bf403490 logpath OpticksGenstepTest.log
    2018-08-23 18:32:13.307 INFO  [116442] [OpticksGenstep::dump@114] OpticksGenstep::dump slice NSlice      0 :  7245 :     1  modulo 1000 margin 10
     desc OpticksGenstep 7245,6,4 num_gensteps 7245 num_photons 1142140 avg_photons_per_genstep 157.645
     i       0 hdr      -1       1      12      80 post ( -16536.295-802084.812 -7066.000       0.844) dpsl (     -2.057     3.180     0.000       3.788)
     i       1 hdr      -4       1      12      30 post ( -16542.469-802075.250 -7066.000       0.882) dpsl (     -0.563     0.870    -0.000       1.037)
     i       2 hdr      -7       1      12     106 post ( -16547.146-802068.062 -7066.000       0.911) dpsl (     -2.058     3.180    -0.001       3.788)
     i       3 hdr     -10       1      12      29 post ( -16553.318-802058.500 -7066.002       0.949) dpsl (     -0.674     1.042    -0.000       1.240)
     i       4 hdr     -13       1      12     104 post ( -16556.533-802053.500 -7066.002       0.969) dpsl (     -2.057     3.181    -0.000       3.788)
     i       5 hdr     -16       1      12      63 post ( -16562.703-802044.000 -7066.003       1.007) dpsl (     -1.332     2.061    -0.000       2.454)
     i       6 hdr     -19       1      12      93 post ( -16566.762-802037.688 -7066.003       1.032) dpsl (     -2.057     3.181    -0.000       3.788)
    ...


**/

struct STranche ; 
class C4PhotonCollector ; 

class CFG4_API CGenstepSource: public CSource
{
    public:
        CGenstepSource(Opticks* ok,  NPY<float>* gs );
        virtual ~CGenstepSource();
        NPY<float>* getSourcePhotons() const ;
    private:
        void init();
    public:
        unsigned getNumGenstepsPerG4Event() const ; 
        unsigned getNumG4Event() const ;
        G4VParticleChange* generatePhotonsFromOneGenstep();
    public:
        void GeneratePrimaryVertex(G4Event *event);    // G4VPrimaryGenerator interface
    private:
        void addPrimaryVertices(G4Event *event, const G4VParticleChange* pc) const ;
    private:
        OpticksGenstep*       m_gs ;
        unsigned              m_num_genstep ; 
        unsigned              m_num_genstep_per_g4event ; 
        STranche*             m_tranche ; 
        unsigned              m_idx ; 
        unsigned              m_generate_count ;   
        C4PhotonCollector*    m_photon_collector ; 
        // event count should be in base class : but base needs a rewrite so leave it here for now

};


