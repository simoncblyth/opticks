#pragma once

class OpticksHub ; 
class Opticks ; 
class OpticksEvent ; 
template <typename T> class OpticksCfg ; 
class CG4 ; 
class TorchStepNPY ; 
class CSource ; 
template <typename T> class NPY ; 

#include "CFG4_API_EXPORT.hh"

/**

CGenerator
===========

Canonical m_generator instance is CG4 resident instanciated within it.    

Instanciation hooks up the configured sources (of photons or primaries) 
into CSource(G4VPrimaryGenerator) ready to be given to Geant4
to make primary vertices with. 

=====================  ==========
source                  dynamic 
=====================  ==========
inputPhotonSource         N 
TorchSource               N
G4GunSource               Y
inputPrimarySource        Y
=====================  ==========


**/

class CFG4_API CGenerator 
{
   public:
       CGenerator(OpticksGen* gen, CG4* g4);
   public:
       void        configureEvent(OpticksEvent* evt);
   public:
       unsigned    getSourceCode() const ;
       CSource*    getSource() const ;
       bool        isDynamic() const ;
       unsigned    getNumG4Event() const ;
       unsigned    getNumPhotonsPerG4Event() const ;
       NPY<float>* getGensteps() const ;
       bool        hasGensteps() const ;
   private:
       void init();
       CSource* initSource(unsigned code);
       CSource* initInputPhotonSource();
       CSource* initInputPrimarySource();
       CSource* initTorchSource();
       CSource* initG4GunSource();
    private:
       void setDynamic(bool dynamic);
       void setNumG4Event(unsigned num);
       void setNumPhotonsPerG4Event(unsigned num);
       void setGensteps(NPY<float>* gensteps);
   private:
       OpticksGen*           m_gen ;
       Opticks*              m_ok ;
       OpticksCfg<Opticks>*  m_cfg ;
       CG4*                  m_g4 ; 
       unsigned              m_source_code ; 
       CSource*              m_source ; 
   private:
       unsigned      m_num_g4evt ; 
       unsigned      m_photons_per_g4evt ;           
       NPY<float>*   m_gensteps ; 
       bool          m_dynamic ; 

};


