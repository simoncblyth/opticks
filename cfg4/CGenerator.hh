#pragma once

class OpticksHub ; 
class Opticks ; 
template <typename T> class OpticksCfg ; 
class CG4 ; 
class TorchStepNPY ; 
class CSource ; 
template <typename T> class NPY ; 

#include "CFG4_API_EXPORT.hh"

class CFG4_API CGenerator 
{
   public:
       CGenerator(OpticksHub* hub, CG4* g4);
       CSource* getSource();
       bool        isDynamic();
       unsigned    getNumG4Event();
       unsigned    getNumPhotonsPerG4Event();
       NPY<float>* getGensteps();
    private:
       void init();
       void setDynamic(bool dynamic);
       void setNumG4Event(unsigned num);
       void setNumPhotonsPerG4Event(unsigned num);
       void setGensteps(NPY<float>* gensteps);
       void setSource(CSource* source);
    private:
       CSource* makeTorchSource();
       CSource* makeG4GunSource();
   private:
       
       OpticksHub*           m_hub ;
       Opticks*              m_ok ;
       OpticksCfg<Opticks>*  m_cfg ;
       CG4*                  m_g4 ; 
       CSource*              m_source ; 
   private:
       unsigned      m_num_g4evt ; 
       unsigned      m_photons_per_g4evt ;           
       NPY<float>*   m_gensteps ; 
       bool          m_dynamic ; 

};


