#pragma once

#include <vector>
#include <map>
#include <string>

class G4PhysicsOrderedFreeVector ;
class G4MaterialPropertiesTable ;
template <typename T> class GProperty ; 
template <typename T> class NPY ; 

#include "CFG4_API_EXPORT.hh"
class CFG4_API CMPT {
   public:
       CMPT(G4MaterialPropertiesTable* mpt, const char* name=NULL);
       void addProperty(const char* lkey,  GProperty<float>* prop, bool spline);
   public:
       void dump(const char* msg="CMPT::dump"); 
       void dumpProperty(const char* lkey);
       void sample(NPY<float>* a, unsigned offset, const char* _keys, float low, float step, unsigned nstep );
       void sampleSurf(NPY<float>* a, unsigned offset, float low, float step, unsigned nstep, bool specular );

       GProperty<double>* makeProperty(const char* key, float low, float step, unsigned nstep);
       G4PhysicsOrderedFreeVector* getVec(const char* key);


       std::string description(const char* msg);
       std::vector<std::string> getPropertyKeys();
       std::vector<std::string> getPropertyDesc();
       std::vector<std::string> getConstPropertyKeys();
       std::vector<double> getConstPropertyValues();
   private:
       G4MaterialPropertiesTable* m_mpt ; 
       const char* m_name ; 
};



