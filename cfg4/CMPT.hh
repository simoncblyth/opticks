#pragma once

#include <vector>
#include <map>
#include <string>

class CVec ; 
class G4PhysicsOrderedFreeVector ;
class G4MaterialPropertiesTable ;
template <typename T> class GProperty ; 
template <typename T> class NPY ; 

/**
CMPT
=====




**/

#include "CFG4_API_EXPORT.hh"
class CFG4_API CMPT {
   public:
       static CMPT* MakeDummy(); 
       static std::string Digest(G4MaterialPropertiesTable* mpt); 
       static void AddDummyProperty(G4MaterialPropertiesTable* mpt, const char* lkey, unsigned nval) ;
       static bool HasProperty(const G4MaterialPropertiesTable* mpt, const char* pname); 
   public:
       void addDummyProperty(const char* lkey, unsigned nval) ; 
       //void dump(const char* msg); 
   public:
       CMPT(G4MaterialPropertiesTable* mpt, const char* name=NULL);
       std::string digest() const ;
       bool hasProperty(const char* pname) const ; 
   public:
       // adds domain and value of prop, assumes prop domain is in ascending nm 
       // (as typically used for optical properties) and thus the order is  
       // reversed to yield energy ascending order and domain is converted to eV 
       void addProperty(const char* lkey,  GProperty<float>* prop, bool spline);
   public:
       void dumpRaw(const char* lkey);
   public:
       static void Dump(G4MaterialPropertiesTable* mpt, const char* msg="CMPT::Dump"); 
       void dump(const char* msg="CMPT::dump") const; 
       void dumpProperty(const char* lkey);
       void sample(NPY<float>* a, unsigned offset, const char* _keys, float low, float step, unsigned nstep );
       void sampleSurf(NPY<float>* a, unsigned offset, float low, float step, unsigned nstep, bool specular );

       GProperty<double>* makeProperty(const char* key, float low, float step, unsigned nstep);
       G4PhysicsOrderedFreeVector* getVec(const char* lkey) const ;
       CVec* getCVec(const char* lkey) const ;

       unsigned splitKeys(std::vector<std::string>& keys, const char* _keys);
       unsigned getVecLength(const char* _keys);
       NPY<float>* makeArray(const char* _keys, bool reverse=true);


       std::string description(const char* msg);
       std::vector<std::string> getPropertyKeys();
       std::vector<std::string> getPropertyDesc() const ;
       std::vector<std::string> getConstPropertyKeys();
       std::vector<double> getConstPropertyValues();
   private:
       G4MaterialPropertiesTable* m_mpt ; 
       const char* m_name ; 
};



