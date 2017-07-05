#pragma once


// TODO: get rid of this, move to nbbox 
// not so easy to jump to nbbox or adopt glm::vec3 here 
// due to stuffing into buffers requirements of GMesh 

#include "GVector.hh"
#include "GGEO_API_EXPORT.hh"

struct nbbox ; 

struct GGEO_API gbbox 
{
   static float MaxDiff( const gbbox& a, const gbbox& b);

   gbbox() : min(gfloat3(0.f)), max(gfloat3(0.f)) {} ;
   gbbox(float s) :  min(gfloat3(-s)), max(gfloat3(s)) {} ; 
   gbbox(const gfloat3& _min, const gfloat3& _max) :  min(_min), max(_max) {} ; 

   gbbox(const gbbox& other ) : min(other.min), max(other.max) {} ;
   gbbox(const nbbox& other );

   gfloat3 dimensions();
   gfloat3 center();
   void enlarge(float factor);  //  multiple of extent
   void include(const gbbox& other);
   gbbox& operator *= (const GMatrixF& m) ;
   float extent(const gfloat3& dim);
   gfloat4 center_extent();

   void Summary(const char* msg) const ;
   std::string description() const ;
   std::string desc() const ;

   // stuffing gbbox into GBuffer makes it not so straightforward to move to glm::vec3 
   gfloat3 min  ; 
   gfloat3 max  ; 


};





