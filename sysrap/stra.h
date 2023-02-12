#pragma once
/**
stra.h
========

Following some investigations of Geant4 transform handling 
and noting that inverses are being done their at source
have concluded that dealing with transforms together with 
their inverses is not worth the overhead and complication. 
Of course inverting should be minimized. 

Hence are bringing over functionality from stran.h as its needed 
in new code. 

**/


#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>



template<typename T>
struct stra
{
    static glm::tmat4x4<T> Translate( const T tx, const T ty, const T tz, const T sc, bool flip=false ); 
    static glm::tmat4x4<T> Translate( const glm::tvec3<T>& tlate, bool flip=false ); 

}; 

template<typename T>
inline glm::tmat4x4<T> stra<T>::Translate(  const T tx, const T ty, const T tz, const T sc, bool flip )
{
    glm::tvec3<T> tlate(tx*sc,ty*sc,tz*sc); 
    return Translate(tlate, flip) ; 
}

template<typename T>
inline glm::tmat4x4<T> stra<T>::Translate( const glm::tvec3<T>& tlate, bool flip )
{
    glm::tmat4x4<T> tr = glm::translate(glm::tmat4x4<T>(1.), tlate ) ;
    return flip ? glm::transpose(tr) : tr ; 
}









