/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <sstream>
#include <cstring>
#include <algorithm>

#include "SLabelCache.hh"
#include "X4.hh"
#include "X4LogicalBorderSurfaceTable.hh"

#include "SGDML.hh"
#include "BStr.hh"
#include "BFile.hh"

#include "PLOG.hh"

const plog::Severity X4::LEVEL = PLOG::EnvLevel("X4", "debug" ); 

SLabelCache<int>* X4::surface_index_cache = nullptr ; 

const char* X4::X4GEN_DIR = "$TMP/x4gen" ; 

const char* X4::ShortName( const std::string& name )
{
    char* shortname = BStr::trimPointerSuffixPrefix(name.c_str(), NULL) ;  
    return strdup( shortname );
}

const char* X4::Name( const std::string& name )
{
    return strdup( name.c_str() );
}

const char* X4::BaseName( const std::string& name)
{
    const std::string base = BFile::Name(name.c_str());  
    return ShortName(base) ;
}

const char* X4::BaseNameAsis( const std::string& name)
{
    const std::string base = BFile::Name(name.c_str());  
    return strdup(base.c_str()) ;
}




template<typename T>
const char* X4::ShortName( const T* const obj )
{    
    if(obj == NULL) return NULL ; 
    const std::string& name = obj->GetName();
    return ShortName(name);
}


template<typename T>
const char* X4::BaseName( const T* const obj )
{    
    if(obj == NULL) return NULL ; 
    const std::string& name = obj->GetName();
    return BaseName(name);
}

template<typename T>
const char* X4::BaseNameAsis( const T* const obj )
{    
    if(obj == NULL) return NULL ; 
    const std::string& name = obj->GetName();
    return BaseNameAsis(name);
}



template<typename T>
const char* X4::GDMLName( const T* const obj )
{    
    if(obj == NULL) return NULL ; 
    const std::string& name = obj->GetName();

    const char* base = BaseNameAsis(name); 
    bool addPointerToName = true ; 
    std::string gn = SGDML::GenerateName( base, obj, addPointerToName  )  ;
    free((void*)base); 
    return strdup(gn.c_str());
}





template<typename T>
const char* X4::Name( const T* const obj )
{    
    if(obj == NULL) return NULL ; 
    const std::string& name = obj->GetName();
    return Name(name);
}


template<typename T>
int X4::GetItemIndex( const std::vector<T*>* vec, const T* const item )
{
    typedef std::vector<T*> V ;
    typename V::const_iterator pos = std::find( vec->begin(),  vec->end(), item ) ; 
    int index = pos == vec->end() ? -1 : std::distance( vec->begin(), pos ); 
    return index ; 
}


class G4LogicalSurface ;
class G4LogicalBorderSurface ;
class G4LogicalSkinSurface ;
class G4VPhysicalVolume ;
class G4Material ;

template X4_API int X4::GetItemIndex<G4Material>(const std::vector<G4Material*>*, const G4Material* const);
template X4_API int X4::GetItemIndex<G4LogicalBorderSurface>(const std::vector<G4LogicalBorderSurface*>*, const G4LogicalBorderSurface* const);
template X4_API int X4::GetItemIndex<G4LogicalSkinSurface>(const std::vector<G4LogicalSkinSurface*>*, const G4LogicalSkinSurface* const);


#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalSurface.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "X4Named.hh"
#include "X4PhysicalVolume.hh"

template X4_API const char* X4::Name<G4OpticalSurface>(const G4OpticalSurface* const);
template X4_API const char* X4::Name<G4LogicalBorderSurface>(const G4LogicalBorderSurface* const);
template X4_API const char* X4::Name<G4LogicalSkinSurface>(const G4LogicalSkinSurface* const);
template X4_API const char* X4::Name<G4LogicalSurface>(const G4LogicalSurface* const);
template X4_API const char* X4::Name<G4VPhysicalVolume>(const G4VPhysicalVolume* const);
template X4_API const char* X4::Name<G4LogicalVolume>(const G4LogicalVolume* const);
template X4_API const char* X4::Name<G4Material>(const G4Material* const);
template X4_API const char* X4::Name<X4Named>(const X4Named* const);
template X4_API const char* X4::Name<X4PhysicalVolume>(const X4PhysicalVolume* const);

template X4_API const char* X4::ShortName<G4OpticalSurface>(const G4OpticalSurface* const);
template X4_API const char* X4::ShortName<G4LogicalBorderSurface>(const G4LogicalBorderSurface* const);
template X4_API const char* X4::ShortName<G4LogicalSkinSurface>(const G4LogicalSkinSurface* const);
template X4_API const char* X4::ShortName<G4LogicalSurface>(const G4LogicalSurface* const);
template X4_API const char* X4::ShortName<G4VPhysicalVolume>(const G4VPhysicalVolume* const);
template X4_API const char* X4::ShortName<G4LogicalVolume>(const G4LogicalVolume* const);
template X4_API const char* X4::ShortName<G4Material>(const G4Material* const);
template X4_API const char* X4::ShortName<X4Named>(const X4Named* const);
template X4_API const char* X4::ShortName<X4PhysicalVolume>(const X4PhysicalVolume* const);

template X4_API const char* X4::BaseName<G4OpticalSurface>(const G4OpticalSurface* const);
template X4_API const char* X4::BaseName<G4LogicalBorderSurface>(const G4LogicalBorderSurface* const);
template X4_API const char* X4::BaseName<G4LogicalSkinSurface>(const G4LogicalSkinSurface* const);
template X4_API const char* X4::BaseName<G4LogicalSurface>(const G4LogicalSurface* const);
template X4_API const char* X4::BaseName<G4VPhysicalVolume>(const G4VPhysicalVolume* const);
template X4_API const char* X4::BaseName<G4LogicalVolume>(const G4LogicalVolume* const);
template X4_API const char* X4::BaseName<G4Material>(const G4Material* const);
template X4_API const char* X4::BaseName<X4Named>(const X4Named* const);
template X4_API const char* X4::BaseName<X4PhysicalVolume>(const X4PhysicalVolume* const);

template X4_API const char* X4::BaseNameAsis<G4OpticalSurface>(const G4OpticalSurface* const);
template X4_API const char* X4::BaseNameAsis<G4LogicalBorderSurface>(const G4LogicalBorderSurface* const);
template X4_API const char* X4::BaseNameAsis<G4LogicalSkinSurface>(const G4LogicalSkinSurface* const);
template X4_API const char* X4::BaseNameAsis<G4LogicalSurface>(const G4LogicalSurface* const);
template X4_API const char* X4::BaseNameAsis<G4VPhysicalVolume>(const G4VPhysicalVolume* const);
template X4_API const char* X4::BaseNameAsis<G4LogicalVolume>(const G4LogicalVolume* const);
template X4_API const char* X4::BaseNameAsis<G4Material>(const G4Material* const);
template X4_API const char* X4::BaseNameAsis<X4Named>(const X4Named* const);
template X4_API const char* X4::BaseNameAsis<X4PhysicalVolume>(const X4PhysicalVolume* const);

template X4_API const char* X4::GDMLName<G4OpticalSurface>(const G4OpticalSurface* const);
template X4_API const char* X4::GDMLName<G4LogicalBorderSurface>(const G4LogicalBorderSurface* const);
template X4_API const char* X4::GDMLName<G4LogicalSkinSurface>(const G4LogicalSkinSurface* const);
template X4_API const char* X4::GDMLName<G4LogicalSurface>(const G4LogicalSurface* const);
template X4_API const char* X4::GDMLName<G4VPhysicalVolume>(const G4VPhysicalVolume* const);
template X4_API const char* X4::GDMLName<G4LogicalVolume>(const G4LogicalVolume* const);
template X4_API const char* X4::GDMLName<G4Material>(const G4Material* const);
template X4_API const char* X4::GDMLName<X4Named>(const X4Named* const);
template X4_API const char* X4::GDMLName<X4PhysicalVolume>(const X4PhysicalVolume* const);







template<typename T>
std::string X4::Value( T v )
{
    std::stringstream ss ; 
    ss << std::fixed << v ; 
    return ss.str(); 
}

template X4_API std::string X4::Value<float>(float);
template X4_API std::string X4::Value<double>(double);
template X4_API std::string X4::Value<int>(int);
template X4_API std::string X4::Value<unsigned>(unsigned);





template<typename T>
std::string X4::Argument( T v )
{
    std::stringstream ss ; 
    ss << std::fixed << v ; 
    return ss.str(); 
}

// template specialization 
template<>
std::string X4::Argument<std::string>( std::string v )
{
    std::stringstream ss ; 
    ss << v ; 
    return ss.str(); 
}

template<>
std::string X4::Argument<double>( double  v )
{
    std::string code ;  
    if( v == CLHEP::twopi ) code = "CLHEP::twopi" ;  
    if( v == CLHEP::pi ) code = "CLHEP::pi" ;  

    std::stringstream ss ; 
    if(!code.empty())
    {
        ss << code ; 
    }
    else
    {
        ss << std::fixed << v ; 
    } 
    return ss.str(); 
}


template X4_API std::string X4::Argument<float>(float);
//template X4_API std::string X4::Argument<double>(double);
template X4_API std::string X4::Argument<int>(int);
template X4_API std::string X4::Argument<unsigned>(unsigned);
//template X4_API std::string X4::Argument<std::string>(std::string);









std::string X4::Array( const double* a, unsigned nv, const char* identifier )
{
    std::stringstream ss ; 

    ss << "double " 
       << identifier 
       << "[]" 
       << "="
       << " { "
       ;

    for(unsigned i=0 ; i < nv ; i++ ) 
    {
        ss << std::fixed << a[i] ;
        if( i < nv - 1) ss << ", " ;
    }

    ss << " } "
       << " ; "
       ;  

    return ss.str(); 
}


/**
X4::MakeSurfaceIndexCache
===========================

Implicit assumption that are not going to be adding more surfaces, 
so only use this once that is true. 

Border and skin surfaces are listed separately by G4 but together by Opticks
so need to define the following convention for surface indices: 

* border surfaces follow the Geant4 order with matched indices
* skin surfaces follow Geant4 order but with indices offset by the number of border surfaces 

* NB for these indices to remain valid, clearly must not add/remove 
  surfaces after accessing the indices 


This was implemeted to avoid the issue::

   notes/issues/zike_reports_slow_surface_initization_with_geant4_1071_when_have_thousands_of_surfaces.rst

**/

const int X4::MISSING_SURFACE = -1 ; 

SLabelCache<int>* X4::MakeSurfaceIndexCache()
{
    SLabelCache<int>* _cache = new SLabelCache<int>(MISSING_SURFACE) ; 

    size_t num_lbs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ; 
    size_t num_sks = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ; 
    LOG(LEVEL) << "[ " << " num_lbs " << num_lbs << " num_sks " << num_sks ; 

    const G4LogicalBorderSurfaceTable* lbs_table = G4LogicalBorderSurface::GetSurfaceTable() ; 
    const std::vector<G4LogicalBorderSurface*>* lbs_vec = X4LogicalBorderSurfaceTable::PrepareVector(lbs_table); 
    assert( num_lbs == lbs_vec->size() );
  
    for(int ibs=0 ; ibs < lbs_vec->size() ; ibs++)
    {
        const G4LogicalBorderSurface* bs = (*lbs_vec)[ibs] ; 
        _cache->add( bs, ibs ); 
    }

    const G4LogicalSkinSurfaceTable*   sks_vec = G4LogicalSkinSurface::GetSurfaceTable() ; 
    assert( num_sks == sks_vec->size() );
    for(int isk=0 ; isk < sks_vec->size() ; isk++)
    {
        const G4LogicalSkinSurface* sk = (*sks_vec)[isk] ; 
        _cache->add( sk, isk + num_lbs ); 
    }

    LOG(LEVEL) << "]" ; 
    return _cache ; 
}


/**
size_t X4::GetOpticksIndex( const G4LogicalSurface* const surf )
==================================================================

NB Implicit assumption that are not going to be adding more surfaces, 
so only use this once that is true. 
 
**/

size_t X4::GetOpticksIndex( const G4LogicalSurface* const surf )
{
    if( surface_index_cache == nullptr ) surface_index_cache = MakeSurfaceIndexCache() ; 
    int index = surface_index_cache->find(surf ); 
    assert( index != MISSING_SURFACE );  
    //LOG(LEVEL) << " index " << index ; 
    return index ; 
}

 
