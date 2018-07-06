#include <cstring>
#include <algorithm>

#include "X4.hh"
#include "BStr.hh"
#include "BFile.hh"

const char* X4::ShortName( const std::string& name )
{
    char* shortname = BStr::trimPointerSuffixPrefix(name.c_str(), NULL) ;  
    return strdup( shortname );
}

const char* X4::Name( const std::string& name )
{
    return strdup( name.c_str() );
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
    const std::string base = BFile::Name(name.c_str());  
    return ShortName(base);
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

template X4_API const char* X4::Name<G4OpticalSurface>(const G4OpticalSurface* const);
template X4_API const char* X4::Name<G4LogicalBorderSurface>(const G4LogicalBorderSurface* const);
template X4_API const char* X4::Name<G4LogicalSkinSurface>(const G4LogicalSkinSurface* const);
template X4_API const char* X4::Name<G4LogicalSurface>(const G4LogicalSurface* const);
template X4_API const char* X4::Name<G4VPhysicalVolume>(const G4VPhysicalVolume* const);
template X4_API const char* X4::Name<G4LogicalVolume>(const G4LogicalVolume* const);
template X4_API const char* X4::Name<G4Material>(const G4Material* const);

template X4_API const char* X4::ShortName<G4OpticalSurface>(const G4OpticalSurface* const);
template X4_API const char* X4::ShortName<G4LogicalBorderSurface>(const G4LogicalBorderSurface* const);
template X4_API const char* X4::ShortName<G4LogicalSkinSurface>(const G4LogicalSkinSurface* const);
template X4_API const char* X4::ShortName<G4LogicalSurface>(const G4LogicalSurface* const);
template X4_API const char* X4::ShortName<G4VPhysicalVolume>(const G4VPhysicalVolume* const);
template X4_API const char* X4::ShortName<G4LogicalVolume>(const G4LogicalVolume* const);
template X4_API const char* X4::ShortName<G4Material>(const G4Material* const);

template X4_API const char* X4::BaseName<G4OpticalSurface>(const G4OpticalSurface* const);
template X4_API const char* X4::BaseName<G4LogicalBorderSurface>(const G4LogicalBorderSurface* const);
template X4_API const char* X4::BaseName<G4LogicalSkinSurface>(const G4LogicalSkinSurface* const);
template X4_API const char* X4::BaseName<G4LogicalSurface>(const G4LogicalSurface* const);
template X4_API const char* X4::BaseName<G4VPhysicalVolume>(const G4VPhysicalVolume* const);
template X4_API const char* X4::BaseName<G4LogicalVolume>(const G4LogicalVolume* const);
template X4_API const char* X4::BaseName<G4Material>(const G4Material* const);






/**
size_t X4::GetOpticksIndex( const G4LogicalSurface* const surf )
==================================================================

Border and skin surfaces are listed separately by G4 but together by Opticks
so need to define the following convention for surface indices: 

* border surfaces follow the Geant4 order with matched indices
* skin surfaces follow Geant4 order but with indices offset by the number of border surfaces 

* NB for these indices to remain valid, clearly must not add/remove 
  surfaces after accessing the indices 

 
**/

size_t X4::GetOpticksIndex( const G4LogicalSurface* const surf )
{
    size_t num_lbs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ; 
    size_t num_sks = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ; 

    const G4LogicalBorderSurfaceTable* lbs_table = G4LogicalBorderSurface::GetSurfaceTable() ; 
    const G4LogicalSkinSurfaceTable*   sks_table = G4LogicalSkinSurface::GetSurfaceTable() ; 

    assert( num_lbs == lbs_table->size() );
    assert( num_sks == sks_table->size() );

    const G4LogicalBorderSurface* const lbs = dynamic_cast<const G4LogicalBorderSurface* const>(surf);
    const G4LogicalSkinSurface*   const sks = dynamic_cast<const G4LogicalSkinSurface* const>(surf);

    assert( (lbs == NULL) ^ (sks == NULL) );   // one or other must be NULL, but not both   

    int idx_lbs = lbs ? GetItemIndex<G4LogicalBorderSurface>( lbs_table, lbs ) : -1 ;    
    int idx_sks = sks ? GetItemIndex<G4LogicalSkinSurface>(   sks_table, sks ) : -1 ;    

    assert( (idx_lbs == -1) ^ (idx_sks == -1) ); // one or other must be -1, but not both 

    return idx_lbs > -1 ? idx_lbs : idx_sks + num_lbs ;     
}

 
