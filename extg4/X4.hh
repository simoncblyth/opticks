#pragma once

#include "X4_API_EXPORT.hh"
#include <string>
#include <vector>

class G4Material ; 
class G4LogicalSurface ; 

class X4_API X4 
{
    public: 
        static const char* X4GEN_DIR ; 
    public: 
        static const char* ShortName( const std::string& name );
        static const char* Name( const std::string& name );

        template<typename T>
        static const char* ShortName( const T* const obj );

        template<typename T>
        static const char* BaseName( const T* const obj );
        
        template<typename T>
        static const char* Name( const T* const obj );

        template<typename T>
        static int GetItemIndex( const std::vector<T*>* vec, const T* const item );

        static size_t GetOpticksIndex( const G4LogicalSurface* const surf );

        static std::string Array( const double* a, unsigned nv, const char* identifier );  

        template<typename T>
        static std::string Value( T v );  

};


