#pragma once

/**
SStackFrame
=============

Used for stack frame introspection based on *cxxabi.h*

**/


#include "SYSRAP_API_EXPORT.hh"
#include <ostream>

struct SYSRAP_API SStackFrame
{
    SStackFrame( char* line ) ;
    ~SStackFrame();

    void parse();
    char* demangle(); // fails for non C++ symbols
    void dump();
    void dump(std::ostream& out);

    char* line ; 
    char* name ; 
    char* offset ;
    char* end_offset ;
 
    char* func ;    // only func is "owned"
};


#include <iostream>
#include <iomanip>
#include <cxxabi.h>


inline SStackFrame::SStackFrame(char* line_)
    :
    line(line_),
    name(NULL),
    offset(NULL),
    end_offset(NULL),
    func(NULL)
{
    parse(); 
} 

inline SStackFrame::~SStackFrame()
{
    free(func);  
}

inline void SStackFrame::dump()
{
    std::ostream& out = std::cout ;
    dump(out);    
}



#ifdef __APPLE__
inline void SStackFrame::parse()
{
    /**
    4   libG4processes.dylib                0x00000001090baee5 _ZN12G4VEmProcess36PostStepGetPhysicalInteractionLengthERK7G4TrackdP16G4ForceCondition + 661 
    **/ 
    for( char *p = line ; *p ; ++p )
    {   
        if (( *p == '_' ) && ( *(p-1) == ' ' )) name = p-1;  // starting from first underscore after space 
        else if ( *p == '+' ) offset = p-1;
    }   

    if( name && offset && ( name < offset ))
    {   
        *name++ = '\0';    // plant terminator into line
        *offset++ = '\0';  // plant terminator into name  
        func = demangle(); 
    }  
}


inline void SStackFrame::dump(std::ostream& out)
{
     if( func )
     {
         out 
             << std::setw(30) << std::left << line  
             << " "
             << std::setw(100) << std::left << func
             << " "
             << std::setw(10) << std::left << offset 
             << " " 
             << std::endl 
             ;
     }
     else
     {
         out << line << std::endl ; 

     }
}

#else
inline void SStackFrame::parse()
{
    /**
    /home/blyth/local/opticks/externals/lib64/libG4tracking.so(_ZN10G4VProcess12PostStepGPILERK7G4TrackdP16G4ForceCondition+0x42) [0x7ffff36ff9b2] 
    **/
    for( char *p = line ; *p ; ++p )
    {   
        if ( *p == '(' ) name = p;
        else if ( *p == '+' ) offset = p;
        else if ( *p == ')' && ( offset || name )) end_offset = p;
    }   

    if( name && end_offset && ( name < end_offset ))
    {   
        *name++ = '\0';    // plant terminator into line
        *end_offset++ = '\0';  // plant terminator into name  
        if(offset) *offset++ = '\0' ; 
        func = demangle(); 
    }  
}
inline void SStackFrame::dump(std::ostream& out)
{
     if( func )
     {
         out 
               << std::setw(25) << std::left << ( end_offset ? end_offset : "" )  // addr
               << " " << std::setw(10) << std::left << offset 
               << " " << std::setw(60) << line  
               << " " << std::setw(60) << std::left << func
               << std::endl 
               ;
     }
     else
     {
         out  << line << std::endl ; 

     }
}
#endif

inline char* SStackFrame::demangle() // demangling fails for non C++ symbols
{
    int status;
    char* ret = abi::__cxa_demangle( name, NULL, NULL, &status );
    return status == 0 ? ret : NULL ; 
}


