
#include "scuda.h"

#include "CSGSolid.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "sstr.h"
#include "SLOG.hh"

const plog::Severity CSGSolid::LEVEL = SLOG::EnvLevel("CSGSolid", "DEBUG") ; 


bool CSGSolid::IsDiff( const CSGSolid& a , const CSGSolid& b ) // static 
{
    return false ; 
}

CSGSolid CSGSolid::Make( const char* label_, int numPrim_, int primOffset_ )
{
    CSGSolid so = {} ; 

    //strncpy( so.label, label_, sizeof(so.label) );
    sstr::truncated_copy( so.label, label_, sizeof(so.label) );  

    so.numPrim = numPrim_ ; 
    so.primOffset = primOffset_ ; 
    so.type = STANDARD_SOLID ;  
    so.center_extent = make_float4(0.f, 0.f, 0.f, 0.f) ;  // changed later 

    return so ; 
}


std::string CSGSolid::desc() const 
{
    std::string label16(label, 16); 
    std::stringstream ss ; 
    ss << "CSGSolid " 
       << std::setw(16) << label16.c_str()
       << " primNum/Offset " 
       << std::setw(5) << numPrim 
       << std::setw(5) << primOffset
       << " ce " << center_extent
       ; 

    if( type == ONE_PRIM_SOLID ) ss << " ONE_PRIM_SOLID " ; 
    if( type == ONE_NODE_SOLID ) ss << " ONE_NODE_SOLID " ; 
    if( type == DEEP_COPY_SOLID ) ss << " DEEP_COPY_SOLID " ; 
    if( type == KLUDGE_BBOX_SOLID ) ss << " KLUDGE_BBOX_SOLID " ; 

    std::string s = ss.str(); 
    return s ; 
}

const char* CSGSolid::getLabel() const 
{
    std::string lab(label, 16);   // array size 16 
    return strdup(lab.c_str()) ;   // avoid newline? 
}

bool CSGSolid::labelMatch(const char* label_) const 
{
    return strncmp(label, label_, sizeof(label)) == 0 ;
}


/**
CSGSolid::getIntent
--------------------

Replace former getLabelPrefix which returned label[0] with a dedicated 
field as when used with general names, some will have unintended effects. 

**/
char CSGSolid::getIntent() const
{
    return intent ; 
}

/**
CSGSolid::setIntent
--------------------

Canonical usage is from CSGImport::importSolid which invokes one of two methods for each solid
depending on stree::get_ridx_type::

1. CSGImport::importSolidGlobal  
   
   * for ridx_type 'R' analytic global solid (from stree.h "rem" vector of nodes) 
   * for ridx_type 'T' triangulated global solid (from stree.h "tri" vector of nodes)

2. CSGImport::importSolidFactor

   * for ridx_type 'F' analytic factor solid 

**/

void CSGSolid::setIntent(char _intent)
{
    assert(_intent == 'R' || _intent == 'F' || _intent == 'T' ); 
    intent = _intent ; 
} 

void CSGSolid::CopyIntent( CSGSolid* d,  const CSGSolid* s )
{
    d->intent = s->intent ; 
    d->pad0 = s->pad0 ; 
    d->pad1 = s->pad1 ; 
    d->pad2 = s->pad2 ; 
}



std::string CSGSolid::MakeLabel(const char* typ0, unsigned idx0, char delim )
{
    std::stringstream ss ; 
    ss << typ0 ; 
    if(delim != '\0') ss << delim ; 
    ss  << idx0 ; 
    std::string s = ss.str();  
    return s ; 
}

std::string CSGSolid::MakeLabel(char typ0, unsigned idx0 ) // static
{
    std::stringstream ss ; 
    ss << typ0 << idx0 ; 
    std::string s = ss.str();  
    return s ; 
}

int CSGSolid::ParseLabel( const char* label_, char& typ0, unsigned& idx0 ) // static 
{
    size_t sz = strlen(label_); 
    if( sz < 2 ) return 1 ; 
    typ0 = label_[0] ; 
    idx0 = std::atoi(label_+1) ; 
    return 0 ; 
}

int CSGSolid::get_ridx() const 
{
    char typ0 ; 
    unsigned idx0 ; 
    int rc = ParseLabel(label, typ0, idx0 ); 
    return rc == 0 ? idx0 : -1 ;    
}







std::string CSGSolid::MakeLabel(char typ0, unsigned idx0, char typ1, unsigned idx1  )
{
    std::stringstream ss ; 
    ss << typ0 << idx0 << typ1 << idx1 ; 
    std::string s = ss.str();  
    return s ; 
}
std::string CSGSolid::MakeLabel(char typ0, unsigned idx0, char typ1, unsigned idx1, char typ2, unsigned idx2  )
{
    std::stringstream ss ; 
    ss << typ0 << idx0 << typ1 << idx1 << typ2 << idx2 ; 
    std::string s = ss.str();  
    return s ; 
}




#endif

