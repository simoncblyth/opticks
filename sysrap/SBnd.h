#pragma once
/**
SBnd.h : Used to fish material properties out of the boundary buffer
=======================================================================

NB: this only works with the standard set of 8 properties, it 
does not work with scintillator properties as those are not 
stored in the boundary buffer.  

Principal user QBnd.hh

* SSim has several functions that perhaps could be relocated here 
* Note that SLOG logging machinery doesnt work with header only imps 

**/

#include <cassert>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <set>

#include "NP.hh"
#include "sstr.h"
#include "sdigest.h"
#include "sproplist.h"
#include "sidxname.h"



struct SBnd
{
    const NP* bnd ; 

    //static constexpr const unsigned MISSING = ~0u ;
    static constexpr const int MISSING = -1 ;
    const std::vector<std::string>& bnames ; 

    SBnd(const NP* bnd_); 

    std::string getItemDigest( int i, int j, int w=8 ) const ;
    std::string descBoundary() const ;
    int getNumBoundary() const ;
    const char* getBoundarySpec(int idx) const ;
    void        getBoundarySpec(std::vector<std::string>& names, const int* idx , int num_idx ) const ;

    int    getBoundaryIndex(const char* spec) const ;

    void        getBoundaryIndices( std::vector<int>& bnd_idx, const char* bnd_sequence, char delim=',' ) const ;
    std::string descBoundaryIndices( const std::vector<int>& bnd_idx ) const ;

    int    getBoundaryLine(const char* spec, int j) const ;
    static int GetMaterialLine( const char* material, const std::vector<std::string>& specs );
    int    getMaterialLine( const char* material ) const ;

    static std::string DescDigest(const NP* bnd, int w=16) ; 

    std::string desc() const ; 

    void getMaterialNames( std::vector<std::string>& names ) const ; 
    static std::string DescNames( std::vector<std::string>& names ) ; 



    bool findName( int& i, int& j, const char* qname ) const ; 
    static bool FindName( int& i, int& j, const char* qname, const std::vector<std::string>& names ) ; 

    NP* getPropertyGroup(const char* qname, int k=-1) const ;  

    template<typename T>
    void getProperty(std::vector<T>& out, const char* qname, const char* propname ) const ; 


    static void FillMaterialLine( 
         std::vector<int>& mtline, 
         const std::vector<int>& mtindex,
         const std::vector<std::string>& mtname, 
         const std::vector<std::string>& specs 
    ); 


    static void GetSpecsFromString( std::vector<std::string>& specs , const char* specs_, char delim='\n' ); 



    NP* bd_from_optical(const NP* op ) const ; 
    NP* mat_from_bd(const NP* bd) const ; 
    NP* reconstruct_sur() const ; 

};


inline SBnd::SBnd(const NP* bnd_)
    :
    bnd(bnd_),
    bnames(bnd->names)
{
    int num_bnames = bnames.size() ; 
    if( num_bnames == 0 ) std::cerr << "SBnd::SBnd no names from bnd " << ( bnd ? bnd->sstr() : "-" ) << std::endl ; 
    //assert(num_bnames > 0 ); 
}

inline std::string SBnd::getItemDigest( int i, int j, int w ) const 
{
    return sdigest::Item(bnd, i, j, w );   // formerly SSim::GetItemDigest 
}
inline std::string SBnd::descBoundary() const
{
    std::stringstream ss ; 
    for(int i=0 ; i < int(bnames.size()) ; i++) 
       ss << std::setw(2) << i << " " << bnames[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
} 


inline int SBnd::getNumBoundary() const
{
    return bnames.size(); 
}
inline const char* SBnd::getBoundarySpec(int idx) const 
{
    assert( idx < int(bnames.size()) );  
    const std::string& s = bnames[idx]; 
    return s.c_str(); 
}
inline void SBnd::getBoundarySpec(std::vector<std::string>& names, const int* idx , int num_idx ) const 
{
    for(int i=0 ; i < num_idx ; i++)
    {   
        int index = idx[i] ;   
        const char* spec = getBoundarySpec(index);   // 0-based 
        names.push_back(spec); 
    }   
} 


inline int SBnd::getBoundaryIndex(const char* spec) const 
{
    int idx = MISSING ; 
    for(int i=0 ; i < int(bnames.size()) ; i++) 
    {
        if(spec && strcmp(bnames[i].c_str(), spec) == 0) 
        {
            idx = i ; 
            break ; 
        }
    }
    return idx ;  
}

inline void SBnd::getBoundaryIndices( std::vector<int>& bnd_idx, const char* bnd_sequence, char delim ) const 
{
    assert( bnd_idx.size() == 0 ); 

    std::vector<std::string> bnd ; 
    sstr::SplitTrim(bnd_sequence,delim, bnd ); 

    for(int i=0 ; i < int(bnd.size()) ; i++)
    {
        const char* spec = bnd[i].c_str(); 
        if(strlen(spec) == 0) continue ;  
        int bidx = getBoundaryIndex(spec); 
        if( bidx == MISSING ) std::cerr << " i " << i << " invalid spec [" << spec << "]" << std::endl ;      
        assert( bidx != MISSING ); 

        bnd_idx.push_back(bidx) ; 
    }
}

inline std::string SBnd::descBoundaryIndices( const std::vector<int>& bnd_idx ) const 
{
    int num_bnd_idx = bnd_idx.size() ; 
    std::stringstream ss ; 
    for(int i=0 ; i < num_bnd_idx ; i++)
    {
        int bidx = bnd_idx[i] ;  
        const char* spec = getBoundarySpec(bidx); 
        ss
            << " i " << std::setw(3) << i 
            << " bidx " << std::setw(3) << bidx
            << " spec " << spec
            << std::endl 
            ;
    }
    std::string s = ss.str(); 
    return s ; 
}

/**
SBnd::getBoundaryLine
-----------------------

The boundary spec allows to obtain the boundary index, 
the boundary line returned is : 4*boundary_index + j 

**/

inline int SBnd::getBoundaryLine(const char* spec, int j) const 
{
    int num_bnames = bnames.size() ;
    int idx = getBoundaryIndex(spec); 
    bool is_missing = idx == MISSING ; 
    bool is_valid = !is_missing && idx < num_bnames ;

    if(!is_valid) 
    {
        std::cerr
            << " not is_valid " 
            << " spec " << spec
            << " idx " << idx
            << " is_missing " << is_missing 
            << " bnames.size " << bnames.size() 
            << std::endl 
            ;  
    }

    assert( is_valid ); 
    int line = 4*idx + j ;    
    return line ;  
}

/**
SBnd::GetMaterialLine
----------------------

The spec strings are assumed to be "/" delimited : omat/osur/isur/imat
The first omat or imat line matching the *material* argument is returned. 

**/

inline int SBnd::GetMaterialLine( const char* material, const std::vector<std::string>& specs ) // static
{
    int line = MISSING ; 
    for(int i=0 ; i < int(specs.size()) ; i++) 
    {
        std::vector<std::string> elem ; 
        sstr::Split(specs[i].c_str(), '/', elem );  
        const char* omat = elem[0].c_str(); 
        const char* imat = elem[3].c_str(); 

        if(strcmp( material, omat ) == 0 )
        {
            line = i*4 + 0 ; 
            break ; 
        }
        if(strcmp( material, imat ) == 0 )
        {
            line = i*4 + 3 ; 
            break ; 
        }
    }
    return line ; 
}

/**
SBnd::getMaterialLine
-----------------------

Searches the bname spec for the *material* name in omat or imat slots, 
returning the first found.  

**/

inline int SBnd::getMaterialLine( const char* material ) const
{
    return GetMaterialLine(material, bnames);
}




/**
SBnd::DescDigest
--------------------

bnd with shape (44, 4, 2, 761, 4, )::

   ni : boundaries
   nj : 0:omat/1:osur/2:isur/3:imat  
   nk : 0 or 1 property group
   nl : wavelengths
   nm : payload   

::

    2022-04-20 14:53:14.544 INFO  [4031964] [test_DescDigest@133] 
    5acc01c3 79cfae67 79cfae67 5acc01c3  Galactic///Galactic
    5acc01c3 79cfae67 79cfae67 8b22bf98  Galactic///Rock
    8b22bf98 79cfae67 79cfae67 5acc01c3  Rock///Galactic
    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air
    8b22bf98 79cfae67 79cfae67 8b22bf98  Rock///Rock
    8b22bf98 79cfae67 0a5eab3f c2759ba7  Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air
    c2759ba7 79cfae67 79cfae67 8b22bf98  Air///Steel

**/


inline std::string SBnd::DescDigest(const NP* bnd, int w )  // static
{
    int ni = bnd->shape[0] ; 
    int nj = bnd->shape[1] ;
 
    const std::vector<std::string>& names = bnd->names ; 
    assert( int(names.size()) == ni ); 

    std::stringstream ss ; 
    ss << "SBnd::DescDigest" << std::endl ; 
    for(int i=0 ; i < ni ; i++)
    {
        ss << std::setw(3) << i << " " ; 
        for(int j=0 ; j < nj ; j++) 
        {
            std::string dig = sdigest::Item(bnd, i, j ) ;    // formerly SDigestNP::Item
            std::string sdig = dig.substr(0, w); 
            ss << std::setw(w) << sdig << " " ; 
        }
        ss << " " << names[i] << std::endl ; 
    }
    std::string s = ss.str();  
    return s ; 
}


inline std::string SBnd::desc() const 
{
    return DescDigest(bnd,8) ;
}

/**
SBnd::getMaterialNames
-----------------------

HMM: name order not the original one 

**/

inline void SBnd::getMaterialNames( std::vector<std::string>& names ) const 
{
    for(int i=0 ; i < int(bnames.size()) ; i++) 
    {
        std::vector<std::string> elem ; 
        sstr::Split(bnames[i].c_str(), '/', elem );  
        const char* omat = elem[0].c_str(); 
        const char* imat = elem[3].c_str(); 

        if(std::find(names.begin(), names.end(), omat) == names.end()) names.push_back(omat); 
        if(std::find(names.begin(), names.end(), imat) == names.end()) names.push_back(imat); 
    }
}
inline std::string SBnd::DescNames( std::vector<std::string>& names ) 
{
    std::stringstream ss ; 
    for(int i=0 ; i < int(names.size()) ; i++) ss << names[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}



/**
SBnd::findName
----------------

Returns the first (i,j)=(bidx,species) with element name 
matching query name *qname*. 

bidx 
    0-based boundary index 

species
    0,1,2,3 for omat/osur/isur/imat

**/

inline bool SBnd::findName( int& i, int& j, const char* qname ) const 
{
    return FindName(i, j, qname, bnames ) ; 
}

inline bool SBnd::FindName( int& i, int& j, const char* qname, const std::vector<std::string>& names ) 
{
    i = -1 ; 
    j = -1 ; 
    for(int b=0 ; b < int(names.size()) ; b++) 
    {
        std::vector<std::string> elem ; 
        sstr::Split(names[b].c_str(), '/', elem );  

        for(int s=0 ; s < 4 ; s++)
        {
            const char* name = elem[s].c_str(); 
            if(strcmp(name, qname) == 0 )
            {
                i = b ; 
                j = s ; 
                return true ; 
            }
        }
    }
    return false ;  
}

/**
SBnd::getPropertyGroup
------------------------

Returns an array of material or surface properties selected by *qname* eg "Water".
For example with a source bnd array of shape  (44, 4, 2, 761, 4, )
the expected spawned property group  array shape depends on the value of k:

k=-1
     (2, 761, 4,)   eight property values across wavelength domain
k=0
     (761, 4)       four property values across wavelength domain 
k=1
     (761, 4)

**/

inline NP* SBnd::getPropertyGroup(const char* qname, int k) const 
{
    int i, j ; 
    bool found = findName(i, j, qname); 
    assert(found); 
    return bnd->spawn_item(i,j,k);  
}

/**
SBnd::getProperty
------------------

::

    // <f8(44, 4, 2, 761, 4, )
    int species = 0 ; 
    int group = 1 ; 
    int wavelength = -1 ; 
    int prop = 0 ; 

**/

template<typename T>
inline void SBnd::getProperty(std::vector<T>& out, const char* qname, const char* propname ) const 
{
    assert( sizeof(T) == bnd->ebyte ); 

    int boundary, species ; 
    bool found_qname = findName(boundary, species, qname); 
    assert(found_qname); 

    //const SBndProp* bp = FindMaterialProp(propname); 

    const sproplist* pm = sproplist::Material(); 
    const sprop* bp = pm->findProp(propname) ; 
    assert(bp); 

    int group = bp->group ; 
    int prop = bp->prop  ; 
    int wavelength = -1 ;   // slice dimension 

    bnd->slice(out, boundary, species, group, wavelength, prop );  
}


/**
SBnd::FillMaterialLine
-----------------------

Used from SSim::import_bnd

Uses the "specs" boundary name list to convert 
all the stree::mtname into st->mtline 

These mtline are used to lookup material properties
from the boundary texture array. 

**/

inline void SBnd::FillMaterialLine( 
     std::vector<int>& mtline, 
     const std::vector<int>& mtindex,
     const std::vector<std::string>& mtname, 
     const std::vector<std::string>& specs )
{
    assert( mtindex.size() == mtname.size() );  
    int num_mt = mtindex.size() ; 
    mtline.clear(); 

    for(int i=0 ; i < num_mt ; i++)
    {
        const char* mt = mtname[i].c_str() ; 
        int mt_line = GetMaterialLine(mt, specs) ;  // unsigned ~0u "MISSING" becomes int -1 
        mtline.push_back(mt_line); 
    }
}


inline void SBnd::GetSpecsFromString( std::vector<std::string>& specs , const char* specs_, char delim )
{
    std::stringstream ss;
    ss.str(specs_)  ;
    std::string s;
    while (std::getline(ss, s, delim)) if(!sstr::Blank(s.c_str())) specs.push_back(s) ;
    std::cout << " specs_ [" << specs_ << "] specs.size " << specs.size()  ;   
}



/**
SBnd::bd_from_optical
----------------------

Hmm, but the bd is new. For reconstruction of old_mat from the old_bnd
I need to use an old_bd. Can recreate that by folding first column of old_optical 
and subtracting 1.::

    bd = np.array( t.old_optical[:,0].reshape(-1,4), dtype=np.int32 ) - 1 

**/

inline NP* SBnd::bd_from_optical(const NP* op ) const 
{

    /* 
    //old optical was unsigned:( num_bd*4, 4 )
    assert( op && op->uifc == 'u' && op->shape.size() == 2 && op->shape[1] == 4 ) ; 
    int num_op = op->shape[0] ; 
    assert( num_op % 4 == 0 ); 
    int num_bd = num_op / 4 ; 
    */

    // new optical is int:(num_bd, 4, 4)  
    assert( op && op->uifc == 'i' && op->shape.size() == 3 && op->shape[1] == 4 && op->shape[2] == 4 ) ; 
    int num_bd = op->shape[0] ; 


    const int* op_v = op->cvalues<int>() ; 
    int ni = num_bd ; 
    int nj = 4 ; 

    NP* bd = NP::Make<int>(ni, nj) ;  
    int* bd_v = bd->values<int>() ; 

    for(int i=0 ; i < ni ; i++) 
    for(int j=0 ; j < nj ; j++)
    {
        int src_index = (i*nj + j)*4 ;
        int dst_index = i*nj + j ;  
        bd_v[dst_index] = int(op_v[src_index]) - 1 ; 
    }
    bd->set_names(bnd->names) ; 
    return bd ; 
}


/**
SBnd::mat_from_bd
-----------------------

Creating mat from bd obtained from optical 1st column and bnd array 
is of course the wrong order. Typically the converse is done 
the bnd is created by interleaving together the mat and sur arrays. 
However in the old GGeo/X4 workflow the mat and sur arrays were not 
persisted with the bnd being created directly.

Hence to facilitate development of mat,sur and bnd in new workflow
it is useful to reconstruct what the old workflow mat and sur would 
actually have been by pulling apart the old bnd and putting it together
to form what the missing old mat and sur would have been. 

When this is applied to an old bnd the old mat it provides can 
be compared with new mat from stree::create_mat which uses sstandard::mat

No gaps for materials, all mt indices in the bd int4 vector form contiguous range::

    In [19]: np.unique( np.hstack( (t.bd[:,0], t.bd[:,3]) ) )
    Out[19]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=int32)

**/

inline NP* SBnd::mat_from_bd(const NP* bd) const 
{
    // 0. check bd array is as expected

    assert( bd && bd->uifc == 'i' && bd->shape.size() == 2 && bd->shape[1] == 4 ) ; 
    assert( int(bd->names.size()) == bd->shape[0] ); 
    const int* bd_v = bd->cvalues<int>();  

    // 1. check consistency between bd (num_bd, 4) int pointers
    //    and bnd data eg (53, 4, 2, 761, 4, )

    const std::vector<int>& bnd_sh = bnd->shape ; 
    assert( bnd && bnd->uifc == 'f' && bnd->ebyte == 8 ); 
    assert( int(bnd_sh.size()) == 5 ) ; 
    assert( bnd_sh[0] == bd->shape[0] );
    const double* bnd_v = bnd->cvalues<double>() ; 


    // 2. first bd pass to find the number of material indices

    int ni = bd->shape[0] ; 
    int nj = 4 ; 

    std::set<sidxname, sidxname_ordering> mm ;  
    for(int i=0 ; i < ni ; i++)
    {
        int omat = bd_v[i*nj+0] ; 
        int imat = bd_v[i*nj+3] ; 

        const char* bdn = bd->names[i].c_str() ; 
        std::vector<std::string> elem ; 
        sstr::Split(bdn, '/', elem );  
        const char* omat_ = elem[0].c_str(); 
        const char* imat_ = elem[3].c_str(); 

        sidxname om(omat,omat_) ; 
        sidxname im(imat,imat_) ; 

        mm.insert(om); 
        mm.insert(im); 
    }

    std::vector<sidxname> vmm(mm.begin(),mm.end()) ; 
    int num_mt = vmm.size() ;

    // 3. assert that bd (omat,imat) contains contiguous set of material indices

    for(int i=0 ; i < num_mt ; i++) assert( i == vmm[i].idx ) ; 
    for(int i=0 ; i < num_mt ; i++) std::cout << vmm[i].desc() << std::endl; 

    std::vector<std::string> names ; 
    for(int i=0 ; i < num_mt ; i++)
    {
        names.push_back( vmm[i].name ) ;  // HMM null termination ?
    }

    std::cout 
        << "SBnd::mat_from_bd"
        << " bd->names.size " << bd->names.size() 
        << " bnd->names.size " << bnd->names.size() 
        << " bnd->shape[0] " << bnd->shape[0]
        << " bnd->sstr() " << bnd->sstr()
        << " ni " << ni 
        << std::endl 
        ; 


    // 4. prep mat array 

    int np = bnd_sh[2]*bnd_sh[3]*bnd_sh[4] ; // num payload values for mat (or sur)

    NP* mat = NP::Make<double>(num_mt,bnd_sh[2], bnd_sh[3], bnd_sh[4] ) ; 
    mat->set_names(names) ; 

    double* mat_v = mat->values<double>(); 

    // 4. second bd pass to populate reconstructed mat array 
    //    (note that each mat may be written multiple times
    //     but that doesnt matter as all the same)

    for(int i=0 ; i < ni ; i++)
    {
        int omat = bd_v[i*nj+0] ; 
        assert( omat > -1 && omat < num_mt ); 
        for(int p=0 ; p < np ; p++) mat_v[omat*np+p] = bnd_v[(i*4+0)*np+p] ; 

        int imat = bd_v[i*nj+3] ; 
        assert( imat > -1 && imat < num_mt ); 
        for(int p=0 ; p < np ; p++) mat_v[imat*np+p] = bnd_v[(i*4+3)*np+p] ; 
    }

    return mat ; 
}


/**
Quite a few gaps for surfaces, not all surfaces are referenced from the bd int4 vector:: 

    In [20]: np.unique( np.hstack( (t.bd[:,1], t.bd[:,2]) ) )
    Out[20]: array([-1,  0,  1,  2,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37, 38, 39], dtype=int32)

**/

inline NP* SBnd::reconstruct_sur() const 
{
    return nullptr ; 
}

