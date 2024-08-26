#pragma once
/**
SLabel.h
=========

After the fact lookups of mmlable indices from labels. 
This was used to provide primitive "post-hoc" trimesh control 
prior to implementation of proper tri control. 

**/


#include <algorithm>
#include <vector>
#include <fstream>

#include "spath.h"
#include "sstr.h"

struct SLabel
{
    static bool IsIdxLabelListed(const std::vector<std::string>& label, unsigned idx, const char* ls, char delim=',' );

    static int FindIdxWithLabel(const std::vector<std::string>& label, const char* q_mml);

    static SLabel* Load(const char* path); 
    static constexpr const char* GEOMLoadPath = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/mmlabel.txt" ;  
    static SLabel* GEOMLoad(); 

    const std::vector<std::string>& label ; 

    SLabel(const std::vector<std::string>& label );  
    int  findIdxWithLabel(const char* q_mml) const ; 
    void findIndicesWithListedLabels( std::vector<unsigned>& indices, const char* ls, char delim );

    std::string detail() const ; 
}; 



/**
SLabel::IsIdxLabelListed
-------------------------

1. forms SLabel instance with the label vector
2. returns true when the 0-based idx label which must be less than the number of labels is present in the delimited string 

**/

inline bool SLabel::IsIdxLabelListed(const std::vector<std::string>& label, unsigned idx, const char* ls, char delim )
{
    assert( idx < label.size() ); 
    if(ls == nullptr) return false ; 

    std::vector<unsigned> indices ; 
    SLabel lab(label); 
    lab.findIndicesWithListedLabels( indices, ls, delim); 

    bool found = std::find( indices.begin(), indices.end(), idx) != indices.end() ; 
    return found ; 
}




SLabel* SLabel::Load(const char* path_)
{
    const char* path = spath::Resolve(path_); 
    if(path == nullptr) 
    {
        std::cerr 
            << "SLabel::Load FAILED to spath::Resolve [" 
            << ( path_ ? path_ : "-" ) 
            << std::endl
            ; 
        return nullptr ; 
    }

    typedef std::vector<std::string> VS ; 
    VS* label = new VS ; 

    std::ifstream ifs(path);
    std::string line;
    while(std::getline(ifs, line)) label->push_back(line) ; 

    SLabel* id = new SLabel(*label) ; 
    return id ;
}

inline SLabel* SLabel::GEOMLoad(){ return Load(GEOMLoadPath); }


inline SLabel::SLabel( const std::vector<std::string>& label_ )
    :
    label(label_)
{
}


inline int SLabel::FindIdxWithLabel(const std::vector<std::string>& label, const char* q_mml)
{
    SLabel lab(label); 
    return lab.findIdxWithLabel(q_mml); 
}



/**
SLabel::findIdxWithLabel
--------------------------------

Returns the 0-based index of CSGSolid aka the mmlabel, for example::

    [blyth@localhost ~]$ GEOM cf
    cd /home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry

    [blyth@localhost CSGFoundry]$ cat mmlabel.txt 
    2923:sWorld
    5:PMT_3inch_pmt_solid
    9:NNVTMCPPMTsMask_virtual
    12:HamamatsuR12860sMask_virtual
    6:mask_PMT_20inch_vetosMask_virtual
    1:sStrutBallhead
    1:base_steel
    1:uni_acrylic1
    130:sPanel
    [blyth@localhost CSGFoundry]$ 

**/

inline int SLabel::findIdxWithLabel(const char* q_mml) const
{
    int idx = -1 ; 
    for(int i=0 ; i < int(label.size()) ; i++) 
    {    
        const char* mml = label[i].c_str(); 
        if(strcmp(q_mml, mml) == 0 )
        {
            idx = i ;  
            break ;    
        }
    }    
    return idx ; 
}

/**
SLabel::findIndicesWithListedLabels
------------------------------------

Populates *indices* vector with 0-based idx of the delimited sub-strings from *ls*
that are present in the label vector. 

**/

inline void SLabel::findIndicesWithListedLabels( std::vector<unsigned>& indices, const char* ls, char delim )
{
    std::vector<std::string> elem ; 
    sstr::Split( ls, delim, elem ); 
    for(unsigned i=0 ; i < label.size() ; i++)
    {   
         const char* l = label[i].c_str() ;
         for(unsigned j=0 ; j < elem.size() ; j++)
         {
             const char* e = elem[j].c_str(); 
             if(strcmp(l,e) == 0) indices.push_back(i) ;  
         }
    }
}


inline std::string SLabel::detail() const 
{
    std::stringstream ss ; 
    ss << "SLabel::detail num_name " << label.size() << std::endl ;
    for(unsigned i=0 ; i < label.size() ; i++) ss << label[i] << std::endl ;  
    std::string str = ss.str(); 
    return str ; 
}


