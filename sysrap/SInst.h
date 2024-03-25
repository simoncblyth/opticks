#pragma once

#include "NP.hh"
#include "stra.h"

struct SInst
{
    static std::string Desc( const NP* _inst, int edge=10 ); 
    const NP* inst ; 
    SInst( const NP* _inst ); 
    std::string desc(int edge=10) const ; 

};

inline std::string SInst::Desc( const NP* _inst, int edge )
{
    SInst si(_inst) ; 
    return si.desc(edge) ; 
}

inline SInst::SInst( const NP* _inst ) 
    :
    inst(_inst)
{
    assert( inst->uifc == 'f' );  
    assert( inst->ebyte == 4 || inst->ebyte == 8 );  
    assert( inst->has_shape(-1,4,4)); 
}

inline std::string SInst::desc(int edge) const
{
    std::stringstream ss ;

    int inst_num = inst->num_items() ; 
    ss << "[ SInst::desc inst_num " << inst_num << std::endl ;

    if( inst->ebyte == 4 )
    {
        const float* inst_values = inst->cvalues<float>() ; 
        ss << stra<float>::DescItems( inst_values, 16, inst_num, edge );
    }
    else if( inst->ebyte == 8 )
    {
        const double* inst_values = inst->cvalues<double>() ; 
        ss << stra<double>::DescItems( inst_values, 16, inst_num, edge );
    }

    ss << "] SInst::desc inst_num " << inst_num << std::endl ;
    std::string str = ss.str() ;
    return str ;
}


