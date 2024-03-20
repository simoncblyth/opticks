#pragma once
/**
SCUDA_Mesh.h
=============

Try following the pattern of SGLFW_Mesh.h 

**/

#include "SMesh.h"
#include "SCU.h"

struct SCUDA_Mesh
{
    static constexpr const int INST_ELEM = 4*4 ; 

    const SMesh*   mesh ;  
    int           inst_num ; 
    const float*  inst_values ; 

    SCU_Buf<float> vtx = {} ;
    SCU_Buf<int>   idx = {} ;  
    SCU_Buf<float> ins = {} ; 

    SCUDA_Mesh(const SMesh* mesh ) ; 
    void init(); 
 
    void set_inst(const NP* _inst );
    void set_inst(int _inst_num, const float* _inst_values );
    bool has_inst() const ;

    std::string desc() const ; 
    std::string descMembers() const ; 
    std::string descInst() const ; 
}; 

inline SCUDA_Mesh::SCUDA_Mesh(const SMesh* _mesh )
    :
    mesh(_mesh),
    inst_num(0),    
    inst_values(nullptr)  
{
    init(); 
}

inline void SCUDA_Mesh::init()
{
    vtx = SCU::UploadBuf<float>( mesh->vtx->cvalues<float>(),   mesh->vtx->num_values(), "vtx" );  
    idx = SCU::UploadBuf<int>(   mesh->tri->cvalues<int>(),     mesh->tri->num_values(), "idx" );
}

inline void SCUDA_Mesh::set_inst(const NP* _inst )
{
    if(_inst == nullptr) return ; 
    assert( _inst->uifc == 'f' ); 
    assert( _inst->ebyte == 4 ); 
    assert( _inst->has_shape(-1,4,4)); 
    set_inst( _inst->num_items(), _inst->cvalues<float>() ); 
}

inline void SCUDA_Mesh::set_inst(int _inst_num, const float* _inst_values )
{
    inst_num = _inst_num ; 
    inst_values = _inst_values ; 
    ins = SCU::UploadBuf<float>( inst_values, inst_num*INST_ELEM, "ins" ); 
}

inline bool SCUDA_Mesh::has_inst() const
{
    return inst_num > 0 && inst_values != nullptr ; 
}


inline std::string SCUDA_Mesh::desc() const
{
    std::stringstream ss ; 
    ss << descMembers() ; 
    ss << descInst() ; 
    std::string str = ss.str() ; 
    return str ; 
}
inline std::string SCUDA_Mesh::descMembers() const
{
    std::stringstream ss ; 
    ss << "[ SCUDA_Mesh::descMembers " << std::endl ; 
    ss 
        << vtx.desc() 
        << idx.desc() 
        << ins.desc()
        << std::endl
        ;
    ss << "] SCUDA_Mesh::descMembers " << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}
inline std::string SCUDA_Mesh::descInst() const
{
    int edge_items = 10 ; 
    std::stringstream ss ; 
    ss << "[ SCUDA_Mesh::descInst inst_num " << inst_num << std::endl ; 
    ss << stra<float>::DescItems( inst_values, 16, inst_num, edge_items ); 
    ss << "] SCUDA_Mesh::descInst inst_num " << inst_num << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

