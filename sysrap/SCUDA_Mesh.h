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

    SCU_Buf<float> vtx = {} ;
    SCU_Buf<int>   idx = {} ;  
    SCU_Buf<float> ins = {} ; 

    SCUDA_Mesh(const SMesh* mesh ) ; 

    CUdeviceptr vtx_pointer(int item=0) const ; 
    CUdeviceptr idx_pointer(int item=0) const ; 
    size_t vtx_num(int item=0) const ;
    size_t idx_num(int item=0) const ;
 
    void free();   // on device buffers 
 
    void set_inst(const NP* _inst );
    void set_inst(int _inst_num, const float* _inst_values );
    bool has_inst() const ;

    std::string desc() const ; 
}; 

inline SCUDA_Mesh::SCUDA_Mesh(const SMesh* mesh )
    :
    vtx(SCU::UploadBuf<float>( mesh->vtx->cvalues<float>(),   mesh->vtx->num_values(), "vtx" )).  
    idx(SCU::UploadBuf<int>(   mesh->tri->cvalues<int>(),     mesh->tri->num_values(), "idx" ))
{
}

CUdeviceptr SCUDA_Mesh::vtx_pointer(int item) const 
{   
    assert( item == 0 ); 
    return vtx.pointer(); 
} 

size_t SCUDA_Mesh::vtx_num(int item) const 
{   
    assert( item == 0 ); 
    return vtx.num_item(); 
} 

CUdeviceptr SCUDA_Mesh::idx_pointer(int item) const 
{   
    assert( item == 0 ); 
    return idx.pointer(); 
} 
size_t SCUDA_Mesh::idx_num(int item) const 
{   
    assert( item == 0 ); 
    return idx.num_item(); 
} 





inline void SCUDA_Mesh::free()
{
    vtx.free(); 
    idx.free();
    ins.free(); 
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
    ins = SCU::UploadBuf<float>( _inst_values, _inst_num*INST_ELEM, "ins" ); 
}

inline bool SCUDA_Mesh::has_inst() const
{
    return ins.ptr != nullptr ; 
}

inline std::string SCUDA_Mesh::desc() const
{
    std::stringstream ss ; 
    ss << "[ SCUDA_Mesh::desc" << std::endl ; 
    ss 
        << vtx.desc() 
        << std::endl
        << idx.desc() 
        << std::endl
        << ins.desc()
        << std::endl
        ;
    ss << "] SCUDA_Mesh::desc" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}


