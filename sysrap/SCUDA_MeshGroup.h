#pragma once

/**
SCUDA_MeshGroup.h
=============

Try following the pattern of SGLFW_Mesh.h 

**/

#include "SMeshGroup.h"
#include "SCU.h"

struct SCUDA_MeshGroup
{
    static constexpr const int INST_ELEM = 4*4 ; 

    SCU_Buf<float> vtx = {} ;
    SCU_Buf<int>   idx = {} ;  
    SCU_Buf<float> ins = {} ; 

    SCUDA_MeshGroup(const SMeshGroup* mg ) ; 

    CUdeviceptr vtx_pointer(int item) const ; 
    CUdeviceptr idx_pointer(int item) const ; 
    size_t vtx_num(int item) const ;
    size_t idx_num(int item) const ;

    std::string desc() const ; 
}; 

inline SCUDA_MeshGroup::SCUDA_MeshGroup(const SMeshGroup* mg )
{
    int num_sub = mg->subs.size() ; 

    for(int i=0 ; i < num_sub ; i++)
    {



    }

    idx(SCU::UploadBuf<int>(   mesh->tri->cvalues<int>(),     mesh->tri->num_values(), "idx" ))


}

CUdeviceptr SCUDA_MeshGroup::vtx_pointer(int item) const 
{   
    assert( item == 0 ); 
    return vtx.pointer(); 
} 

size_t SCUDA_MeshGroup::vtx_num(int item) const 
{   
    assert( item == 0 ); 
    return vtx.num_item(); 
} 

CUdeviceptr SCUDA_MeshGroup::idx_pointer(int item) const 
{   
    assert( item == 0 ); 
    return idx.pointer(); 
} 
size_t SCUDA_MeshGroup::idx_num(int item) const 
{   
    assert( item == 0 ); 
    return idx.num_item(); 
} 




