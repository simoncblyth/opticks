#pragma once

/**
SCUDA_MeshGroup.h
=============

Try following the pattern of SGLFW_Mesh.h 

**/

#include "SMeshGroup.h"
#include "SCU_BufferView.h"

struct SCUDA_MeshGroup
{
    static constexpr const int INST_ELEM = 4*4 ; 

    std::vector<const NP*> _vtx ; 
    std::vector<const NP*> _idx ; 

    SCU_BufferView<float> vtx = {} ;
    SCU_BufferView<int>   idx = {} ;  
    SCU_BufferView<float> ins = {} ; 

    size_t num_part() const ; 
    void free(); 
    std::string desc() const ; 

    SCUDA_MeshGroup(const SMeshGroup* mg ) ; 

}; 

inline size_t SCUDA_MeshGroup::num_part() const 
{
    return vtx.num_part(); 
} 

inline void SCUDA_MeshGroup::free()
{
    vtx.free(); 
    idx.free();
    ins.free(); 
}

inline std::string SCUDA_MeshGroup::desc() const
{
    std::stringstream ss ; 
    ss << "[SCUDA_MeshGroup::desc\n" ;
    ss << vtx.desc() ;  
    ss << idx.desc() ;  
    ss << ins.desc() ;  
    ss << "]SCUDA_MeshGroup::desc\n" ;
    std::string str = ss.str(); 
    return str ; 
}


inline SCUDA_MeshGroup::SCUDA_MeshGroup(const SMeshGroup* mg )
{
    int num_sub = mg->subs.size() ; 
    for(int i=0 ; i < num_sub ; i++)
    {
        const SMesh* m = mg->subs[i] ;  
        _vtx.push_back(m->vtx);          
        _idx.push_back(m->tri);   // TODO: change tri to idx in SMesh           
    }

    vtx.upload(_vtx); 
    idx.upload(_idx); 

    //std::cout << "SCUDA_MeshGroup::SCUDA_MeshGroup\n" << desc() ; 

}


