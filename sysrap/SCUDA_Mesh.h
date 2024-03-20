#pragma once
/**
SCUDA_Mesh.h
=============

Try following the pattern of SGLFW_Mesh.h 

**/

struct NP ; 
struct SMesh ; 

struct SCUDA_Mesh
{
    const SMesh*   mesh ;  

    int           inst_num ; 
    const float*  inst_values ; 

    SCUDA_Mesh(const SMesh* mesh ) ; 
    void init(); 
 
    void set_inst(const NP* _inst );
    void set_inst(int _inst_num, const float* _inst_values );
    bool has_inst() const ;

    std::string descInst() const ; 
    std::string desc() const ; 
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
    // upload vtx tri 

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

    int itemsize = 4*4*sizeof(float) ; 
    int num_bytes = inst_num*itemsize ;

    ins = new SCUDA_Buffer( num_bytes, inst_values ); 
    ins->bind();
    ins->upload(); 


}

inline bool SCUDA_Mesh::has_inst() const
{
    return inst_num > 0 && inst_values != nullptr ; 
}


inline std::string SCUDA_Mesh::desc() const
{
    std::stringstream ss ; 
    ss << descInst() ; 
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

