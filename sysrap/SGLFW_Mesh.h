#pragma once
/**
SGLFW_Mesh.h : create OpenGL buffers with SMesh and instance data and render
==============================================================================

Canonical use is from SGLFW_Scene::initMesh

* (in a former incarnation this was called SGLFW_Render)

**/


struct NP ; 
struct SMesh ; 
struct SGLFW_Program ; 


struct SGLFW_Mesh
{
    bool  dump ; 
    const SMesh*   mesh ;  

    SGLFW_Buffer*  vtx ; 
    SGLFW_Buffer*  nrm ;
    SGLFW_Buffer*  ins ;

    SGLFW_VAO*     vao ;  
    SGLFW_Buffer*  idx ; 

    int           inst_num ; 
    const float*  inst_values ; 
    int           render_count ; 

    SGLFW_Mesh(const SMesh* mesh ) ; 
    void init(); 

    void set_inst(const NP* _inst );
    void set_inst(int _inst_num, const float* _inst_values );
    bool has_inst() const ;

    std::string descInst() const ; 
    std::string desc() const ; 

    void render(const SGLFW_Program* prog); 
    void render_drawElements() const ; 
};

inline SGLFW_Mesh::SGLFW_Mesh(const SMesh* _mesh )
    :
    dump(false),
    mesh(_mesh),
    vtx(nullptr),
    nrm(nullptr),
    ins(nullptr),
    vao(nullptr),
    idx(nullptr),
    inst_num(0),    
    inst_values(nullptr),    
    render_count(0)
{
    init(); 
}

/**
SGLFW_Mesh::init
----------------

Creates vtx, nrm, idx OpenGL buffers using SGLFW_Buffers


**/


inline void SGLFW_Mesh::init()
{
    vtx = new SGLFW_Buffer( mesh->vtx->arr_bytes(), mesh->vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    vtx->bind();
    vtx->upload(); 

    nrm = new SGLFW_Buffer( mesh->nrm->arr_bytes(), mesh->nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    nrm->bind();
    nrm->upload(); 

    vao = new SGLFW_VAO ;  // vao: establishes context for OpenGL attrib state and element array (not GL_ARRAY_BUFFER)
    vao->bind(); 

    idx = new SGLFW_Buffer( mesh->tri->arr_bytes(), mesh->tri->cvalues<int>()  , GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    idx->bind();
    idx->upload(); 
}

inline void SGLFW_Mesh::set_inst(const NP* _inst )
{
    if(_inst == nullptr) return ; 
    assert( _inst->uifc == 'f' ); 
    assert( _inst->ebyte == 4 ); 
    assert( _inst->has_shape(-1,4,4)); 
    set_inst( _inst->num_items(), _inst->cvalues<float>() ); 
}

inline void SGLFW_Mesh::set_inst(int _inst_num, const float* _inst_values )
{
    inst_num = _inst_num ; 
    inst_values = _inst_values ; 

    int itemsize = 4*4*sizeof(float) ; 
    int num_bytes = inst_num*itemsize ;
    ins = new SGLFW_Buffer( num_bytes, inst_values, GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    ins->bind();
    ins->upload(); 
}

inline bool SGLFW_Mesh::has_inst() const
{
    return inst_num > 0 && inst_values != nullptr ; 
}


inline std::string SGLFW_Mesh::desc() const
{
    std::stringstream ss ; 
    ss << descInst() ; 
    std::string str = ss.str() ; 
    return str ; 
}
inline std::string SGLFW_Mesh::descInst() const
{
    int edge_items = 10 ; 
    std::stringstream ss ; 
    ss << "[SGLFW_Mesh::descInst inst_num " << inst_num << std::endl ; 
    ss << stra<float>::DescItems( inst_values, 16, inst_num, edge_items ); 
    ss << "]SGLFW_Mesh::descInst inst_num " << inst_num << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

/**
SGLFW_Mesh::render
---------------------

Use argument prog to render the mesh 

NB: careful that the intended buffer is bound (making it the active GL_ARRAY_BUFFER)
when the vertex attrib is enabled.  Getting this wrong can for example easily cause
normals to appear in position slots causing perplexing renders. 

**/

inline void SGLFW_Mesh::render(const SGLFW_Program* prog)
{
   prog->use(); 
   vao->bind(); 

   vtx->bind();
   prog->enableVertexAttribArray( prog->vtx_attname, SMesh::VTX_SPEC ); 

   nrm->bind();
   prog->enableVertexAttribArray( prog->nrm_attname, SMesh::NRM_SPEC ); 

   if(ins)
   {
       ins->bind(); 
       prog->enableVertexAttribArray_OfTransforms( prog->ins_attname ) ; 
   }

   idx->bind();
   prog->updateMVP();

   render_drawElements(); 
   render_count += 1 ; 
}

inline void SGLFW_Mesh::render_drawElements() const 
{
    GLenum mode = GL_TRIANGLES ; 
  	GLsizei count = mesh->indices_num() ;  // number of elements to render (eg 3 for 1 triangle)
  	GLenum type = GL_UNSIGNED_INT ; 
  	const void * indices = (GLvoid*)(sizeof(GLuint) * mesh->indices_offset() ) ;
  	GLsizei instancecount = ins ? inst_num : 0 ; 

    if(instancecount > 0)
    {

#ifdef __APPLE__
        glDrawElementsInstanced(mode, count, type, indices, instancecount );
        if(dump && render_count < 10 ) std::cout 
            << "SGLFW_Mesh::render_drawElements.glDrawElementsInstanced" 
            << " render_count " << render_count
            << " instancecount " << instancecount
            << std::endl
            ;

#else
        GLint basevertex = 0 ; 
        GLuint baseinstance = 0 ; 
        glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, baseinstance ); 
        // SEGV on laptop, OK on worktation 
        // https://github.com/moderngl/moderngl/issues/346
        if(dump && render_count < 10 ) std::cout 
            << "SGLFW_Mesh::render_drawElements.glDrawElementsInstancedBaseVertexBaseInstance" 
            << " render_count " << render_count
            << " instancecount " << instancecount
            << std::endl
            ;
#endif 

    }
    else
    {
        glDrawElements(mode, count, type, indices );
    }
}



