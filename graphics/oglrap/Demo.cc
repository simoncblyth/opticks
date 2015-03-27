#include "Demo.hh"
#include "stdio.h"
#include "assert.h"

#include "GMesh.hh"

/*
 
           y ^
             |
             R
             | 
             |
      -------|--------> x
             |
             |
        B    |    G 
 

*/



const float Demo::pvertex[] = {
   0.0f,  1.0f,  0.0f,
   1.0f, -1.0f,  0.0f,
  -1.0f, -1.0f,  0.0f
};

const gfloat3 Demo::gvertex[] = {
   {0.0f,  1.0f,  0.0f},
   {1.0f, -1.0f,  0.0f},
   {-1.0f, -1.0f,  0.0f}
};


const float Demo::pcolor[] = {
  1.0f, 0.0f,  0.0f,
  0.0f, 1.0f,  0.0f,
  0.0f, 0.0f,  1.0f
};
const gfloat3 Demo::gcolor[] = {
  {1.0f, 0.0f,  0.0f},
  {0.0f, 1.0f,  0.0f},
  {0.0f, 0.0f,  1.0f}
};


const unsigned int Demo::pindex[] = {
      0,  1,  2
};
const guint3 Demo::gindex[] = {
      { 0,  1,  2 }
};




const float Demo::pmatrix[] = {
  1.0f, 0.0f,  0.0f, 0.0f,
  0.0f, 1.0f,  0.0f, 0.0f,
  0.0f, 0.0f,  1.0f, 0.0f,  
  0.0f, 0.0f,  0.0f, 1.0f  
};




Demo::Demo() :
    GDrawable(),
    m_vertices(NULL),
    m_colors(NULL),
    m_indices(NULL),
    m_model_to_world(NULL)
{
    m_vertices = new GArray<float>(9, &pvertex[0]);
    m_colors   = new GArray<float>(9, &pcolor[0]);
    m_indices  = new GArray<unsigned int>(3,  &pindex[0]);
    m_model_to_world = new GArray<float>(16,  &pmatrix[0]);
}

Demo::~Demo()
{
}

unsigned int Demo::getNumElements()
{
    return m_indices ? m_indices->getLength() : 0 ; 
}


GMesh* Demo::asMesh()
{
    return NULL ;
}




GBuffer* Demo::getVertices()
{
    return m_vertices ; 
}
GBuffer* Demo::getColors()
{
    return m_colors ; 
}
GBuffer* Demo::getIndices()
{
    return m_indices ; 
}
GBuffer* Demo::getModelToWorld()
{
    return m_model_to_world  ; 
}



