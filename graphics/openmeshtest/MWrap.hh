#pragma once

template <typename MeshT>
class MWrap { 
   public:
       MWrap(MeshT* mesh);
       MeshT* getMesh();
   public:
      int labelConnectedComponents();

   private:
       MeshT* m_mesh ; 

};


template <typename MeshT>
inline MWrap<MeshT>::MWrap(MeshT* mesh) : m_mesh(mesh) 
{
}

template <typename MeshT>
inline MeshT* MWrap<MeshT>::getMesh()
{
    return m_mesh ; 
}



