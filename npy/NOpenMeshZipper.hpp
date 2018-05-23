#pragma once

template <typename T> struct NOpenMesh ; 

/*
NOpenMeshZipper
==================

Each side has a frontier ribbon mesh::

      +---------+---------+---------+---          outer loop (SDF other >  0)   
       \       / \       / \       / \
      . * . . * . * . . *. .*. . .* . * . . . .   analytic frontier (SDF other = 0 )       
         \   /     \   /     \   /     \
          \ /       \ /       \ /       \
      -----+---------+---------+---------+        inner loop (SDF other < 0) 


CSG Union requires zippering of

* inner loops of both sides with the frontier vertices in the middle 



*/ 

template <typename T>
struct NPY_API  NOpenMeshZipper
{
    typedef typename T::Point              P ; 

    NOpenMeshZipper(
                    const NOpenMesh<T>& lhs, 
                    const NOpenMesh<T>& rhs
                   );



    void init();

    void dump();
    void dump_boundary(int index, const NOpenMeshBoundary<T>& loop, const char* msg);


    const NOpenMesh<T>&  lhs ; 
    const NOpenMesh<T>&  rhs ; 
}; 




