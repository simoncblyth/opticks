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

*/ 

template <typename T>
struct NPY_API  NOpenMeshZipper
{
    NOpenMeshZipper(
                    const NOpenMesh<T>& lhs, 
                    const NOpenMesh<T>& rhs
                   );

    void init();

    const NOpenMesh<T>&  lhs ; 
    const NOpenMesh<T>&  rhs ; 
}; 




