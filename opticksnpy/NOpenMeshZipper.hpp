#pragma once

template <typename T> struct NOpenMeshBoundary ; 

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
                    const NOpenMeshBoundary<T>& lhs, 
                    const NOpenMeshBoundary<T>& rhs
                   );

    void init();

    const NOpenMeshBoundary<T>&  lhs ; 
    const NOpenMeshBoundary<T>&  rhs ; 
}; 




