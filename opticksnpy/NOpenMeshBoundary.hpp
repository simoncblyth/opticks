#pragma once

#include "NOpenMeshType.hpp"

template <typename T>
struct NPY_API  NOpenMeshBoundary
{
     static void CollectLoop( const T* mesh, typename T::HalfedgeHandle start, std::vector<typename T::HalfedgeHandle>& loop);

     NOpenMeshBoundary( const T* mesh, typename T::HalfedgeHandle start );

     bool contains(const typename T::HalfedgeHandle heh);

     std::vector<typename T::HalfedgeHandle> loop ; 
};
 



