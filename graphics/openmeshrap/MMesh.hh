#pragma once


#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#ifdef __clang__
#pragma clang diagnostic pop
#endif


// https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00020.html
// https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00058.html

struct MyTraits : public OpenMesh::DefaultTraits
{
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits>  MMesh;



