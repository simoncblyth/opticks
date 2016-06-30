#pragma once


#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

// without this get assert regarding status property on delete_face, see omc-   
#pragma GCC visibility push(default)
#endif

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#ifdef __clang__
#pragma clang diagnostic pop

#pragma GCC visibility pop
#endif


// https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00020.html
// https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00058.html

struct MyTraits : public OpenMesh::DefaultTraits
{
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits>  MMesh;



