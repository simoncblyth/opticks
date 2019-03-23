#pragma once

/**

MMesh
=======

Header to setup the OpenMesh Mesh class

**/




#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

// without this get assert regarding status property on delete_face, see omc-   
#pragma GCC visibility push(default)

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#elif defined(_MSC_VER)

#pragma warning( push )
// OpenMesh/Core/Mesh/AttribKernelT.hh(140): warning C4127: conditional expression is constant
#pragma warning( disable : 4127 )
// OpenMesh/Core/Utils/vector_cast.hh(94): warning C4100: '_dst': unreferenced formal parameter
#pragma warning( disable : 4100 )
// openmesh\core\utils\property.hh(156): warning C4702: unreachable code  
#pragma warning( disable : 4702 )

#endif


#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>



#ifdef __clang__

#pragma clang diagnostic pop
#pragma GCC visibility pop

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)

#pragma warning( pop )

#endif




// https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00020.html
// https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00058.html

struct MyTraits : public OpenMesh::DefaultTraits
{
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits>  MMesh;



