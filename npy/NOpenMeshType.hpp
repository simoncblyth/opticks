#pragma once

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

// without this get assert regarding status property on delete_face, see omc-   
#pragma GCC visibility push(default)

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

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
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeTraits.hh>

#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeT.hh>
#include <OpenMesh/Tools/Subdivider/Adaptive/Composite/RulesT.hh>



#ifdef __clang__

#pragma clang diagnostic pop
#pragma GCC visibility pop

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)

#pragma warning( pop )

#endif



#include "NPY_API_EXPORT.hh"


/*
// status traits needed for deleting things

struct NPY_API NOpenMeshTraits : public OpenMesh::DefaultTraits
{
    FaceAttributes( OpenMesh::Attributes::Status );        
    VertexAttributes( OpenMesh::Attributes::Status );           
    EdgeAttributes( OpenMesh::Attributes::Status );           
    HalfedgeAttributes( OpenMesh::Attributes::Status );           
};
*/

// boatload of traits needed for composite subdiv
//    OpenMesh/Tools/Subdivider/Adaptive/Composite/CompositeTraits.hh

struct NPY_API NOpenMeshTraits : public OpenMesh::Subdivider::Adaptive::CompositeTraits
{
    EdgeAttributes( OpenMesh::Attributes::Status );           
    HalfedgeAttributes( OpenMesh::Attributes::Status );           
};



typedef OpenMesh::TriMesh_ArrayKernelT<NOpenMeshTraits>  NOpenMeshType ;


