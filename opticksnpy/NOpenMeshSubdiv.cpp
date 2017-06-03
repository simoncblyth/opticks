#include <sstream>

#include "NOpenMeshSubdiv.hpp"
#include "PLOG.hh"

using namespace OpenMesh::Subdivider;

template <typename T>
NOpenMeshSubdiv<T>::NOpenMeshSubdiv( T& mesh ) 
    : 
    mesh(mesh),
    subdivider(new subdivider_t(mesh)) 
{
    init();
} 

template <typename T>
void NOpenMeshSubdiv<T>::init()
{
   typedef typename Adaptive::RuleInterfaceT<T>  Rule ; 

   typedef typename Adaptive::Tvv3<T> Tvv3_r ; 
   typedef typename Adaptive::VF<T>   VF_r ; 
   typedef typename Adaptive::FF<T>   FF_r ; 
   typedef typename Adaptive::FVc<T>  FVc_r ; 

   // http://www.multires.caltech.edu/pubs/sqrt3.pdf

   subdivider->template add<Tvv3_r>();
   subdivider->template add<VF_r>();
   subdivider->template add<FF_r>();
   subdivider->template add<FVc_r>();

   LOG(info) << "NOpenMeshSubdiv<T>::init()"
             << " desc " << desc()
             ;

   assert(subdivider->initialize()); 
}

template <typename T>
std::string NOpenMeshSubdiv<T>::desc()
{
    std::stringstream ss ; 

    ss << subdivider->rules_as_string() ;

    return ss.str();
}
 
template <typename T>
void NOpenMeshSubdiv<T>::refine(typename T::FaceHandle fh)
{
    subdivider->refine(fh); 
}


template struct NOpenMeshSubdiv<NOpenMeshType> ;

