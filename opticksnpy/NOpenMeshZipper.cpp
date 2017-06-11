#include "PLOG.hh"


#include "NOpenMeshBoundary.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMesh.hpp"
#include "NOpenMeshZipper.hpp"

#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"



template <typename T>
NOpenMeshZipper<T>::NOpenMeshZipper(
          const NOpenMesh<T>& lhs, 
          const NOpenMesh<T>& rhs
         )
   :
   lhs(lhs), 
   rhs(rhs)
{
    init();
}

template <typename T>
void NOpenMeshZipper<T>::init()
{
    LOG(info) << "NOpenMeshZipper::init"
              << " lhs " << lhs.brief()
              << " rhs " << rhs.brief()
              ;
}



template struct NOpenMeshZipper<NOpenMeshType> ;

