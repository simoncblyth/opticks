#include "NOct.hpp"

#include "NGrid3.hpp"
#include "NField3.hpp"
#include "NFieldGrid3.hpp"


template <typename T>
int NOct::Corners( const T& ijk, NFieldGrid3* fg, const nvec4& /*ce*/, const int /*ncorner*/, const int /*size*/ )
{
    nvec3 fpos = fg->grid->fpos(ijk); 
    int corners = fg->field->zcorners(fpos, fg->grid->elem ) ;  
    return corners ; 
}



template int NOct::Corners<nivec3>( const nivec3&, NFieldGrid3*, const nvec4&, const int, const int) ;
template int NOct::Corners<glm::ivec3>( const glm::ivec3&, NFieldGrid3*, const nvec4&, const int, const int) ;


//void NOct::PopulateLeaf(int corners, NOct* leaf, NFieldGrid3* fg, const nvec4& ce )
void NOct::PopulateLeaf(int , NOct* leaf, NFieldGrid3* , const nvec4& )
{
    leaf->type = NOct::Node_Leaf;
}


void NOct::GenerateVertexIndices(NOct* /*node*/, std::vector<glm::vec3>& /*vertices*/,std::vector<glm::vec3>& /*normals*/ , const nbbox& /*bb*/, const nvec4& /*ce*/, NFieldGrid3* /*fg*/)
{
}

void NOct::ContourCellProc(NOct* /*node*/, std::vector<int>& /*indexBuffer*/)
{
}

NOct* NOct::ConstructOctreeNodes(NOct* node, NFieldGrid3* /*fg*/, const nvec4& /*ce*/, int& /*count*/)
{
   return node ; 
}
NOct* NOct::SimplifyOctree(NOct* node, float /*threshold*/)
{
   return node ; 
}




