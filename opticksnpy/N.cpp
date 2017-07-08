#include "N.hpp"

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"
#include "NSDF.hpp"
#include "NNode.hpp"
#include "NNodePoints.hpp"
#include "NCSG.hpp"


N::N(nnode* node, const nmat4triple* transform, const NSceneConfig* config, float surface_epsilon ) 
    : 
         node(node), 
         transform(transform),
         config(config),
         points(new NNodePoints(node, config, surface_epsilon)),
         nsdf(node->sdf(), transform->v)
{

         tots = points->collect_surface_points() ;
         const std::vector<glm::vec3>& model = points->getCompositePoints();

         transform->apply_transform_t( local, model );

         num = model.size();
         assert( local.size() == num );

} 

glm::uvec4 N::classify(const std::vector<glm::vec3>& qq, float epsilon, unsigned expect, bool dump )
{
      nsdf.classify(qq, epsilon, expect, dump);
      return nsdf.tot ; 
}

std::string N::desc() const 
{
      std::stringstream ss ;  
      ss
         << ( node->label ? node->label : "-" )
         << " nsdf: " << nsdf.desc() 
         << nsdf.detail()
         ; 
     return ss.str();
}




void N::dump_points(const char* msg)
{
     std::cout << " local points are model points transformed with transform->t (the placing transform) " << std::endl ;  
     std::cout << msg 
               << std::endl 
               << gpresent( "t", transform->t )
               << std::endl 
               ;

      // NB the length of local and model will not typically be the same as the queried points
      for(unsigned i=0 ; i < num ; i++) 
             std::cout 
               << " model " << gpresent( model[i] ) 
               << " local " << gpresent( local[i] ) 
               << std::endl 
               ;

}


