// TEST=EllipseTest om-t:34

#include "Ellipse.hpp"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
     OPTICKS_LOG(argc, argv);    

     ellipse e(100., 100. ) ; 

     std::vector<glm::dvec2> points = {
         glm::dvec2(200,   0), 
         glm::dvec2(200, 200),
         glm::dvec2(0  , 200),
         glm::dvec2(-200,200),
         glm::dvec2(-200,-200),
         glm::dvec2(0,   -200),
         glm::dvec2(200, -200)
     };

     glm::dvec2 o(0,0); 

     int w(40) ; 

     for(unsigned i=0 ; i < points.size() ; i++)
     {
         const glm::dvec2& pt = points[i] ;  
         glm::dvec2 ep = e.closest_approach_to_point( pt ); 

         double epo = glm::distance( ep, o );     
         LOG(info) 
              << " i " << i 
              << " pt " << std::setw(w) << glm::to_string( pt ) 
              << " ep " << std::setw(w) << glm::to_string( ep )
              << " epo " << epo
              ; 
     }

     return 0 ; 
}



