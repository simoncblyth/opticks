#include "NGLM.hpp"
#include "GMatrix.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
     PLOG_(argc, argv);
     GGEO_LOG_ ;  


     glm::vec3 sc(10.) ; 
     glm::vec3 tr(1.,2.,3.) ; 
    

     GMatrix<float> s(sc.x);
     s.Summary("s");     

     GMatrix<float> t(tr.x,tr.y,tr.z,sc.x);
     t.Summary("t");     

     GMatrix<float> tt((float*)t.getPointer());
     tt.Summary("tt");


     glm::mat4 scale = glm::scale(glm::mat4(1.0f), sc);
     glm::mat4 translate = glm::translate(glm::mat4(1.0f), tr);
     glm::mat4 mat = translate * scale ; 

     GMatrix<float> gg(glm::value_ptr(mat));
     gg.Summary("gg");


     return 0 ; 
}
