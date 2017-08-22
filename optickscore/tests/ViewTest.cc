#include <iomanip>
#include <cassert>


#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include "View.hh"


#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    View v ; 
    //v.setEye(1,0,0) ;

    v.Summary();

    //glm::mat4 m2w ; 

    
    
    return 0 ;
}

/*

View::Summary
            eye vec4      -1.000     -1.000      0.000      1.000 
           look vec4       0.000      0.000      0.000      1.000 
             up vec4       0.000      0.000      1.000      0.000 


*/


