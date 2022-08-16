#include <cstdlib>
#include <iostream>
#include <glm/gtx/string_cast.hpp>

#include "ssys.h"
#include "strid.h"
#include "snode.h"
#include "stree.h"

const char* BASE = getenv("BASE"); 

int main(int argc, char** argv)
{
    stree st ; 

    snode nd ; 
    nd = {} ; 
    nd.sensor_id = -1 ; 
    nd.sensor_index = -1 ; 
 
    st.nds.push_back(nd); 

    glm::tmat4x4<double> tr_m2w(1.) ;   
    glm::tmat4x4<double> tr_w2m(1.) ; 
    int gas_idx = 1 ; 
    int nidx = 0 ; 
    st.add_inst(tr_m2w, tr_w2m, gas_idx, nidx ); 

    st.narrow_inst(); 

    std::cout << st.desc_inst() << std::endl ; 
    std::cout << " tr_m2w " << strid::Desc<double, int64_t>(tr_m2w) << std::endl ; 

    const glm::tmat4x4<double>& inst_0 = st.inst[0] ;  
    std::cout << " inst_0 " << strid::Desc<double, int64_t>(inst_0) << std::endl ; 

    const glm::tmat4x4<float>& inst_f4_0 = st.inst_f4[0] ;  
    std::cout << " inst_f4_0 " << strid::Desc<float, int32_t>(inst_f4_0) << std::endl ; 


    st.save(BASE); 


    return 0 ; 
}


 
