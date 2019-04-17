#pragma once

#include <string>
#include "NPY_API_EXPORT.hh"
struct nnode ; 
struct ncone ; 

struct NPY_API NTreeJUNO
{
    NTreeJUNO(nnode* root_) ;
    nnode* root ; 
    ncone* cone ; 

    ncone* replacement_cone() const ; 
    void rationalize(); 


};


