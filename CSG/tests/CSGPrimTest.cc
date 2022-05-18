/**
CSGPrimTest.cc
=================

NB to control which geometry is loaded invoke this executable via the CSGPrimTest.sh script::

    epsilon:CSG blyth$ ./CSGPrimTest.sh remote | grep bnd:31
      pri:3090     lpr:1   gas:1 msh:120  bnd:31   nno:1 nod:23210 ce (      0.00,      0.00,      4.06,     24.00) meshName PMT_3inch_body_solid_ell_ell_helper bndName   Water///Pyrex
      pri:3097     lpr:3   gas:2 msh:116  bnd:31   nno:1 nod:23243 ce (      0.00,      0.00,      5.39,    184.00) meshName NNVTMCPPMT_PMT_20inch_pmt_solid_head bndName   Water///Pyrex
      pri:3104     lpr:3   gas:3 msh:109  bnd:31  nno:15 nod:23276 ce (      0.00,      0.00,      8.39,    254.00) meshName HamamatsuR12860_PMT_20inch_pmt_solid_1_4 bndName   Water///Pyrex
      pri:3109     lpr:2   gas:4 msh:133  bnd:31   nno:3 nod:23307 ce (      0.00,      0.00,     87.00,    254.00) meshName PMT_20inch_veto_pmt_solid_1_2 bndName   Water///Pyrex
    epsilon:CSG blyth$ 
    epsilon:CSG blyth$ 



**/

#include "SSys.hh"
#include "scuda.h"
#include "CSGFoundry.h"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::Load(); 

    LOG(info) << "fd.desc" << fd->desc() ; 
    LOG(info) << "fd.summary" ; 
    fd->summary(); 

    LOG(info) << "fd.detailPrim" << std::endl << fd->detailPrim() ; 

    return 0 ; 
}

