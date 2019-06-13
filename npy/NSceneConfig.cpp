
#include <cmath>

#include "SSys.hh"
#include "BConfig.hh"
#include "PLOG.hh"
#include "NSceneConfig.hpp"

const char* NSceneConfig::CSG_BBOX_ANALYTIC_ = "CSG_BBOX_ANALYTIC" ; 
const char* NSceneConfig::CSG_BBOX_POLY_     = "CSG_BBOX_POLY" ; 
const char* NSceneConfig::CSG_BBOX_PARSURF_  = "CSG_BBOX_PARSURF" ; 
const char* NSceneConfig::CSG_BBOX_G4POLY_  = "CSG_BBOX_G4POLY" ; 

const char* NSceneConfig::BBoxType( NSceneConfigBBoxType bbty )
{
    const char* s = NULL ; 
    switch( bbty )
    {
       case CSG_BBOX_ANALYTIC: s = CSG_BBOX_ANALYTIC_ ;break; 
       case CSG_BBOX_POLY    : s = CSG_BBOX_POLY_     ;break; 
       case CSG_BBOX_PARSURF : s = CSG_BBOX_PARSURF_  ;break; 
       case CSG_BBOX_G4POLY  : s = CSG_BBOX_G4POLY_   ;break; 
    }
    return s ; 
}


NSceneConfig::NSceneConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),

    check_surf_containment(0),
    check_aabb_containment(0),
    disable_instancing(0),
    csg_bbox_analytic(0),
    csg_bbox_poly(0),
    csg_bbox_parsurf(0),
    csg_bbox_g4poly(0),

    parsurf_epsilon(-5),
    parsurf_target(200),
    parsurf_level(2),
    parsurf_margin(0),
    verbosity(0),
    polygonize(0),
    instance_repeat_min(100),
    instance_vertex_min(0),

    default_csg_bbty(CSG_BBOX_PARSURF)
{
    LOG(debug) 
        << " cfg [" << ( cfg ? cfg : "NULL" ) << "]"
        ;

    bconfig->addInt("check_surf_containment", &check_surf_containment );
    bconfig->addInt("check_aabb_containment", &check_aabb_containment );
    bconfig->addInt("disable_instancing",     &disable_instancing );
    bconfig->addInt("csg_bbox_analytic",      &csg_bbox_analytic);
    bconfig->addInt("csg_bbox_poly",          &csg_bbox_poly);
    bconfig->addInt("csg_bbox_parsurf",       &csg_bbox_parsurf);
    bconfig->addInt("csg_bbox_g4poly",        &csg_bbox_g4poly);
    bconfig->addInt("parsurf_epsilon",        &parsurf_epsilon);
    bconfig->addInt("parsurf_target",         &parsurf_target);
    bconfig->addInt("parsurf_level",          &parsurf_level);
    bconfig->addInt("parsurf_margin",         &parsurf_margin);
    bconfig->addInt("verbosity",              &verbosity);
    bconfig->addInt("polygonize",             &polygonize);

    bconfig->addInt("instance_repeat_min",    &instance_repeat_min);
    bconfig->addInt("instance_vertex_min",    &instance_vertex_min);

    bconfig->parse();
    env_override(); 
}


void NSceneConfig::env_override()
{
    int env_verbosity = SSys::getenvint("VERBOSITY", 0) ;
    if(verbosity != env_verbosity) 
    {
        LOG(info) 
            << " VERBOSITY envvar override " 
            << " env_verbosity " << env_verbosity 
            << " verbosity " << verbosity 
            ;   
        verbosity = env_verbosity ; 
    }
}



void NSceneConfig::dump(const char* msg) const
{
    bconfig->dump(msg);
    LOG(info) << "bbox_type_string : " << bbox_type_string() ; 
}

const char* NSceneConfig::bbox_type_string() const 
{
    NSceneConfigBBoxType bbty = bbox_type() ;
    return BBoxType(bbty);
}

float NSceneConfig::get_parsurf_epsilon() const
{
    return std::pow(10, parsurf_epsilon );

}



NSceneConfigBBoxType NSceneConfig::bbox_type() const 
{
    NSceneConfigBBoxType bbty = default_csg_bbty ;

    int csg_bbox_sum = !!(csg_bbox_analytic > 0) + !!(csg_bbox_poly > 0)  + !!(csg_bbox_parsurf > 0) + !!(csg_bbox_g4poly > 0) ; 
    
    if( csg_bbox_sum == 0  )
    {
        LOG(debug) << "no csg_bbox_ specified using default " ;
    }
    else if( csg_bbox_sum > 1  )
    {
        LOG(warning) << "multiple csg_bbox_ specified using default " ;
    }
    else if( csg_bbox_sum == 1)
    {
        if(      csg_bbox_analytic > 0) bbty = CSG_BBOX_ANALYTIC ;
        else if( csg_bbox_poly    > 0)  bbty = CSG_BBOX_POLY ;
        else if( csg_bbox_parsurf > 0)  bbty = CSG_BBOX_PARSURF ;
        else if( csg_bbox_g4poly  > 0)  bbty = CSG_BBOX_G4POLY ;  // only available from GScene level 
    }
    return bbty ; 
}


