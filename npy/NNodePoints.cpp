#include <ios>
#include <iostream>
#include <set>
#include <map>
#include <sstream>
#include "OpticksCSG.h"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NNode.hpp"
#include "NBBox.hpp"
#include "Nuv.hpp"
#include "NSceneConfig.hpp"
#include "NNodePoints.hpp"

#include "PLOG.hh"


NNodePoints::NNodePoints(nnode* root, const NSceneConfig* config)
    :
    m_root(root),
    m_config(config),
    m_verbosity(config ? config->verbosity : root->verbosity),
    m_epsilon(config ? config->get_parsurf_epsilon() : 1e-5),
    m_level(config ? config->parsurf_level : 2 ), 
    m_margin(config ? config->parsurf_margin : 0 ), 
    m_target(config ? config->parsurf_target : 200) 
{
    init();
}

void NNodePoints::setEpsilon(float epsilon)
{
    m_epsilon = epsilon ; 
}

void NNodePoints::init()
{
    m_root->collect_prim_for_edit(m_primitives);    // recursive collection of list of all primitives in tree
}

std::string NNodePoints::desc() const 
{
    std::stringstream ss; 

    ss << "NNP"
       << " verbosity " << m_verbosity
       << " level " << m_level
       << " margin " << m_margin
       << " target " << m_target
       << " num_prim " << m_primitives.size()
       << " num_composite_points " << m_composite_points.size()
       << " epsilon " << std::scientific << m_epsilon << std::fixed 
        ;
    return ss.str();
}




nbbox NNodePoints::bbox_surface_points() const 
{
    unsigned num_cp = getNumCompositePoints() ; 
    if(num_cp==0)  
    {
        LOG(debug) << "NNodePoints::bbox_surface_points NONE FOUND : probably need larger parsurf_level   " << desc() ;  
    }

    return nbbox::from_points(getCompositePoints(), m_verbosity);    
}

const std::vector<glm::vec3>& NNodePoints::getCompositePoints() const 
{
    return m_composite_points ;
}
unsigned NNodePoints::getNumCompositePoints() const 
{
    return m_composite_points.size() ;
}
float NNodePoints::getEpsilon() const 
{
    return m_epsilon ; 
}



glm::uvec4 NNodePoints::collect_surface_points() 
{
    if(m_verbosity > 2 )
    {
    LOG(info) << "NNodePoints::collect_surface_points"
              << " verbosity " << m_verbosity 
              ;
    }

    if(m_config && m_verbosity > 2) m_config->dump("NNodePoints::collect_surface_points");

   /*

               level                       divisions   (+1 for uv points)
                 1    +-----+------+   0x1 << 1  =     2     
                 2    +--+--+--+---+   0x1 << 2  =     4 
                 3                     0x1 << 3  =     8 
                 4                     0x1 << 4  =    16
                 5                     0x1 << 5  =    32 
                 6                     0x1 << 6  =    64 
                 7                     0x1 << 7  =   128
                 8                     0x1 << 8  =   256
                 9                     0x1 << 9  =   512
                10                     0x1 << 10 =  1024

     * Divisions are then effectively squared to give uv samplings
     * margin > 0 , skips both ends 

     The below uses adaptive uv-levels, upping level from the configured
     initial level up to 8 times or until the target number of points is exceeded.
   */
  
    unsigned pointmask = POINT_SURFACE ; 

    unsigned num_composite_points = 0 ; 
    int countdown = 8 ; 
    unsigned level = m_level ; 

    glm::uvec4 tots ; 

    while( num_composite_points < m_target && countdown-- )
    {
        clear();
        tots = collectCompositePoints( level, m_margin , pointmask);

        if(m_verbosity > 2)
        std::cout 
                  << " verbosity " << m_verbosity 
                  << " countdown " << countdown  
                  << " level " << level   
                  << " target " << m_target
                  << " num_composite_points " << num_composite_points
                  << " tots (inside/surface/outside/selected) " << gpresent(tots)  
                  << std::endl ; 
 
        level++ ; 
        num_composite_points = m_composite_points.size() ;
    }

    return tots ; 
}

void NNodePoints::clear()
{
    m_composite_points.clear();
    m_composite_coords.clear();
    m_prim_bb.clear();
    m_prim_bb_selected.clear();
}


glm::uvec4 NNodePoints::collectCompositePoints( unsigned level, int margin , unsigned pointmask ) 
{
    glm::uvec4 tot(0,0,0,0);

    unsigned num_prim = m_primitives.size(); 

    for(unsigned prim_idx=0 ; prim_idx < num_prim ; prim_idx++)
    {
        nnode* prim = m_primitives[prim_idx] ; 

        
        if(m_verbosity > 4) 
        {
        LOG(info) << "NNodePoints::collectCompositePoints"
                  << " prim_idx " << prim_idx 
                  << " level " << level
                  << " margin " << margin
                  ;
        }


        prim->collectParPoints(prim_idx, level, margin, FRAME_GLOBAL , m_verbosity );

        glm::uvec4 isos = selectBySDF(prim, prim_idx, pointmask ); 

        tot += isos ;  

        if(m_verbosity > 4) 
        std::cout << "NNodePoints::getCompositePoints" 
                  << " prim " << std::setw(3) << prim_idx 
                  << " pointmask " << std::setw(20) << NNodeEnum::PointMask(pointmask)
                  << " num_inside " << std::setw(6) << isos.x
                  << " num_surface " << std::setw(6) << isos.y
                  << " num_outside " << std::setw(6) << isos.z
                  << " num_select " << std::setw(6) << isos.w 
                  << std::endl ; 
                  ;
    }
    return tot ; 
}


glm::uvec4 NNodePoints::selectBySDF(const nnode* prim, unsigned prim_idx, unsigned pointmask ) 
{
    // this is invoked from root level, so no need to pass down a verbosity 

    std::function<float(float,float,float)> _sdf = m_root->sdf() ;

    typedef std::vector<glm::vec3> VV ; 
    typedef std::vector<nuv> VC ; 
    const VV& prim_points = prim->get_par_points();
    const VC& prim_coords = prim->get_par_coords();
    
    unsigned num_prim_points = prim_points.size() ;
    unsigned num_prim_coords = prim_coords.size() ;

    unsigned num_inside(0);
    unsigned num_outside(0);
    unsigned num_surface(0);
    unsigned num_select(0);


    if(m_verbosity > 5) 
    {
         LOG(info) << "NNodePoints::selectBySDF"
                   << " verbosity " << m_verbosity
                   << " prim_points " << num_prim_points
                   << " prim_coords " << num_prim_coords
                   ;
    }

    assert( num_prim_points == num_prim_coords );

    std::vector<glm::vec3> _points ; 
    std::vector<nuv> _coords ; 

    for(unsigned i=0 ; i < num_prim_points ; i++) 
    {
          glm::vec3 p = prim_points[i] ;
          nuv      uv = prim_coords[i] ;

          assert( uv.p() == prim_idx );


          // If there is a gtransform on the node, the inverse gtransform->v is 
          // applied to the query point within the primitives operator()
          // thusly query points are treated as being in the CSG root frame.

          float sd =  _sdf(p.x, p.y, p.z) ;
          
          NNodePointType pt = NNodeEnum::PointClassify(sd, m_epsilon ); 

          if( pt & pointmask ) 
          {
              num_select++ ;  
              _points.push_back(p);
              _coords.push_back(uv);
          }

          switch(pt)
          {
              case POINT_INSIDE  : num_inside++ ; break ; 
              case POINT_SURFACE : num_surface++ ; break ; 
              case POINT_OUTSIDE : num_outside++ ; break ; 
          }

          if(m_verbosity > 5) 
          std::cout
               << " i " << std::setw(4) << i 
               << " p " << gpresent(p) 
               << " pt " << std::setw(15) << NNodeEnum::PointType(pt)
               << " sd(fx4) " << std::setw(10) << std::fixed << std::setprecision(4) << sd 
               << " sd(sci) " << std::setw(10) << std::scientific << sd 
               << " sd(def) " << std::setw(10) << std::fixed  << sd 
               << std::endl
               ; 
    }


    std::copy( _points.begin(), _points.end(), std::back_inserter(m_composite_points) );
    std::copy( _coords.begin(), _coords.end(), std::back_inserter(m_composite_coords) );

    nbbox pbb =  nbbox::from_points( prim_points, m_verbosity);    
    m_prim_bb.push_back(pbb);

    nbbox sbb =  nbbox::from_points(     _points, m_verbosity);    
    m_prim_bb_selected.push_back(sbb);

    return glm::uvec4(num_inside, num_surface, num_outside, num_select );
}




void NNodePoints::dump(const char* msg, unsigned dmax) const 
{
    unsigned num_composite_points = m_composite_points.size() ;
    LOG(info) << msg 
              << " num_composite_points " << num_composite_points
              << " dmax " << dmax
              << " desc " << desc()
              ;


    nbbox bbsp = bbox_surface_points();
    std::cout << " bbsp " << bbsp.desc() << std::endl ; 

    glm::vec3 lsp ; 
    for(unsigned i=0 ; i < std::min<unsigned>(num_composite_points,dmax) ; i++)
    {
        glm::vec3 sp = m_composite_points[i]; 
        nuv       uv = m_composite_coords[i]; 
        if(sp != lsp)
        std::cout 
            << " i " << std::setw(4) << i 
            << " sp " << gpresent( sp ) 
            << " uv " << uv.desc()
            << std::endl 
            ;

        lsp = sp ; 
    }

    dump_bb();
    dump_sheets();
}



nbbox NNodePoints::selectPointsBBox( unsigned prim, unsigned sheet ) const 
{
    std::vector<glm::vec3> points ; 
    std::vector<nuv> coords ; 
    selectPoints(points, coords, prim, sheet);
    return nbbox::from_points(points, m_verbosity );
}
 

void NNodePoints::selectPoints(std::vector<glm::vec3>& points, std::vector<nuv>& coords, unsigned prim, unsigned sheet) const 
{
    unsigned num_composite_points = m_composite_points.size() ;
    unsigned num_composite_coords = m_composite_coords.size() ;
    assert( num_composite_points == num_composite_coords );

    for(unsigned i=0 ; i < num_composite_coords ; i++)
    {
        glm::vec3 p = m_composite_points[i]; 
        nuv uv = m_composite_coords[i]; 
        if(uv.matches(prim,sheet)) 
        {
            points.push_back(p) ;
            coords.push_back(uv) ;
        }
    }
}



void NNodePoints::dump_bb() const 
{
    unsigned num_prim_bb = m_prim_bb.size();
    unsigned num_prim_bb_selected = m_prim_bb_selected.size();


    LOG(info) << "NNodePoints::dump_bb"
              << " num_prim_bb " << num_prim_bb
              << " num_prim_bb_selected  " << num_prim_bb_selected
              ;

    assert( num_prim_bb == m_primitives.size()  );
    assert( num_prim_bb_selected == m_primitives.size()  );

    std::cout << " prim_bb " << std::endl ; 
    for(unsigned i=0 ; i < num_prim_bb ; i++)
    {
        std::cout << std::setw(4) << i 
                  << " " << m_prim_bb[i].desc()
                  << std::endl 
                  ;
    }

    std::cout << " prim_bb_selected " << std::endl ; 
    for(unsigned i=0 ; i < num_prim_bb ; i++)
    {
        std::cout << std::setw(4) << i 
                  << " " << m_prim_bb_selected[i].desc()
                  << std::endl 
                  ;
    }


}


void NNodePoints::dump_sheets() const 
{
    unsigned num_composite_points = m_composite_points.size() ;
    unsigned num_composite_coords = m_composite_coords.size() ;

    LOG(info) << "NNodePoints::dump_sheets"
              << " num_composite_points " << num_composite_points
              << " num_composite_coords " << num_composite_coords
              ;

    assert( num_composite_points == num_composite_coords );


    typedef std::map<unsigned, unsigned> MUU ; 
    MUU ps ; 

    for(unsigned i=0 ; i < num_composite_coords ; i++)
    {
        nuv uv = m_composite_coords[i]; 
        ps[uv.ps()]++ ;  // prim idx and sheet encoded into unsigned
    }

    LOG(info) << "NNodePoints::dump_sheets"
              << " nps " << ps.size()
               ;


    nbbox bbsp = bbox_surface_points();
    std::cout << " bbsp " << bbsp.desc() << std::endl ; 

    unsigned dmax = 200 ; 
    unsigned num_pass = 1 ; 

    for(unsigned pass=0 ; pass < num_pass ; pass++)
    {
        for(MUU::const_iterator it=ps.begin() ; it != ps.end() ; it++)
        {
             unsigned ps_ = it->first ; 
             unsigned count = it->second ; 

             unsigned prim = nuv::ps_to_prim(ps_);
             unsigned sheet = nuv::ps_to_sheet(ps_);


             std::vector<glm::vec3> points ; 
             std::vector<nuv> coords ; 

             selectPoints(points, coords, prim, sheet);

             nbbox ps_bbox = nbbox::from_points(points, m_verbosity );

             std::cout 
                   << " prim_sheet " << std::setw(5) << ps_
                   << " prim " << std::setw(5) << prim
                   << " sheet " << std::setw(5) << sheet 
                   << " count " << std::setw(5) << count 
                   << " ps_bbox " << ps_bbox.desc()
                   << std::endl ; 

             if(pass == 1)
             {
                 for(unsigned i=0 ; i < std::min<unsigned>(points.size(), dmax) ; i++)
                 {
                    std::cout << " i " << std::setw(5) << i
                              << " p " << gpresent(points[i])
                              << " c " << coords[i].detail()
                              << std::endl 
                              ;
                 }
             } 
        }
    }

}







