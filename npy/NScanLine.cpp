/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <string>
#include <iomanip>
#include <sstream>

#include "PLOG.hh"

#include "GLMFormat.hpp"
#include "NGLM.hpp"
#include "NNode.hpp"
#include "NScanLine.hpp"


const std::string& NScanLine::get_message() const 
{
    return m_message ; 
}


float NScanLine::path_length() const 
{
    return glm::length(m_path) ;    
}
float NScanLine::step_length() const 
{
    return glm::length(m_step) ;    
}

unsigned NScanLine::num_step() const 
{
    float pl = path_length();
    float sl = step_length();
    float fstep = pl/sl ;
    return unsigned(fstep);
}

unsigned NScanLine::get_num_step() const 
{
    return m_num_step ; 
}

glm::vec3 NScanLine::position(int ustep) const 
{
    if(ustep < 0) ustep += m_num_step ; 
    return m_begin + float(ustep) * m_step ;  
} 


NScanLine::NScanLine( const glm::vec3& begin, const glm::vec3& end, const glm::vec3& step, unsigned verbosity )
    :
    m_begin(begin),
    m_end(end),
    m_path(m_end-m_begin),
    m_step(step),
    m_verbosity(verbosity),
    m_num_step(num_step()),
    m_node(NULL),
    m_num_nodes(0),
    m_message("")
{
    if(m_num_step > 1e5) 
    {
        LOG(warning) << "limiting steps " << m_num_step ; 
        m_step.x = 100*m_step.x ; 
        m_step.y = 100*m_step.y ; 
        m_step.z = 100*m_step.z ; 
        m_num_step /= 100 ; 
    }
}


std::string NScanLine::desc() const 
{
    glm::vec3 p0 = position(0) ;
    glm::vec3 p1 = position(1) ;
    glm::vec3 p_2 = position(-2) ;
    glm::vec3 p_1 = position(-1) ;

    std::stringstream ss ; 
    ss
        << "NScanLine "
        << " verbosity " << m_verbosity 
        << " path_length " << path_length()
        << " step_length " << step_length()
        << " num_step " << m_num_step 
        << std::endl 
        << gpresent("beg", m_begin) 
        << gpresent("end", m_end) 
        << gpresent("path", m_path) 
        << gpresent("stp", m_step) 
        << gpresent("p0",  p0) 
        << gpresent("p1",  p1) 
        << gpresent("p-2",  p_2) 
        << gpresent("p-1",  p_1) 
        ;
        return ss.str();
}

std::string NScanLine::desc_zeros() const 
{
    unsigned count_zeros_ = count_zeros(0);

    assert(m_node);
    std::stringstream ss ; 
    ss 
        << "NScanLine::desc_zeros"
        << " node " << m_node->tag() 
        << " count_zeros " << count_zeros_
        << " num_zeros " << m_zeros.size() << std::endl
        ;

    unsigned num_zeros = m_zeros.size() ;
    for(unsigned i=0 ; i < num_zeros ; i++)
    {
        glm::uvec4 t_brak = m_zeros[i] ;
        glm::vec3 pos = position(t_brak.x);
        
        unsigned step = t_brak.x ; 
        unsigned touchzero = t_brak.z ; 
        unsigned node_idx  = t_brak.w ; 

        ss 
           << " st " << std::setw(10) << std::fixed << std::setprecision(3) << step
           << " ni " << std::setw(10) << std::fixed << std::setprecision(3) << node_idx
           << ( touchzero ? " ZERO " : "      " )
           << " pos " << gpresent( pos ) 
           << std::endl   
           ;
    }
    return ss.str();
}


unsigned NScanLine::count_zeros(unsigned node_idx) const
{
    std::vector<glm::uvec4> zeros ; 
    get_zeros(zeros, node_idx);    
    return zeros.size();
}

void NScanLine::get_zeros(std::vector<glm::uvec4>& zeros, unsigned node_idx) const
{
    for(unsigned i=0 ; i < m_zeros.size() ; i++)
    {
        glm::uvec4 zero = m_zeros[i] ;
        if(zero.w == node_idx) zeros.push_back(zero) ; 
    }
}
void NScanLine::dump_zeros(unsigned node_idx, unsigned step_window) 
{
    std::vector<glm::uvec4> zeros ; 
    get_zeros(zeros, node_idx);    
    unsigned nzero = zeros.size();

    LOG(info) << "NScanLine::dump_zeros"
              << " nzero " << nzero
              << " node_idx " << node_idx 
              << " step_window " << step_window
              ;

    for(unsigned i=0 ; i < nzero ; i++)
    {
        glm::uvec4 zero = zeros[i];
        int t0 = std::max<int>( int(zero.x-step_window), 0 );  
        int t1 = std::min<int>( int(zero.x+step_window), m_num_step );  
        dump("dump_zeros", t0, t1 );
    }
}


void NScanLine::setNode(const nnode* node)
{
    m_node = node ; 
    m_nodes.push_back(node);
    m_num_nodes = 1 ; 
}
void NScanLine::setNode_(const nnode* node)
{
    m_node = node ; 
}

void NScanLine::setNodes(const std::vector<const nnode*>& nodes)
{
    m_nodes.clear();
    m_num_nodes = nodes.size() ; 
    for(unsigned i=0 ; i < m_num_nodes ; i++ ) m_nodes.push_back(nodes[i]) ; 

    m_node = m_nodes[0] ;
}

float NScanLine::sdf( int ustep ) const 
{
    glm::vec3 pos = position(ustep); 
    assert(m_node);
    return (*m_node)(pos.x, pos.y, pos.z) ; 
}

float NScanLine::sdf0() const 
{
    return sdf(0);
}
float NScanLine::sdf1() const 
{
    return sdf(-1);
}

bool NScanLine::has_zero( unsigned t0 ) const 
{
    for(unsigned i=0 ; i < m_zeros.size() ; i++)
    {
        glm::uvec4 tbrak = m_zeros[i] ; 
        if( tbrak.x == t0 ) return true ; 
    }
    return false ; 
}

bool NScanLine::has_zero(const glm::uvec4& brak) const 
{
    for(unsigned i=0 ; i < m_zeros.size() ; i++)
    {
        glm::uvec4 tbrak = m_zeros[i] ; 
        if( tbrak.x == brak.x ) return true ; 
    }
    return false ; 
}

void NScanLine::add_zero(const glm::uvec4& tbrak )
{
    if(!has_zero(tbrak)) m_zeros.push_back(tbrak);
}

void NScanLine::find_zeros()
{
    bool reverse = false ; 
    for(unsigned i=0 ; i < m_num_nodes ; i++)
    {
        const nnode* n = m_nodes[reverse ? m_num_nodes - 1 - i : i ] ; 
        setNode_(n);
        find_zeros_one();
        if(m_verbosity > 2)
        std::cout << desc_zeros() << std::endl ;         
    }
}

void NScanLine::find_zeros_one()
{
    float sd0 = sdf0();
    float sd1 = sdf1();

    unsigned node_idx = m_node->idx ;  
    bool start_end_outside = sd0 > 0 && sd1 > 0 ;

    if(node_idx == 0)
    {
        if(!start_end_outside)
        {
            m_message.assign("not outside?");

            LOG(debug) << "NScanLine::find_zeros_one"
                         << " SCANLINE DOESNT START AND END OUTSIDE GEOMETRY " 
                         << " sd0 " << sd0 
                         << " sd1 " << sd1
                         ;
        }
        //assert( start_end_outside > 0 && "NScanLine expects to start and end outside the geometry" ) ; 
        // hmm what about complements  : root node should not be complemented
    }

    float lsd = sd0 ; 
    int prior_zero_step = -2 ; 

    for(unsigned step=0  ; step <= m_num_step ; step++)
    {
        float sd = sdf(step);

        float sd_lsd = sd*lsd ; 
        bool touchzero = sd_lsd == 0.f ; 
        bool signchange =  sd_lsd <= 0.f ;
    
/* 
        glm::vec3 pos = position(step) ; 
        if(signchange || sd < 0)
        std::cout << "NScanLine::find_zeros_one"
                  << " step " << step
                  << " sd " << sd
                  << " lsd " << lsd
                  << " pos " << glm::to_string(pos) 
                  << ( signchange ? " SIGNCHANGE" : "" )
                  << std::endl ; 
*/

        if(signchange && int(step) != prior_zero_step+1)  // avoid double adding of actual zero hits 
        {
            glm::uvec4 bracket(step,0,  touchzero ? 1 : 0,  node_idx);
            add_zero(bracket) ;
            prior_zero_step = step ;  
        }
        lsd = sd ;
    }
}


void NScanLine::sample(std::vector<float>& sd) const 
{
    for(unsigned step=0  ; step <= m_num_step ; step++)
    {
        float sd_ = sdf(step);
        sd.push_back(sd_);
    }
}




void NScanLine::dump( const char* msg, int t0_, int t1_)  // not const as uses setNode
{
    if(t0_ < 0 ) t0_ += m_num_step ; 
    if(t1_ < 0 ) t1_ += m_num_step ; 

    unsigned t0 = t0_ ;
    unsigned t1 = t1_ ;

    LOG(info) << msg 
              << " t0_ " << t0_ 
              << " t1_ " << t1_
              << " t0 " << t0 
              << " t1 " << t1
              << " num_step " << m_num_step
              ;

    std::cout << desc() << std::endl ; 

    int wid = 14 ; 
    std::cout
         << std::setw(wid) << " i " 
         << std::setw(wid) << " x " 
         << std::setw(wid) << " y " 
         << std::setw(wid) << " z " 
         ;

    for(unsigned i=0 ; i < m_num_nodes ; i++)
    {
        const nnode* n = m_nodes[i] ; 
        std::cout << std::setw(wid) << n->tag() ;
    }
    std::cout << std::endl ; 

    for(unsigned t=t0 ; t < t1 ; t++ )
    {
        glm::vec3 p = position(t);

        std::cout
              <<  std::fixed << std::setprecision(4) << std::setw(wid) << t
              <<  std::fixed << std::setprecision(4) << std::setw(wid) << p.x 
              <<  std::fixed << std::setprecision(4) << std::setw(wid) << p.y 
              <<  std::fixed << std::setprecision(4) << std::setw(wid) << p.z 
              ;
        
        for(unsigned i=0 ; i < m_num_nodes ; i++)
        {
            const nnode* n = m_nodes[i] ; 
            setNode_(n);
            float sd  = sdf(t );

            std::cout
                << "  "
                <<  std::fixed << std::setprecision(4) << std::setw(wid-4) << sd  
                << ( sd < 0 ? " *" : "  " )
             
                ;
        } 
        std::cout << std::endl ; 
    }
}





