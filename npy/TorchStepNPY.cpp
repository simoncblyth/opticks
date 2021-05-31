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

#include <vector>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cassert>


// brap-
#include "BStr.hh"

// npy-
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "uif.h"
#include "NStep.hpp"
#include "GenstepNPY.hpp"
#include "TorchStepNPY.hpp"
#include "PLOG.hh"


// frame=3153,      DYB NEAR AD center
// frame=62593      j1808 lAcrylic0x4bd3f20         ce  0.000   0.000   0.009 17820.008
//    "frame=3153_"
//    "frame=62593_"

const char* TorchStepNPY::DEFAULT_CONFIG = 
    "type=sphere_"
    "frame=0_"
    "source=0,0,0_"
    "target=0,0,1_"
    "photons=10000_"
    "material=GdDopedLS_"   
    "wavelength=430_"
    "weight=1.0_"
    "time=0.1_"
    "zenithazimuth=0,1,0,1_"
    "radius=0_" ;

//  Aug 2016: change default torch wavelength from 380nm to 430nm
//  Sep 2018: reduce 100k to 10k for faster test runnig 
//
// NB time 0.f causes 1st step record rendering to be omitted, as zero is special
// NB the material string needs to be externally translated into a material line
//
// TODO: material lookup based on the frame volume solid ?
//
 
const char* TorchStepNPY::T_UNDEF_    = "undef" ; 
const char* TorchStepNPY::T_POINT_    = "point" ; 
const char* TorchStepNPY::T_SPHERE_    = "sphere" ; 
const char* TorchStepNPY::T_DISC_      = "disc" ; 
const char* TorchStepNPY::T_DISC_INTERSECT_SPHERE_ = "discIntersectSphere" ; 
const char* TorchStepNPY::T_DISC_INTERSECT_SPHERE_DUMB_ = "discIntersectSphereDumb" ; 
const char* TorchStepNPY::T_DISCLIN_   = "disclin" ; 
const char* TorchStepNPY::T_DISCAXIAL_   = "discaxial" ; 
const char* TorchStepNPY::T_INVSPHERE_ = "invsphere" ; 
const char* TorchStepNPY::T_REFLTEST_  = "refltest" ; 
const char* TorchStepNPY::T_INVCYLINDER_ = "invcylinder" ; 
const char* TorchStepNPY::T_RING_ = "ring" ; 

Torch_t TorchStepNPY::ParseType(const char* k)  // static 
{
    Torch_t type = T_UNDEF ;
    if(       strcmp(k,T_SPHERE_)==0)    type = T_SPHERE ; 
    else if(  strcmp(k,T_POINT_)==0)     type = T_POINT ; 
    else if(  strcmp(k,T_DISC_)==0)      type = T_DISC ; 
    else if(  strcmp(k,T_DISCLIN_)==0)   type = T_DISCLIN ; 
    else if(  strcmp(k,T_DISCAXIAL_)==0) type = T_DISCAXIAL ; 
    else if(  strcmp(k,T_DISC_INTERSECT_SPHERE_)==0) type = T_DISC_INTERSECT_SPHERE ; 
    else if(  strcmp(k,T_DISC_INTERSECT_SPHERE_DUMB_)==0) type = T_DISC_INTERSECT_SPHERE_DUMB ; 
    else if(  strcmp(k,T_INVSPHERE_)==0) type = T_INVSPHERE ; 
    else if(  strcmp(k,T_INVCYLINDER_)==0) type = T_INVCYLINDER ; 
    else if(  strcmp(k,T_REFLTEST_)==0)  type = T_REFLTEST ; 
    else if(  strcmp(k,T_RING_)==0)      type = T_RING ; 
    return type ;   
}

::Torch_t TorchStepNPY::getType() const 
{
    unsigned utype = m_onestep->getBaseType();
    Torch_t type = T_UNDEF ;
    switch(utype)
    {
       case T_SPHERE:      type=T_SPHERE     ;break; 
       case T_DISC:        type=T_DISC       ;break; 
       case T_POINT:       type=T_POINT      ;break; 
       case T_DISCLIN:     type=T_DISCLIN    ;break; 
       case T_DISCAXIAL:   type=T_DISCAXIAL  ;break; 
       case T_DISC_INTERSECT_SPHERE:   type=T_DISC_INTERSECT_SPHERE  ;break; 
       case T_DISC_INTERSECT_SPHERE_DUMB:   type=T_DISC_INTERSECT_SPHERE_DUMB  ;break; 
       case T_INVSPHERE:   type=T_INVSPHERE  ;break; 
       case T_REFLTEST:    type=T_REFLTEST   ;break; 
       case T_INVCYLINDER: type=T_INVCYLINDER;break; 
       case T_RING:        type=T_RING       ;break; 
    }
    return type ; 
}

const char* TorchStepNPY::getTypeName() const 
{
    ::Torch_t type = getType();
    const char* name = NULL ; 
    switch(type)
    {
       case T_SPHERE:      name=T_SPHERE_     ;break; 
       case T_DISC:        name=T_DISC_       ;break; 
       case T_POINT:       name=T_POINT_      ;break; 
       case T_DISCLIN:     name=T_DISCLIN_    ;break; 
       case T_DISCAXIAL:   name=T_DISCAXIAL_  ;break; 
       case T_DISC_INTERSECT_SPHERE:   name=T_DISC_INTERSECT_SPHERE_  ;break; 
       case T_DISC_INTERSECT_SPHERE_DUMB:   name=T_DISC_INTERSECT_SPHERE_DUMB_  ;break; 
       case T_INVSPHERE:   name=T_INVSPHERE_  ;break; 
       case T_REFLTEST:    name=T_REFLTEST_   ;break; 
       case T_INVCYLINDER: name=T_INVCYLINDER_;break; 
       case T_RING:        name=T_RING_       ;break; 

       case T_UNDEF:       name=T_UNDEF_      ;break; 
       default:    
                           assert(0)  ;

    }
    return name ; 
}


void TorchStepNPY::setType(const char* s)
{
    ::Torch_t type = ParseType(s) ;
    unsigned utype = unsigned(type);
    m_onestep->setBaseType(utype);
}





const char* TorchStepNPY::M_SPOL_ = "spol" ; 
const char* TorchStepNPY::M_PPOL_ = "ppol" ; 
const char* TorchStepNPY::M_FIXPOL_ = "fixpol" ; 
const char* TorchStepNPY::M_FLAT_THETA_ = "flatTheta" ; 
const char* TorchStepNPY::M_FLAT_COSTHETA_ = "flatCosTheta" ; 
const char* TorchStepNPY::M_WAVELENGTH_SOURCE_ = "wavelengthSource" ; 
const char* TorchStepNPY::M_WAVELENGTH_COMB_ = "wavelengthComb" ; 

Mode_t TorchStepNPY::ParseMode(const char* k)  // static 
{
    Mode_t mode = M_UNDEF ;
    if(       strcmp(k,M_SPOL_)==0)               mode = M_SPOL ; 
    else if(  strcmp(k,M_PPOL_)==0)               mode = M_PPOL ; 
    else if(  strcmp(k,M_FLAT_THETA_)==0)         mode = M_FLAT_THETA ; 
    else if(  strcmp(k,M_FLAT_COSTHETA_)==0)      mode = M_FLAT_COSTHETA ; 
    else if(  strcmp(k,M_FIXPOL_)==0)             mode = M_FIXPOL ; 
    else if(  strcmp(k,M_WAVELENGTH_SOURCE_)==0)  mode = M_WAVELENGTH_SOURCE ; 
    else if(  strcmp(k,M_WAVELENGTH_COMB_)==0)    mode = M_WAVELENGTH_COMB ; 
    return mode ;   
}

::Mode_t TorchStepNPY::getMode() const 
{
    ::Mode_t mode = (::Mode_t)m_onestep->getBaseMode() ;
    return mode ; 
}

std::string TorchStepNPY::getModeString() const 
{
    std::stringstream ss ; 
    ::Mode_t mode = getMode();

    if(mode & M_SPOL)              ss << M_SPOL_ << " " ;
    if(mode & M_PPOL)              ss << M_PPOL_ << " " ;
    if(mode & M_FLAT_THETA)        ss << M_FLAT_THETA_ << " " ; 
    if(mode & M_FLAT_COSTHETA)     ss << M_FLAT_COSTHETA_ << " " ; 
    if(mode & M_FIXPOL)            ss << M_FIXPOL_ << " " ; 
    if(mode & M_WAVELENGTH_SOURCE) ss << M_WAVELENGTH_SOURCE_ << " " ; 
    if(mode & M_WAVELENGTH_COMB)   ss << M_WAVELENGTH_COMB_ << " " ; 

    return ss.str();
} 



std::string TorchStepNPY::description() const 
{
    glm::vec3 pos = m_onestep->getPosition() ;
    glm::vec3 dir = m_onestep->getDirection() ;
    glm::vec3 pol = m_onestep->getPolarization() ;

    float radius = m_onestep->getRadius(); 
    float wavelength = m_onestep->getWavelength(); 
    float time = m_onestep->getTime();

    std::stringstream ss ; 
    ss
        << " typeName " << getTypeName() 
        << " modeString " << getModeString() 
        << " position " << gformat(pos)
        << " direction " << gformat(dir)
        << " polarization " << gformat(pol)
        << " radius " << radius
        << " wavelength " << wavelength
        << " time " << time
        ; 

    return ss.str();
}






const char* TorchStepNPY::TYPE_ = "type"; 
const char* TorchStepNPY::MODE_ = "mode"; 
const char* TorchStepNPY::POLARIZATION_ = "polarization"; 
const char* TorchStepNPY::FRAME_ = "frame"; 
const char* TorchStepNPY::TRANSFORM_ = "transform"; 
const char* TorchStepNPY::SOURCE_ = "source"; 
const char* TorchStepNPY::TARGET_ = "target" ; 
const char* TorchStepNPY::PHOTONS_ = "photons" ; 
const char* TorchStepNPY::MATERIAL_ = "material" ; 
const char* TorchStepNPY::ZENITHAZIMUTH_ = "zenithazimuth" ; 
const char* TorchStepNPY::WAVELENGTH_     = "wavelength" ; 
const char* TorchStepNPY::WEIGHT_     = "weight" ; 
const char* TorchStepNPY::TIME_     = "time" ; 
const char* TorchStepNPY::RADIUS_   = "radius" ; 
const char* TorchStepNPY::DISTANCE_   = "distance" ; 

TorchStepNPY::Param_t TorchStepNPY::ParseParam(const char* k)  // static 
{
    Param_t param = UNRECOGNIZED ; 
    if(     strcmp(k,FRAME_)==0)          param = FRAME ; 
    else if(strcmp(k,TRANSFORM_)==0)      param = TRANSFORM ; 
    else if(strcmp(k,TYPE_)==0)           param = TYPE ; 
    else if(strcmp(k,MODE_)==0)           param = MODE ; 
    else if(strcmp(k,POLARIZATION_)==0)   param = POLARIZATION ; 
    else if(strcmp(k,SOURCE_)==0)         param = SOURCE ; 
    else if(strcmp(k,TARGET_)==0)         param = TARGET ; 
    else if(strcmp(k,PHOTONS_)==0)        param = PHOTONS ; 
    else if(strcmp(k,MATERIAL_)==0)       param = MATERIAL ; 
    else if(strcmp(k,ZENITHAZIMUTH_)==0)  param = ZENITHAZIMUTH ; 
    else if(strcmp(k,WAVELENGTH_)==0)     param = WAVELENGTH ; 
    else if(strcmp(k,WEIGHT_)==0)         param = WEIGHT ; 
    else if(strcmp(k,TIME_)==0)           param = TIME ; 
    else if(strcmp(k,RADIUS_)==0)         param = RADIUS ; 
    else if(strcmp(k,DISTANCE_)==0)       param = DISTANCE ; 
    return param ;  
}

void TorchStepNPY::set(Param_t p, const char* s)
{
    bool ignore = false ; 
    switch(p)
    {
        case TYPE           : setType(s)           ;break;
        case MODE           : setMode(s)           ;break;
        case TRANSFORM      : setFrameTransform(s) ;break;
        case FRAME          : setFrame(s)          ;break;
        case POLARIZATION   : setPolarizationLocal(s) ;break;
        case SOURCE         : setSourceLocal(s)    ;break;
        case TARGET         : setTargetLocal(s)    ;break;
        case MATERIAL       : setMaterial(s)       ;break;
        case PHOTONS        : m_onestep->setNumPhotons(s)     ;break;
        case ZENITHAZIMUTH  : m_onestep->setZenithAzimuth(s)  ;break;
        case WAVELENGTH     : m_onestep->setWavelength(s)     ;break;
        case WEIGHT         : m_onestep->setWeight(s)         ;break;
        case TIME           : m_onestep->setTime(s)           ;break;
        case RADIUS         : m_onestep->setRadius(s)         ;break;
        case DISTANCE       : m_onestep->setDistance(s)       ;break;
        case UNRECOGNIZED   : ignore=true ;  
    }

    if(ignore)
    {
        LOG(fatal) << "TorchStepNPY::set WARNING ignoring unrecognized parameter [" << s << "]"  ; 
        assert(0);
    }
}





// set the below defaults to avoid a messy undefined polarization with OpSnapTest

TorchStepNPY::TorchStepNPY(unsigned gentype, unsigned int num_step, const char* config) 
    :  
    GenstepNPY(gentype,  num_step, config ? strdup(config) : DEFAULT_CONFIG, config == NULL ),
    m_source_local(0,0,0,1),
    m_target_local(0,0,0,1),
    m_polarization_local(0,0,1,0),
    m_src(0,0,1,1),
    m_tgt(0,0,0,1),
    m_pol(0,0,1,1),
    m_dir(0,0,1),
    m_level(debug)
{
    init();
}



void TorchStepNPY::init()
{
    const char* config = getConfig(); 

    LOG(m_level) 
        << " config " <<  config 
        ;

    typedef std::pair<std::string,std::string> KV ; 
    std::vector<KV> ekv = BStr::ekv_split(config,'_',"=");

    LOG(m_level) 
        << " config " <<  config 
        << " ekv " << ekv.size() 
        ;


    for(std::vector<KV>::const_iterator it=ekv.begin() ; it!=ekv.end() ; it++)
    {
        const char* k = it->first.c_str() ;  
        const char* v = it->second.c_str() ;  

        Param_t p = ParseParam(k) ;

        LOG(m_level) << std::setw(20) << k << ":" << v  ; 

        if(p == UNRECOGNIZED)
        {
            LOG(fatal) << "TorchStepNPY::init "
                       << " UNRECOGNIZED parameter "
                       << " [" << k << "] = [" << v << "]"
                       ;
            assert(0);
        }


        if(strlen(v)==0)
        {
            LOG(warning) << "TorchStepNPY::init skip empty value for key " << k ; 
        }
        else
        {
            set(p, v);
        }
    }
}





glm::vec4 TorchStepNPY::getSourceLocal() const 
{
    return m_source_local ; 
}
glm::vec4 TorchStepNPY::getTargetLocal() const 
{
    return m_target_local ; 
}
glm::vec4 TorchStepNPY::getPolarizationLocal() const 
{
    return m_polarization_local ; 
}





bool TorchStepNPY::isIncidentSphere() const
{
    ::Torch_t type = getType();
    return type == T_DISC_INTERSECT_SPHERE  ;
}

bool TorchStepNPY::isDisc() const
{
    ::Torch_t type = getType();
    return type == T_DISC  ;
}
bool TorchStepNPY::isDiscLinear() const
{
    ::Torch_t type = getType();
    return type == T_DISCLIN  ;
}
bool TorchStepNPY::isRing() const
{
    ::Torch_t type = getType();
    return type == T_RING  ;
}
bool TorchStepNPY::isPoint() const
{
    ::Torch_t type = getType();
    return type == T_POINT  ;
}
bool TorchStepNPY::isSphere() const
{
    ::Torch_t type = getType();
    return type == T_SPHERE  ;
}

bool TorchStepNPY::isReflTest() const
{
    ::Torch_t type = getType();
    return type == T_REFLTEST ;
}


bool TorchStepNPY::isSPolarized() const
{
    ::Mode_t  mode = getMode();
    return (mode & M_SPOL) != 0  ;
}
bool TorchStepNPY::isPPolarized() const
{
    ::Mode_t  mode = getMode();
    return (mode & M_PPOL) != 0  ;
}
bool TorchStepNPY::isFixPolarized() const
{
    ::Mode_t  mode = getMode();
    return (mode & M_FIXPOL) != 0  ;
}





void TorchStepNPY::setSourceLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_source_local.x = v.x;
    m_source_local.y = v.y;
    m_source_local.z = v.z;
    m_source_local.w = 1.0;
}

void TorchStepNPY::setTargetLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_target_local.x = v.x;
    m_target_local.y = v.y;
    m_target_local.z = v.z;
    m_target_local.w = 1.0;
}

void TorchStepNPY::setPolarizationLocal(const char* s)
{
    std::string ss(s);
    glm::vec3 v = gvec3(ss) ; 
    m_polarization_local.x = v.x;
    m_polarization_local.y = v.y;
    m_polarization_local.z = v.z;
    m_polarization_local.w = 0.0;  // direction not position
}

/**
TorchStepNPY::updateAfterSetFrameTransform
--------------------------------------------

This fulfils a virtual method from the base class, and is
invoked from their after setFrameTransform is called on the base.
The new frame transform is used to convert from local frame 
source, target and polarization into the frame provided. 

**/

void TorchStepNPY::updateAfterSetFrameTransform()
{
   // direction from: target - source
   // position from : source

    LOG(LEVEL); 

    const glm::mat4& frame_transform = getFrameTransform() ;

    m_src = frame_transform * m_source_local  ; 
    m_tgt = frame_transform * m_target_local  ; 
    glm::vec4 pol = frame_transform * m_polarization_local  ;   // yields unnormalized, but GenstepNPY setter normalizes

    m_dir = glm::vec3(m_tgt) - glm::vec3(m_src) ;

    glm::vec3 dir = glm::normalize( m_dir );

    m_onestep->setPosition(m_src);
    m_onestep->setDirection(dir);
    m_onestep->setPolarization(pol); 
}


void TorchStepNPY::setMode(const char* s)
{
    std::vector<std::string> tp; 
    BStr::split(tp, s, ',' ); 

    //::Mode_t mode = M_UNDEF ; 
    unsigned int mode = 0 ; 

    for(unsigned int i=0 ; i < tp.size() ; i++)
    {
        const char* emode_ = tp[i].c_str() ;
        ::Mode_t emode = ParseMode(emode_) ;
        mode |= emode ; 
        LOG(debug) << "TorchStepNPY::setMode" 
                  << " emode_ " << emode_
                  << " emode " << emode
                  << " mode " << mode 
                  ; 
    }

    m_onestep->setBaseMode(mode);
}

std::string TorchStepNPY::desc(const char* msg) const 
{
    int wid = 10 ; 
    int prec = 3 ; 

    std::stringstream ss ; 
    ss
        << msg << std::endl  
        << GLMFormat::Format(m_source_local, wid, prec) << "m_source_local      " 
        << GLMFormat::Format(m_src,          wid, prec) << "m_src"
        << std::endl  
        << GLMFormat::Format(m_target_local, wid, prec) << "m_target_local      " 
        << GLMFormat::Format(m_tgt,          wid, prec) << "m_tgt" 
        << std::endl  
        << GLMFormat::Format(m_polarization_local, wid, prec) << "m_polarization_local"
        << GLMFormat::Format(m_pol,          wid, prec) << "m_pol" 
        << std::endl  
        << GLMFormat::Format3(m_dir, wid, prec) << "m_dir  nrm(m_tgt - m_src)" 
        << std::endl  
        ;
 
    ss << m_onestep->desc("TorchStepNPY::desc") ; 
    std::string s = ss.str(); 
    return s ; 
}

void TorchStepNPY::dump(const char* msg) const 
{
    LOG(info) << desc(msg) ; 
}


void TorchStepNPY::Summary(const char* msg) const
{
    LOG(info) << msg  << description() ; 
}


