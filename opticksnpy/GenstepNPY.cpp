
#include <boost/lexical_cast.hpp>

// npy-
#include "GenstepNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "uif.h"

#include "PLOG.hh"




GenstepNPY::GenstepNPY(unsigned genstep_type, unsigned num_step, const char* config) 
       :  
       m_genstep_type(genstep_type),
       m_num_step(num_step),
       m_config(config ? strdup(config) : NULL),
       m_material(NULL),
       m_npy(NPY<float>::make(num_step, 6, 4)),
       m_step_index(0),
       m_ctrl(0,0,0,0),
       m_post(0,0,0,0),
       m_dirw(0,0,0,0),
       m_polw(0,0,0,0),
       m_zeaz(0,0,0,0),
       m_beam(0,0,0,0),
       m_frame(-1,0,0,0),
       m_frame_transform(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1),
       m_frame_targetted(false),
       m_num_photons_per_g4event(10000)
{
    m_npy->zero();
}





// used from cfg4-
void GenstepNPY::setNumPhotonsPerG4Event(unsigned int n)
{
    m_num_photons_per_g4event = n ; 
}
unsigned int GenstepNPY::getNumPhotonsPerG4Event()
{
    return m_num_photons_per_g4event ;
}
unsigned int GenstepNPY::getNumG4Event()
{
    unsigned int num_photons = getNumPhotons();
    unsigned int ppe = m_num_photons_per_g4event ; 
    unsigned int num_g4event ; 
    if(num_photons < ppe)
    {
        num_g4event = 1 ; 
    }
    else
    {
        assert( num_photons % ppe == 0 && "expecting num_photons to be exactly divisible by NumPhotonsPerG4Event " );
        num_g4event = num_photons / ppe ; 
    }
    return num_g4event ; 
}








void GenstepNPY::addActionControl(unsigned long long  action_control)
{
    m_npy->addActionControl(action_control);
}

const char* GenstepNPY::getMaterial()
{
    return m_material ; 
}
const char* GenstepNPY::getConfig()
{
    return m_config ; 
}

void GenstepNPY::setMaterial(const char* s)
{
    m_material = strdup(s);
}

unsigned GenstepNPY::getNumStep()
{
   return m_num_step ;  
}


/*
frame #4: 0x00000001007ce6d9 libNPY.dylib`GenstepNPY::addStep(this=0x0000000108667020, verbose=false) + 57 at GenstepNPY.cpp:56
frame #5: 0x0000000101e2a7d6 libOpticksGeometry.dylib`OpticksGen::makeTorchstep(this=0x0000000108664f50) + 150 at OpticksGen.cc:182
frame #6: 0x0000000101e2a32e libOpticksGeometry.dylib`OpticksGen::initInputGensteps(this=0x0000000108664f50) + 606 at OpticksGen.cc:74
frame #7: 0x0000000101e2a095 libOpticksGeometry.dylib`OpticksGen::init(this=0x0000000108664f50) + 21 at OpticksGen.cc:37
frame #8: 0x0000000101e2a073 libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108664f50, hub=0x0000000105609f20) + 131 at OpticksGen.cc:32
frame #9: 0x0000000101e2a0bd libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108664f50, hub=0x0000000105609f20) + 29 at OpticksGen.cc:33
frame #10: 0x0000000101e27706 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105609f20) + 118 at OpticksHub.cc:96
frame #11: 0x0000000101e27610 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105609f20, ok=0x0000000105421710) + 416 at OpticksHub.cc:81
frame #12: 0x0000000101e277ad libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105609f20, ok=0x0000000105421710) + 29 at OpticksHub.cc:83
frame #13: 0x0000000103790294 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfedd8, argc=1, argv=0x00007fff5fbfeeb8) + 260 at OKMgr.cc:46
*/

void GenstepNPY::addStep(bool verbose)
{
    bool dummy_frame = isDummyFrame();
    bool target_acquired = dummy_frame ? true : m_frame_targetted ;
    if(!target_acquired) 
         LOG(fatal) << "GenstepNPY::addStep target MUST be set for non-dummy frame " 
                    << " dummy_frame " << dummy_frame
                    << " m_frame_targetted " << m_frame_targetted
                    << brief()
                    ;

    assert(target_acquired);

    assert(m_npy && m_npy->hasData());

    unsigned int i = m_step_index ; 

    setGenstepType( m_genstep_type ) ;    

    update(); 

    if(verbose) dump("GenstepNPY::addStep");

    m_npy->setQuadI(m_ctrl, i, 0 );
    m_npy->setQuad( m_post, i, 1);
    m_npy->setQuad( m_dirw, i, 2);
    m_npy->setQuad( m_polw, i, 3);
    m_npy->setQuad( m_zeaz, i, 4);
    m_npy->setQuad( m_beam, i, 5);

    m_step_index++ ; 
}

NPY<float>* GenstepNPY::getNPY()
{
    assert( m_step_index == m_num_step && "GenstepNPY is incomplete, must addStep according to declared num_step");
    return m_npy ; 
}







// m_ctrl

void GenstepNPY::setGenstepType(unsigned genstep_type)
{
    m_ctrl.x = genstep_type ;  // eg TORCH
}
void GenstepNPY::setMaterialLine(unsigned int ml)
{
    m_ctrl.z = ml ; 
}


void GenstepNPY::setNumPhotons(const char* s)
{
    setNumPhotons(boost::lexical_cast<unsigned int>(s)) ; 
}
void GenstepNPY::setNumPhotons(unsigned int num_photons)
{
    m_ctrl.w = num_photons ; 
}
unsigned int GenstepNPY::getNumPhotons()
{
    return m_ctrl.w ; 
}



// m_post

void GenstepNPY::setPosition(const glm::vec4& pos)
{
    m_post.x = pos.x ; 
    m_post.y = pos.y ; 
    m_post.z = pos.z ; 
}

void GenstepNPY::setTime(const char* s)
{
    m_post.w = boost::lexical_cast<float>(s) ;
}
float GenstepNPY::getTime()
{
    return m_post.w ; 
}

glm::vec3 GenstepNPY::getPosition()
{
    return glm::vec3(m_post);
}




// m_dirw

void GenstepNPY::setDirection(const char* s)
{
    std::string ss(s);
    glm::vec3 dir = gvec3(ss) ;
    setDirection(dir);
}

void GenstepNPY::setDirection(const glm::vec3& dir)
{
    m_dirw.x = dir.x ; 
    m_dirw.y = dir.y ; 
    m_dirw.z = dir.z ; 
}

glm::vec3 GenstepNPY::getDirection()
{
    return glm::vec3(m_dirw);
}












void GenstepNPY::setWeight(const char* s)
{
    m_dirw.w = boost::lexical_cast<float>(s) ;
}


// m_polw

void GenstepNPY::setPolarization(const glm::vec4& pol)
{
    glm::vec4 npol = glm::normalize(pol);

    m_polw.x = npol.x ; 
    m_polw.y = npol.y ; 
    m_polw.z = npol.z ; 

    LOG(fatal) << "GenstepNPY::setPolarization"
              << " pol " << gformat(pol)
              << " npol " << gformat(npol)
              << " m_polw " << gformat(m_polw)
              ;

}
void GenstepNPY::setWavelength(const char* s)
{
    m_polw.w = boost::lexical_cast<float>(s) ;
}
float GenstepNPY::getWavelength()
{
    return m_polw.w ; 
}
glm::vec3 GenstepNPY::getPolarization()
{
    return glm::vec3(m_polw);
}





// m_zeaz

void GenstepNPY::setZenithAzimuth(const char* s)
{
    std::string ss(s);
    m_zeaz = gvec4(ss) ;
}
glm::vec4 GenstepNPY::getZenithAzimuth()
{
    return m_zeaz ; 
}



/// m_beam

void GenstepNPY::setRadius(const char* s)
{
    setRadius(boost::lexical_cast<float>(s)) ;
}
void GenstepNPY::setRadius(float radius)
{
    m_beam.x = radius ;
}
float GenstepNPY::getRadius()
{
    return m_beam.x ; 
}



void GenstepNPY::setDistance(const char* s)
{
    setDistance(boost::lexical_cast<float>(s)) ;
}
void GenstepNPY::setDistance(float distance)
{
    m_beam.y = distance ;
}



unsigned GenstepNPY::getBaseMode()
{
    uif_t uif ;
    uif.f = m_beam.z ; 
    return uif.u ; 
}
void GenstepNPY::setBaseMode(unsigned umode)
{
    uif_t uif ; 
    uif.u = umode ; 
    m_beam.z = uif.f ;
}




void GenstepNPY::setBaseType(unsigned utype)
{
    uif_t uif ; 
    uif.u = utype ; 
    m_beam.w = uif.f ;
}

unsigned GenstepNPY::getBaseType()
{
    uif_t uif ;
    uif.f = m_beam.w ; 
    return uif.u ; 
}





/*
frame #4: 0x00000001007cfdf8 libNPY.dylib`GenstepNPY::setFrameTransform(this=0x0000000108742400, frame_transform=0x00007fff5fbfe3f8)0>&) + 56 at GenstepNPY.cpp:317
frame #5: 0x0000000101e2bf87 libOpticksGeometry.dylib`OpticksGen::targetGenstep(this=0x0000000108740330, gs=0x0000000108742400) + 903 at OpticksGen.cc:126
frame #6: 0x0000000101e2b774 libOpticksGeometry.dylib`OpticksGen::makeTorchstep(this=0x0000000108740330) + 52 at OpticksGen.cc:177
frame #7: 0x0000000101e2b32e libOpticksGeometry.dylib`OpticksGen::initInputGensteps(this=0x0000000108740330) + 606 at OpticksGen.cc:74
frame #8: 0x0000000101e2b095 libOpticksGeometry.dylib`OpticksGen::init(this=0x0000000108740330) + 21 at OpticksGen.cc:37
frame #9: 0x0000000101e2b073 libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108740330, hub=0x0000000105609f20) + 131 at OpticksGen.cc:32
frame #10: 0x0000000101e2b0bd libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108740330, hub=0x0000000105609f20) + 29 at OpticksGen.cc:33
frame #11: 0x0000000101e28706 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105609f20) + 118 at OpticksHub.cc:96
frame #12: 0x0000000101e28610 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105609f20, ok=0x0000000105421710) + 416 at OpticksHub.cc:81
frame #13: 0x0000000101e287ad libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105609f20, ok=0x0000000105421710) + 29 at OpticksHub.cc:83
*/

void GenstepNPY::setFrameTransform(const char* s)
{
    std::string ss(s);
    bool flip = true ;  
    glm::mat4 transform = gmat4(ss, flip);
    setFrameTransform(transform);
}


void GenstepNPY::setFrameTransform(glm::mat4& frame_transform)
{
    m_frame_transform = frame_transform ;
    setFrameTargetted(true);
}
const glm::mat4& GenstepNPY::getFrameTransform()
{
    return m_frame_transform ;
}


void GenstepNPY::setFrameTargetted(bool targetted)
{
    m_frame_targetted = targetted ;
}
bool GenstepNPY::isFrameTargetted()
{
    return m_frame_targetted ;
} 

void GenstepNPY::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}
void GenstepNPY::setFrame(unsigned int vindex)
{
    m_frame.x = vindex ; 
    m_frame.y = 0 ; 
    m_frame.z = 0 ; 
    m_frame.w = 0 ; 
}
glm::ivec4& GenstepNPY::getFrame()
{
    return m_frame ; 
}

int GenstepNPY::getFrameIndex()
{
    return m_frame.x ; 
}

std::string GenstepNPY::brief()
{
    std::stringstream ss ; 

    ss << "GenstepNPY "
       << " frameIndex " << getFrameIndex()
       << " frameTargetted " << isFrameTargetted()
       << " frameTransform " << gformat(m_frame_transform)
       ;

    return ss.str();
}



bool GenstepNPY::isDummyFrame()
{
    return m_frame.x == -1 ; 
}



void GenstepNPY::dump(const char* msg)
{
    dumpBase(msg);
}

void GenstepNPY::dumpBase(const char* msg)
{
    LOG(info) << msg  
              << " config " << m_config 
              << " material " << m_material
              ; 

    print(m_ctrl, "m_ctrl : id/pid/MaterialLine/NumPhotons" );
    print(m_post, "m_post : position, time " ); 
    print(m_dirw, "m_dirw : direction, weight" ); 
    print(m_polw, "m_polw : polarization, wavelength" ); 
    print(m_zeaz, "m_zeaz: zenith, azimuth " ); 
    print(m_beam, "m_beam: radius,... " ); 

    print(m_frame, "m_frame ");
}





