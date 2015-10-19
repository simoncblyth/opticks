#include "Composition.hh"

// npy-
#include "NPY.hpp"
#include "NumpyEvt.hpp"
#include "RecordsNPY.hpp"
#include "PhotonsNPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "AxisNPY.hpp"


// oglrap-
#include "Camera.hh"
#include "Trackball.hh"
#include "View.hh"
#include "Light.hh"
#include "Clipper.hh"
#include "Scene.hh"
#include "Animator.hh"

#include "CameraCfg.hh"
#include "TrackballCfg.hh"
#include "ViewCfg.hh"
#include "ClipperCfg.hh"

#include "CompositionCfg.hh"

// npy-
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>


// TODO: rearrange defines to avoid duplicating this
// mm/ns
#define SPEED_OF_LIGHT 299.792458f


#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#ifdef GUI_
#include <imgui.h>
#endif

#include "limits.h"


const char* Composition::PRINT = "print" ; 
const char* Composition::SELECT = "select" ; 
const char* Composition::RECSELECT = "recselect" ; 
const char* Composition::PICKPHOTON = "pickphoton" ; 
const char* Composition::PICKFACE = "pickface" ; 



const char* Composition::WHITE_ = "white" ; 
const char* Composition::MAT1_  = "mat1"; 
const char* Composition::MAT2_  = "mat2"; 
const char* Composition::FLAG1_ = "flag1"; 
const char* Composition::FLAG2_ = "flag2"; 
const char* Composition::POL1_  = "pol1"; 
const char* Composition::POL2_  = "pol2" ; 
 

const char* Composition::getColorStyleName(Composition::ColorStyle_t style)
{
    switch(style)
    {
       case WHITE:return WHITE_ ; break ; 
       case MAT1 :return MAT1_ ; break ; 
       case MAT2 :return MAT2_ ; break ; 
       case FLAG1:return FLAG1_ ; break ; 
       case FLAG2:return FLAG2_ ; break ; 
       case POL1 :return POL1_ ; break ; 
       case POL2 :return POL2_ ; break ; 
       case NUM_COLORSTYLE :assert(0) ; break ; 
    }
    assert(0);
    return NULL ; 
}



void Composition::init()
{
    m_camera = new Camera() ;
    m_view   = new View() ;
    m_light = new Light() ;
    m_trackball = new Trackball() ;
    m_clipper = new Clipper() ;

    initAxis();
}




Composition::~Composition()
{
}

std::vector<std::string> Composition::getTags()
{
    std::vector<std::string> tags ;
    tags.push_back(SELECT);
    tags.push_back(RECSELECT);
    return tags ; 
}

bool Composition::accepts(const char* name)
{
    return 
         strcmp(name,SELECT)==0 || 
         strcmp(name,RECSELECT)==0 ;
}

void Composition::set(const char* name, std::string& s)
{
    if(     strcmp(name,SELECT)==0) setSelection(s);
    else if(strcmp(name,RECSELECT)==0) setRecSelect(s);
    else if(strcmp(name,PICKPHOTON)==0) setPickPhoton(s);
    else if(strcmp(name,PICKFACE)==0) setPickFace(s);
    else
         printf("Composition::set bad name %s\n", name);
}

std::string Composition::get(const char* name)
{
   std::string s ; 

   if(     strcmp(name,SELECT)==0)    s = gformat(getSelection()) ;
   else if(strcmp(name,RECSELECT)==0) s = gformat(getRecSelect()) ;
   else if(strcmp(name,PICKPHOTON)==0) s = gformat(getPickPhoton()) ;
   else if(strcmp(name,PICKFACE)==0) s = gformat(getPickFace()) ;
   else
         printf("Composition::get bad name %s\n", name);

   return s ; 
}

void Composition::configure(const char* name, const char* value_)
{
    std::string value(value_);
    set(name, value);
}

void Composition::configureS(const char* name, std::vector<std::string> values)
{
    if(values.empty()) return ;

    std::string last = values.back();
    set(name, last);
}


void Composition::configureI(const char* name, std::vector<int> values )
{
    printf("Composition::configureI\n");
    if(values.empty()) return ; 
    if(strcmp(name, PRINT)==0)
    {
        //int print = values.back();
        Summary("Composition::configureI");
    }
}





void Composition::addConfig(Cfg* cfg)
{
    // hmm problematic with bookmarks that swap out Camera, View, ...
    cfg->add(new CompositionCfg<Composition>("composition", this,          true));
    cfg->add(new CameraCfg<Camera>(          "camera",      getCamera(),   true));
    cfg->add(new ViewCfg<View>(              "view",        getView(),     true));
    cfg->add(new TrackballCfg<Trackball>(    "trackball",   getTrackball(),true));
    cfg->add(new ClipperCfg<Clipper>(        "clipper",     getClipper(),  true));
}



void Composition::initAnimator()
{
    // must defer creation (to render time) as domain_time not set at initialization
    float* target = glm::value_ptr(m_param) + 3 ;
    m_animator = new Animator(target, 200, m_domain_time.x, m_domain_time.z ); 
    //
    //  m_domain_time.x  start                       (0ns)
    //  m_domain_time.y  end      getTimeMax()       (200ns ) 
    //  m_domain_time.z           getAnimTimeMax()   (previously 0.25*TimeMax as all fun in first 50ns)
    //
    m_animator->setModeRestrict(Animator::FAST);
    m_animator->Summary("Composition::gui setup Animation");
}

void Composition::nextMode(unsigned int modifiers)
{
    if(!m_animator) initAnimator() ;
    m_animator->nextMode(modifiers);
}

unsigned int Composition::tick()
{
    m_count++ ; 

    if(!m_animator) initAnimator();
    bool bump(false);
    m_animator->step(bump);

    m_view->tick();

    return m_count ; 
}



void Composition::gui()
{
#ifdef GUI_
    if (!ImGui::CollapsingHeader("Composition")) return ;


    if(ImGui::Button("home")) home();

    float* param = glm::value_ptr(m_param) ;
    ImGui::SliderFloat( "param.x", param + 0,  0.f, 1000.0f, "%0.3f", 2.0f);
    ImGui::SliderFloat( "param.y", param + 1,  0.f, 1.0f, "%0.3f", 2.0f );
    ImGui::SliderFloat( "z:alpha", param + 2,  0.f, 1.0f, "%0.3f");

    float* lpos = m_light->getPositionPtr() ;
    ImGui::SliderFloat3( "lightposition", lpos,  -2.0f, 2.0f, "%0.3f");

    float* ldir = m_light->getDirectionPtr() ;
    ImGui::SliderFloat3( "lightdirection", ldir,  -2.0f, 2.0f, "%0.3f");

    if(m_animator)
    {
         m_animator->gui("time (ns)", "%0.3f", 2.0f);

         float* target = m_animator->getTarget();
         ImGui::Text(" time (ns) * c (.299792458 m/ns) horizon : %10.3f m ", *target * SPEED_OF_LIGHT / 1000.f );
    }

    int* pick = glm::value_ptr(m_pick) ;
    ImGui::SliderInt( "pick.x", pick + 0,  1, 100 );  // modulo scale down
    ImGui::SliderInt( "pick.w", pick + 3,  0, 1e6 );  // single photon pick

    int* colpar = glm::value_ptr(m_colorparam) ;
    ImGui::SliderInt( "colorparam.x", colpar + 0,  0, NUM_COLORSTYLE  );  // record color mode
    ImGui::Text(" colorstyle : %s ", getColorStyleName()); 


    int* np = glm::value_ptr(m_nrmparam) ;
    ImGui::SliderInt( "nrmparam.x", np + 0,  0, 1  );  
    ImGui::Text(" (nrm) normals : %s ",  *(np + 1) == 0 ? "NOT flipped" : "FLIPPED" );   

    ImGui::Text("pick %d %d %d %d ",
       m_pick.x, 
       m_pick.y, 
       m_pick.z, 
       m_pick.w);

#endif    
}




void Composition::home()
{
    m_view->home();
    m_trackball->home();
}


unsigned int Composition::getWidth()
{
   return m_camera->getWidth();
}
unsigned int Composition::getHeight()
{
   return m_camera->getHeight();
}

unsigned int Composition::getPixelWidth()
{
   return m_camera->getPixelWidth();
}
unsigned int Composition::getPixelHeight()
{
   return m_camera->getPixelHeight();
}
unsigned int Composition::getPixelFactor()
{
   return m_camera->getPixelFactor();
}

void Composition::setSize(glm::uvec4 size)
{
    LOG(info) << "Composition::setSize "
              << " x " << size.x 
              << " y " << size.y 
              << " z " << size.z
              ; 
    setSize(size.x, size.y, size.z);
}
void Composition::setSize(unsigned int width, unsigned int height, unsigned int factor)
{
    m_camera->setSize(width/factor,height/factor);
    m_camera->setPixelFactor(factor);
}


void Composition::setTarget(unsigned int target)
{
    if(!m_scene)
    {
        LOG(warning) << "Composition::setTarget requires composition.setScene(scene) " ; 
        return ; 
    }
    m_scene->setTarget(target);
}

void Composition::setSelection(std::string selection)
{
    setSelection(givec4(selection));
}
void Composition::setSelection(glm::ivec4 selection) 
{
    m_selection = selection ;  
}


void Composition::setRecSelect(std::string recselect)
{
    setRecSelect(givec4(recselect));
}
void Composition::setRecSelect(glm::ivec4 recselect) 
{
    m_recselect = recselect ;  
}

void Composition::setPickPhoton(std::string pickphoton)
{
    setPickPhoton(givec4(pickphoton));
}



void Composition::setPickPhoton(glm::ivec4 pickphoton) 
{
   // currently this relies on photon/record data being downloaded to host

    m_pickphoton = pickphoton ;  
    if(m_pickphoton.x > 0)
    {
        print(m_pickphoton, "Composition::setPickPhoton single photon targetting");
        unsigned int photon_id = m_pickphoton.x ;
        NumpyEvt* evt = m_scene ? m_scene->getNumpyEvt() : NULL ; 
        RecordsNPY* recs = evt ? evt->getRecordsNPY() : NULL ; 
        if(recs)
        {
            glm::vec4 ce = recs->getCenterExtent(photon_id);
            print(ce, "Composition::setPickPhoton single photon center extent");
            setCenterExtent(ce);
        }
        PhotonsNPY* pho  = evt ? evt->getPhotonsNPY() : NULL ; 
        if(pho)
        {
            pho->dump(photon_id, "Composition::setPickPhoton");
        }
    }
}




void Composition::setPickFace(std::string pickface)
{
    setPickFace(givec4(pickface));
}


void Composition::setPickFace(glm::ivec4 pickface) 
{
    // gets called on recieving udp messages via boost bind done in CompositionCfg 
    m_pickface = pickface ;  
    if(m_pickface.x > 0)
    {
        print(m_pickface, "Composition::setPickFace face targetting");
        if(m_scene)
        {
            unsigned int face_index0= m_pickface.x ;
            unsigned int face_index1= m_pickface.y ;
            unsigned int solid_index= m_pickface.z ;
            unsigned int mesh_index = m_pickface.w ;

            //m_scene->setFaceTarget(face_index, solid_index, mesh_index);
            m_scene->setFaceRangeTarget(face_index0, face_index1, solid_index, mesh_index);
        }
        else
        {
            LOG(warning) << "Composition::setPickFace requires Scene lodged in Composition " ;
        }
    }
}



void Composition::setColorParam(std::string colorparam)
{
    setColorParam(givec4(colorparam));
}
void Composition::setColorParam(glm::ivec4 colorparam) 
{
    m_colorparam = colorparam ;  
}









void Composition::setFlags(std::string flags) 
{
    setFlags(givec4(flags));
}
void Composition::setFlags(glm::ivec4 flags) 
{
    m_flags = flags ;  
}


void Composition::setPick(std::string pick) 
{
    setPick(givec4(pick));
}
void Composition::setPick(glm::ivec4 pick) 
{
    m_pick = pick ;  
}






void Composition::setParam(std::string param)
{
    setParam(gvec4(param));
}
void Composition::setParam(glm::vec4 param)
{
    m_param = param ;
}
 
 
void Composition::setTimeDomain(gfloat4 td)
{
    m_domain_time = glm::vec4(td.x, td.y, td.z, td.w); 
}
void Composition::setColorDomain(gfloat4 cd)
{
    m_domain_color = glm::vec4(cd.x, cd.y, cd.z, cd.w); 
}

//void Composition::setLightPositionEye(gfloat4 lp)
//{
//    m_light_position_eye = glm::vec4(lp.x, lp.y, lp.z, lp.w); 
//}




void Composition::setDomainCenterExtent(gfloat4 ce)
{
    m_domain_center_extent.x = ce.x ;
    m_domain_center_extent.y = ce.y ;
    m_domain_center_extent.z = ce.z ;
    m_domain_center_extent.w = ce.w ;

    glm::vec3 sc(ce.w);
    glm::vec3 tr(ce.x, ce.y, ce.z);

    m_domain_isnorm = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);   
    // inverse signed normalize , ie letting coords out of signed unit box 
}

void Composition::setCenterExtent(glm::vec4& ce, bool autocam)
{
    setCenterExtent(gfloat4(ce.x,ce.y,ce.z,ce.w), autocam);
}

void Composition::setCenterExtent(gfloat4 ce, bool autocam) // replaces setModelToWorld
{  
    m_center_extent.x = ce.x ;
    m_center_extent.y = ce.y ;
    m_center_extent.z = ce.z ;
    m_center_extent.w = ce.w ;

    glm::vec3 sc(ce.w);
    glm::vec3 tr(ce.x, ce.y, ce.z);

    m_model_to_world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc); 
    m_extent = ce.w ; 


    if(autocam)
    {
        m_trackball->home();
        m_camera->setNear( m_extent/10.f ); 
        m_camera->setFar(  m_extent*20.f );  
    }
}
float Composition::getNear()
{
    return m_camera->getNear();
}
float Composition::getFar()
{
    return m_camera->getFar();
}




float* Composition::getWorld2EyePtr()
{
    return glm::value_ptr(m_world2eye);
}
float* Composition::getWorld2ClipPtr()
{
    return glm::value_ptr(m_world2clip);
}
float* Composition::getWorld2ClipISNormPtr()
{
    return glm::value_ptr(m_world2clip_isnorm);
}
float* Composition::getProjectionPtr()  
{
    return glm::value_ptr(m_projection) ;
}

float* Composition::getLightPositionPtr()  
{
    return glm::value_ptr(m_light_position) ;
}
float* Composition::getLightDirectionPtr()  
{
    return glm::value_ptr(m_light_direction) ;
}

float* Composition::getParamPtr()  
{
    return glm::value_ptr(m_param) ;
}

int* Composition::getNrmParamPtr()  
{
    return glm::value_ptr(m_nrmparam) ;
}

glm::ivec4& Composition::getNrmParam()  
{
    return m_nrmparam  ;
}




glm::mat4& Composition::getProjection()  
{
     return m_projection ;
}



float* Composition::getIdentityPtr()
{
    return glm::value_ptr(m_identity);
}




float* Composition::getClipPlanePtr()
{
    return glm::value_ptr(m_clipplane);
}
int Composition::getClipMode()
{
    return m_clipper->getMode();
}




glm::mat4& Composition::getWorld2Eye()  // ModelView
{
     return m_world2eye ;
}
glm::mat4& Composition::getEye2World()  // ModelViewInverse
{
     return m_eye2world ;
}
glm::mat4& Composition::getWorld2Camera()  // just view, no trackballing
{
     return m_world2camera ;
}
glm::mat4& Composition::getCamera2World()  // just view, no trackballing
{
     return m_camera2world ;
}
glm::vec4& Composition::getGaze()
{
     return m_gaze ; 
}
float& Composition::getGazeLength()
{
     return m_gazelength;
}
glm::mat4& Composition::getWorld2Clip()  // ModelViewProjection
{
     return m_world2clip ;
}
glm::mat4& Composition::getWorld2ClipISNorm() 
{
     return m_world2clip_isnorm ;
}




glm::mat4& Composition::getTrackballing()  
{
     assert(0);
     return m_trackballing ;
}
glm::mat4& Composition::getITrackballing()  
{
     assert(0);
     return m_itrackballing ;
}


glm::vec3 Composition::unProject(unsigned int x, unsigned int y, float z)
{
     glm::vec3 win(x,y,z);
     return glm::unProject(win, m_world2eye, m_projection, m_viewport);
}


bool Composition::hasChanged()
{
    return m_animator->isAnimating() || m_view->hasChanged() || m_camera->hasChanged() || m_trackball->hasChanged() ;
}

void Composition::setChanged(bool changed)
{
    m_view->setChanged(changed);
    m_camera->setChanged(changed);
    m_trackball->setChanged(changed);
}




void Composition::update()
{
    //   use like this:
    //
    //       if(!m_composition->hasChanged()) return   // dont bother updating renders, nothing changed
    //       m_composition->update()
    //
    //       proceed to trace/render/whatever using the new transforms
    //
    //
    //  Update matrices based on 
    //
    //      m_view
    //      m_camera
    //      m_trackball
    //
    // view inputs are in model coordinates (model coordinates are all within -1:1)
    // model_to_world matrix constructed from geometry center and extent
    // is used to construct the lookat matrix 
    //
    //   eye frame
    //       eye  (0,0,0)
    //       look (0,0,-m_gazelength) 
    //

    m_view->setChanged(false);
    m_camera->setChanged(false);
    m_trackball->setChanged(false);

    m_viewport = glm::vec4( 0.f, 0.f, getPixelWidth(), getPixelHeight() );


    m_view->getTransforms(m_model_to_world, m_world2camera, m_camera2world, m_gaze );   // model_to_world is input, the others are updated

    m_gazelength = glm::length(m_gaze);

    m_eye2look = glm::translate( glm::mat4(1.), glm::vec3(0,0,m_gazelength));  

    m_look2eye = glm::translate( glm::mat4(1.), glm::vec3(0,0,-m_gazelength));

    m_trackball->getOrientationMatrices(m_trackballrot, m_itrackballrot);
    m_trackball->getTranslationMatrices(m_trackballtra, m_itrackballtra);
    //m_trackball->getCombinedMatrices(m_trackballing, m_itrackballing);

    m_world2eye = m_trackballtra * m_look2eye * m_trackballrot * m_eye2look * m_world2camera ;           // ModelView

    m_eye2world = m_camera2world * m_look2eye * m_itrackballrot * m_eye2look * m_itrackballtra ;          // InverseModelView

    m_projection = m_camera->getProjection();

    m_world2clip = m_projection * m_world2eye ;    //  ModelViewProjection

    m_world2clip_isnorm = m_world2clip * m_domain_isnorm  ;   // inverse snorm (signed normalization)

    m_clipplane = m_clipper->getClipPlane(m_model_to_world) ;

    m_light_position = m_light->getPosition(m_model_to_world);

    m_light_direction = m_light->getDirection(m_model_to_world);


    //print(m_light_position, "Composition::update m_light_position");


    m_axis_data->setQuad(0,0,  m_light_position  );
    m_axis_data->setQuad(0,1,  m_axis_x );
    m_axis_data->setQuad(0,2,  m_axis_x_color );

    m_axis_data->setQuad(1,0,  m_light_position  );
    m_axis_data->setQuad(1,1,  m_axis_y );
    m_axis_data->setQuad(1,2,  m_axis_y_color );

    m_axis_data->setQuad(2,0,  m_light_position  );
    m_axis_data->setQuad(2,1,  m_axis_z );
    m_axis_data->setQuad(2,2,  m_axis_z_color );



/*
  //  env/geant4/geometry/collada/g4daeview/daetransform.py

 51     def _get_world2eye(self):
 52         """
 53         Objects are transformed from **world** space to **eye** space using GL_MODELVIEW matrix, 
 54         as daeviewgl regards model spaces as just input parameter conveniences
 55         that OpenGL never gets to know about those.  
 56 
 57         So need to invert MODELVIEW and apply it to the origin (eye position in eye space)
 58         to get world position of eye.  Can then convert that into model position.  
 59 
 60         Motivation:
 61 
 62            * determine effective view point (eye,look,up) after trackballing around
 ..
 88         return reduce(np.dot, [self.downscale,
 89                                self.trackball.translate,
 90                                self.view.translate_look2eye,   # (0,0,-distance)
 91                                self.trackball.rotation,
 92                                self.view.translate_eye2look,   # (0,0,+distance)
 93                                self.view.world2camera ])
 94     world2eye = property(_get_world2eye)   # this matches GL_MODELVIEW
 ..
 96     def _get_eye2world(self):
 97         return reduce(np.dot, [self.view.camera2world,
 98                                self.view.translate_look2eye,
 99                                self.trackball.rotation.T,
100                                self.view.translate_eye2look,
101                                self.trackball.untranslate,
102                                self.upscale])
103     eye2world = property(_get_eye2world)
*/


}



void Composition::initAxis()
{
    NPY<float>* axis_data = NPY<float>::make(3, 3, 4); // three axes x,y,z each with 3 float4 vpos,vdir,vcol 
    axis_data->fill(0.f);
    setAxisData(axis_data);
}

void Composition::setAxisData(NPY<float>* axis_data)
{
    m_axis_data = axis_data ;  
    m_axis_attr = new MultiViewNPY();
    //                                              j k sz   type          norm   iatt
    m_axis_attr->add(new ViewNPY("vpos",m_axis_data,0,0,4,ViewNPY::FLOAT, false, false));     
    m_axis_attr->add(new ViewNPY("vdir",m_axis_data,1,0,4,ViewNPY::FLOAT, false, false));     
    m_axis_attr->add(new ViewNPY("vcol",m_axis_data,2,0,4,ViewNPY::FLOAT, false, false));     
}


void Composition::dumpAxisData(const char* msg)
{
    AxisNPY ax(m_axis_data);
    ax.dump(msg);
}






void Composition::getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W)
{
   update();

   float tanYfov = m_camera->getTanYfov();
   float aspect = m_camera->getAspect();

   float v_half_height = m_gazelength * tanYfov ; 
   float u_half_width  = v_half_height * aspect ; 
   float w_depth       = m_gazelength ; 

   //  Eye frame axes and origin 
   //  transformed into world frame

   glm::vec4 right( 1., 0., 0., 0.);
   glm::vec4   top( 0., 1., 0., 0.);
   glm::vec4  gaze( 0., 0.,-1., 0.);

   glm::vec4 origin(0., 0., 0., 1.);

   // and scaled to focal plane dimensions 

   U = glm::vec3( m_eye2world * right ) * u_half_width ;  
   V = glm::vec3( m_eye2world * top   ) * v_half_height ;  
   W = glm::vec3( m_eye2world * gaze  ) * w_depth  ;  

   eye = glm::vec3( m_eye2world * origin );   

}

    
void Composition::getEyeUVW_no_trackball(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W)
{
   glm::vec3 e ;  
   glm::vec3 unorm ;  
   glm::vec3 vnorm ;  
   glm::vec3 gaze ;  

   m_view->getFocalBasis( m_model_to_world, e,unorm,vnorm, gaze );

   float tanYfov = m_camera->getTanYfov();
   float aspect = m_camera->getAspect();
   float v_half_height = glm::length(gaze) * tanYfov ; 
   float u_half_width  = v_half_height * aspect ; 

   eye = e ;
   U = unorm * u_half_width ; 
   V = vnorm * v_half_height ; 
   W = gaze ; 
}


void Composition::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_trackball->Summary("m_trackball");
    m_camera->Summary("m_camera");
    m_view->Summary("m_view");
}



void Composition::test_getEyeUVW()
{
    glm::vec3 eye ;
    glm::vec3 U ;
    glm::vec3 V ;
    glm::vec3 W ;

    printf("test_getEyeUVW\n");
    getEyeUVW(eye,U,V,W);

    print(eye, "eye");
    print(U,"U");
    print(V,"V");
    print(W,"W");

    printf("test_getEyeUVW (no trackball)\n");

    getEyeUVW_no_trackball(eye,U,V,W);
    print(eye, "eye");
    print(U,"U");
    print(V,"V");
    print(W,"W");

}


void Composition::Details(const char* msg)
{
    update();
    print(m_gaze, "m_gaze = look - eye ");
    print(m_eye2look, "m_eye2look translation");
    print(m_look2eye, "m_look2eye translation");
    print(m_world2eye, "m_world2eye ");
    print(m_eye2world, "m_eye2world ");
    print(m_projection, "m_projection ");
    print(m_world2clip, "m_world2clip");
    print(m_world2camera, "m_world2camera");
    print(m_camera2world, "m_camera2world");
    print(m_trackballing, "m_trackballing");
    print(m_itrackballing, "m_itrackballing");
}



