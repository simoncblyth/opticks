#ifdef _MSC_VER
// object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif


#include <climits>

#include <boost/math/constants/constants.hpp>
#include "BStr.hh"
#include "NGLM.hpp"
#include "SCtrl.hh"


// npy-

#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include "NPY.hpp"
#include "RecordsNPY.hpp"
#include "PhotonsNPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "AxisNPY.hpp"
#include "NState.hpp"


// okc-
#include "View.hh"
#include "ViewCfg.hh"
#include "Camera.hh"
#include "CameraCfg.hh"
#include "Trackball.hh"
#include "TrackballCfg.hh"
#include "Clipper.hh"
#include "ClipperCfg.hh"

#include "InterpolatedView.hh"
#include "OrbitalView.hh"
#include "TrackView.hh"

#include "Animator.hh"
#include "Light.hh"
#include "Bookmarks.hh"
#include "FlightPath.hh"

#include "OpticksConst.hh"
#include "OpticksEvent.hh"

#include "Composition.hh"
#include "CompositionCfg.hh"

#include "PLOG.hh"

const char* Composition::PREFIX = "composition" ;
const char* Composition::getPrefix()
{
   return PREFIX ; 
}


const char* Composition::PRINT = "print" ; 
const char* Composition::SELECT = "select" ; 
const char* Composition::RECSELECT = "recselect" ; 
const char* Composition::PICKPHOTON = "pickphoton" ; 
const char* Composition::EYEW = "eyew" ;
const char* Composition::LOOKW = "lookw" ;
const char* Composition::UPW = "upw" ;


const char* Composition::WHITE_ = "white" ; 
const char* Composition::MAT1_  = "mat1"; 
const char* Composition::MAT2_  = "mat2"; 
const char* Composition::FLAG1_ = "flag1"; 
const char* Composition::FLAG2_ = "flag2"; 
const char* Composition::POL1_  = "pol1"; 
const char* Composition::POL2_  = "pol2" ; 

const glm::vec3 Composition::X = glm::vec3(1.f,0.f,0.f) ;  
const glm::vec3 Composition::Y = glm::vec3(0.f,1.f,0.f) ;  
const glm::vec3 Composition::Z = glm::vec3(0.f,0.f,1.f) ;  
 

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
       case NUM_COLOR_STYLE :assert(0) ; break ; 
    }
    assert(0);
    return NULL ; 
}


const char* Composition::DEF_GEOMETRY_ = "default" ; 
const char* Composition::NRMCOL_GEOMETRY_ = "nrmcol" ; 
const char* Composition::VTXCOL_GEOMETRY_ = "vtxcol" ; 
const char* Composition::FACECOL_GEOMETRY_ = "facecol" ; 
 
const char* Composition::getGeometryStyleName(Composition::GeometryStyle_t style)
{
    switch(style)
    {
       case DEF_GEOMETRY:return DEF_GEOMETRY_ ; break ; 
       case NRMCOL_GEOMETRY:return NRMCOL_GEOMETRY_ ; break ; 
       case VTXCOL_GEOMETRY:return VTXCOL_GEOMETRY_ ; break ; 
       case FACECOL_GEOMETRY:return FACECOL_GEOMETRY_ ; break ; 
       case NUM_GEOMETRY_STYLE :assert(0) ; break ; 
    }
    assert(0);
    return NULL ; 
}










Composition::Composition()
  :
  m_lodcut(5000.f,10000.f,0.f,0.f),
  m_model2world(),
  m_world2model(),
  m_extent(1.0f),
  m_center_extent(),
  m_domain_center_extent(),
  m_domain_isnorm(),
  m_domain_time(),
  m_domain_color(),
  m_light_position(),
  m_light_direction(),
  m_pickphoton(0,0,0,0), 
  m_pickface(0,0,0,0), 
  m_recselect(), 
  m_colorparam(int(POL1),0,0,0), 
  m_selection(-INT_MAX,-INT_MAX,-INT_MAX,-INT_MAX),  // not 0, as that is liable to being meaningful
  m_flags(),
  m_pick( 1,0,0,0),      // initialize modulo scaledown to 1, 0 causes all invisible 
  m_pick_f(),
  m_param(1.f,0.030f,0.f,0.f),        // x: scaling of genstep DeltaPosition, y: vector length dfrac
  m_scanparam(0.f,1.0f,0.5f,0.01f),   // ct scan  x:clip-z-cut y:slice-width
  m_nrmparam(DEF_NORMAL,NRMCOL_GEOMETRY,0,0),
  m_animated(false),
  m_animator(NULL),
  m_rotator(NULL),
  m_camera(NULL),
  m_trackball(NULL),
  m_bookmarks(NULL),
  m_flightpath(NULL),
  m_view(NULL),
  m_standard_view(NULL),
  m_viewtype(View::STANDARD),
  m_animator_period(200),
  m_ovperiod(180),
  m_tvperiod(100),
  m_track(NULL), 
  m_light(NULL),
  m_clipper(NULL),
  m_count(0),
  m_axis_data(NULL),
  m_axis_attr(NULL),
  m_changed(true), 
  m_evt(NULL), 
  m_ctrl(NULL),
  m_lookphi(0.f), 
  m_axis_x(1000.f,    0.f,    0.f, 0.f),
  m_axis_y(0.f   , 1000.f,    0.f, 0.f),
  m_axis_z(0.f   ,    0.f, 1000.f, 0.f),
  m_axis_x_color(1.f,0.f,0.f,1.f),
  m_axis_y_color(0.f,1.f,0.f,1.f),
  m_axis_z_color(0.f,0.f,1.f,1.f),
  m_command_length(256),
  m_frame_position(0,0,0,0)
{
    init();
}


void Composition::setFramePosition(const glm::uvec4& position)
{
    m_frame_position = position ; 
}
glm::uvec4& Composition::getFramePosition()
{
    return m_frame_position ; 
}


Camera* Composition::getCamera()
{
    return m_camera ;
}

View* Composition::getView()
{
    return m_view ;
}
Light* Composition::getLight()
{
    return m_light ;
}
Trackball* Composition::getTrackball()
{
    return m_trackball ;
}
Clipper* Composition::getClipper()
{
    return m_clipper ;
}

void Composition::setCamera(Camera* camera)
{
    m_camera = camera ; 
}


void Composition::setBookmarks(Bookmarks* bookmarks)
{
    m_bookmarks = bookmarks ; 
}
void Composition::setFlightPath(FlightPath* flightpath)
{
    m_flightpath = flightpath ; 
}



OpticksEvent* Composition::getEvt()
{
    return m_evt ; 
}
void Composition::setEvt(OpticksEvent* evt)
{
    m_evt = evt ; 
}
void Composition::setCtrl(SCtrl* ctrl)
{
    m_ctrl = ctrl ; 
}





glm::vec4& Composition::getCenterExtent()
{
    return m_center_extent ; 
}
glm::vec4& Composition::getDomainCenterExtent()
{
    return m_domain_center_extent ; 
}
glm::vec4& Composition::getTimeDomain()
{
    return m_domain_time ; 
}
glm::uvec4& Composition::getColorDomain()
{
    return m_domain_color ; 
}
glm::vec4& Composition::getLightPosition()
{
    return m_light_position ; 
}
glm::vec4& Composition::getLightDirection()
{
    return m_light_direction ; 
}

void Composition::setOrbitalViewPeriod(int ovperiod)
{
    m_ovperiod = ovperiod ; 
}
void Composition::setAnimatorPeriod(int period)
{
    m_animator_period = period ; 
}


void Composition::setTrackViewPeriod(int tvperiod)
{
    m_tvperiod = tvperiod ; 
}
void Composition::setTrack(NPY<float>* track)
{
    m_track = track ; 
}






glm::mat4& Composition::getDomainISNorm()
{
    return m_domain_isnorm ; 
}


glm::ivec4& Composition::getPickPhoton()
{
    return m_pickphoton ; 
}

glm::ivec4& Composition::getPickFace()
{
    return m_pickface ; 
}



glm::ivec4& Composition::getRecSelect()
{
    return m_recselect ; 
}

glm::ivec4& Composition::getColorParam()
{
    return m_colorparam ; 
}

glm::ivec4& Composition::getSelection()
{
    return m_selection ; 
}

glm::ivec4& Composition::getFlags()
{
    return m_flags ; 
}
glm::ivec4& Composition::getPick()
{
    return m_pick; 
}
glm::vec4& Composition::getParam()
{
    return m_param ; 
}
glm::mat4& Composition::getModelToWorld()
{
    return m_model2world ; 
}
glm::mat4& Composition::getWorldToModel()
{
    return m_world2model ; 
}

const glm::vec4& Composition::getLODCut() const 
{
    return m_lodcut ; 
}




float Composition::getExtent()
{
    return m_extent ; 
}



unsigned int Composition::getCount()
{
    return m_count ; 
}

NPY<float>* Composition::getAxisData()
{
    return m_axis_data ; 
}

MultiViewNPY* Composition::getAxisAttr()
{
    return m_axis_attr ; 
}

void Composition::nextColorStyle()
{
    int next = (getColorStyle() + 1) % NUM_COLOR_STYLE ; 
    setColorStyle( (ColorStyle_t)next ) ; 
}



void Composition::nextNormalStyle()
{
    int next = (getNormalStyle() + 1) % NUM_NORMAL_STYLE ; 
    setNormalStyle( (NormalStyle_t)next ) ; 
}
void Composition::setNormalStyle(NormalStyle_t style)
{
    m_nrmparam.x = int(style) ;
}
Composition::NormalStyle_t Composition::getNormalStyle()
{
    return (NormalStyle_t)m_nrmparam.x ;
}




void Composition::nextGeometryStyle()
{
    int next = (getGeometryStyle() + 1) % NUM_GEOMETRY_STYLE ; 
    setGeometryStyle( (GeometryStyle_t)next ) ; 
}
void Composition::setGeometryStyle(GeometryStyle_t style)
{
    m_nrmparam.y = int(style) ;
}
Composition::GeometryStyle_t Composition::getGeometryStyle()
{
    return (GeometryStyle_t)m_nrmparam.y ;
}
const char* Composition::getGeometryStyleName()
{
    return Composition::getGeometryStyleName(getGeometryStyle());
}





void Composition::nextViewType(unsigned int /*modifiers*/)  // U KEY 
{
    int next = (getViewType() + 1) % View::NUM_VIEW_TYPE ; 
    setViewType( (View::View_t)next ) ; 
}

void Composition::setViewType(View::View_t type)
{
    m_viewtype = type ;
    applyViewType();
}

View::View_t Composition::getViewType()
{
    return m_viewtype ;
}



void Composition::nextPickPhotonStyle()
{
    int next = (getPickPhotonStyle() + 1) % NUM_PICKPHOTON_STYLE ; 
    setPickPhotonStyle( (PickPhotonStyle_t)next ) ; 
}
void Composition::setPickPhotonStyle(PickPhotonStyle_t style)
{
    m_pickphoton.y = int(style) ;
}
Composition::PickPhotonStyle_t Composition::getPickPhotonStyle()
{
    return (PickPhotonStyle_t)m_pickphoton.y ;
}



void Composition::setColorStyle(ColorStyle_t style)
{
    m_colorparam.x = int(style);
}
Composition::ColorStyle_t Composition::getColorStyle()
{
    return (ColorStyle_t)m_colorparam.x ; 
}

const char* Composition::getColorStyleName()
{
    return Composition::getColorStyleName(getColorStyle());
}




void Composition::setLookAngle(float phi)
{
    m_lookphi = phi ; 
}
float* Composition::getLookAnglePtr()
{
    return &m_lookphi ; 
}














void Composition::init()
{
   // hmm state is very different, maybe keep in the App not here 

    m_camera = new Camera() ;
    m_view   = new View() ;
    m_trackball = new Trackball() ;
    m_clipper = new Clipper() ;

    m_light = new Light() ;
    m_command.resize(m_command_length);

    initAxis();
}

void Composition::addConstituentConfigurables(NState* state)
{
    state->addConfigurable(m_trackball);
    state->addConfigurable(m_view);
    state->addConfigurable(m_camera);
    state->addConfigurable(m_clipper);
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
    else if(strcmp(name,LOOKW)==0) setLookW(s);
    else if(strcmp(name,EYEW)==0) setEyeW(s);
    else if(strcmp(name,UPW)==0) setUpW(s);
    else
         printf("Composition::set bad name %s\n", name);
}

std::string Composition::get(const char* name)
{
   std::string s ; 

   if(     strcmp(name,SELECT)==0)    s = gformat(getSelection()) ;
   else if(strcmp(name,RECSELECT)==0) s = gformat(getRecSelect()) ;
   else if(strcmp(name,PICKPHOTON)==0) s = gformat(getPickPhoton()) ;
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

void Composition::configureF(const char*, std::vector<float>  )
{
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

void Composition::addConfig(BCfg* cfg)
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

    m_animator = new Animator(target, m_animator_period, m_domain_time.x, m_domain_time.z ); 
    //
    //  m_domain_time.x  start                       (0ns)
    //  m_domain_time.y  end      getTimeMax()       (200ns ) 
    //  m_domain_time.z           getAnimTimeMax()   (previously 0.25*TimeMax as all fun in first 50ns)
    //
    m_animator->setModeRestrict(Animator::FAST);
    m_animator->Summary("Composition::gui setup Animation");
}


Animator* Composition::getAnimator()
{
    if(!m_animator) initAnimator() ;
    return m_animator ; 
}


void Composition::initRotator()
{
    m_rotator = new Animator(&m_lookphi, 180, -180.f, 180.f ); 
    m_rotator->setModeRestrict(Animator::NORM);  // only OFF and SLOW 
    m_rotator->Summary("Composition::initRotator");
}


void Composition::nextAnimatorMode(unsigned modifiers)
{
    if(!m_animator) initAnimator() ;
    m_animator->nextMode(modifiers);
}

void Composition::nextRotatorMode(unsigned modifiers)
{
    if(!m_rotator) initRotator();
    m_rotator->nextMode(modifiers);
}


void Composition::nextViewMode(unsigned int modifiers)    // T KEY
{
    if(m_view->isStandard())
    {
       LOG(info) << "Composition::nextViewMode(KEY_T) does nothing in standard view, switch to alt views with U:nextViewType " ; 
       return ;
    }

    if(OpticksConst::isShiftOption(modifiers)) 
    {
        LOG(info) << " SHIFT+OPTION+T : resetting view " ;  
        m_view->reset() ;
    }
    else
    {
        m_view->nextMode(modifiers);    
    } 
}


OrbitalView* Composition::makeOrbitalView()
{
    View* basis = m_view->isStandard() ? m_view : m_standard_view ; 
    bool verbose = true ; 
    return new OrbitalView(basis, m_ovperiod, verbose );
}

TrackView* Composition::makeTrackView()
{
    bool verbose = true ; 
    return m_track ? new TrackView(m_track, m_tvperiod, verbose ) : NULL ;
}






void Composition::applyViewType() // invoked by nextViewType/setViewType
{
    LOG(warning) << "Composition::applyViewType(KEY_U) switching " << View::TypeName(m_viewtype) ; 
    if(m_viewtype == View::STANDARD)
    {
        resetView();
    }
    else if(m_viewtype == View::FLIGHTPATH)
    {
        assert(m_flightpath);

        m_flightpath->refreshInterpolatedView();

        InterpolatedView* iv = m_flightpath->getInterpolatedView();     
        if(!iv)
        {
            LOG(warning) << "Composition::changeView"
                         << " FAILED "
                         << " FLIGHTPATH interpolated view requires at least 2 views " 
                         ;
            return  ;
        }

        iv->Summary("Composition::changeView(KEY_U)");

        setView(iv); 
    }
    else if(m_viewtype == View::INTERPOLATED)
    {
        assert(m_bookmarks);

        m_bookmarks->refreshInterpolatedView();

        InterpolatedView* iv = m_bookmarks->getInterpolatedView();     
        if(!iv)
        {
            LOG(warning) << "Composition::changeView"
                         << " FAILED "
                         << " interpolated view requires at least 2 bookmarks " 
                         ;
            return  ;
        }

        iv->Summary("Composition::changeView(KEY_U)");

        setView(iv); 

    }
    else if(m_viewtype == View::ORBITAL)
    {
        OrbitalView* ov = makeOrbitalView();
        setView(ov);
    }
    else if(m_viewtype == View::TRACK)
    {
        TrackView* tv = makeTrackView();
        if(tv == NULL)
        {
            LOG(warning) << "Composition::applyViewType(KEY_U) requires track information ";
            return ;
        }

        //bool external = false ; 
        bool external = true ; 
        if(external)
        { 
           // idea is to tie the TrackView interpolated position with the event animator
           // but needs debug : scoots view off to infinity very quickly 
            Animator* animator = getAnimator();
            tv->setAnimator(animator);    
        }

        setView(tv);
    } 


}

void Composition::setView(View* view)
{
    if(m_view->isStandard())
    {
        m_standard_view = m_view ;   // keep track of last standard view
    }
    m_view = view ; 
}

void Composition::resetView()
{
    if(!m_standard_view)
    {
        LOG(warning) << "Composition::resetView NULL standard_view" ;
        return ; 
    }
    m_view = m_standard_view ; 
}


unsigned int Composition::tick()
{
    m_count++ ; 

    if(!m_animator) initAnimator();
    if(!m_rotator)  initRotator();

    bool bump(false);
    m_animator->step(bump);
    m_rotator->step(bump);

    m_view->tick();   // does nothing for standard view, animates for altview

    return m_count ; 
}





std::string Composition::getEyeString()
{
    glm::vec4 eye  = m_view->getEye(m_model2world);
    return gformat(eye) ;
}

std::string Composition::getLookString()
{
    glm::vec4 look  = m_view->getLook(m_model2world);
    return gformat(look) ;
}

std::string Composition::getGazeString()
{
    glm::vec4 gaze  = m_view->getGaze(m_model2world);
    return gformat(gaze) ;
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

void Composition::setSize(const glm::uvec4& size)
{
    LOG(debug) << "Composition::setSize "
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


void Composition::setSelection(std::string selection)
{
    glm::ivec4 sel = givec4(selection);
    setSelection(sel);
}
void Composition::setSelection(const glm::ivec4& selection) 
{
    m_selection = selection ;  
}


void Composition::setRecSelect(std::string recselect)
{
    glm::ivec4 sel = givec4(recselect);
    setRecSelect(sel);
}
void Composition::setRecSelect(const glm::ivec4& recselect) 
{
    m_recselect = recselect ;  
}

void Composition::setPickPhoton(std::string pickphoton)
{
    setPickPhoton(givec4(pickphoton));
}



void Composition::setPickPhoton(const glm::ivec4& pickphoton) 
{
   // currently this relies on photon/record data being downloaded to host

    m_pickphoton = pickphoton ;  
    if(m_pickphoton.x > 0)
    {
        print(m_pickphoton, "Composition::setPickPhoton single photon targetting");
        unsigned int photon_id = m_pickphoton.x ;

        RecordsNPY* recs = m_evt ? m_evt->getRecordsNPY() : NULL ; 
        if(recs)
        {
            glm::vec4 ce = recs->getCenterExtent(photon_id);
            print(ce, "Composition::setPickPhoton single photon center extent");
            setCenterExtent(ce);
        }

        PhotonsNPY* pho  = m_evt ? m_evt->getPhotonsNPY() : NULL ; 
        if(pho)
        {
            pho->dump(photon_id, "Composition::setPickPhoton");
        }
    }
}






void Composition::setPickFace(const glm::ivec4& pickface) 
{
    m_pickface = pickface ;  
    // the aiming is done by GGeo::setPickFace
}






void Composition::setColorParam(std::string colorparam)
{
    glm::ivec4 cp = givec4(colorparam);
    setColorParam(cp);
}
void Composition::setColorParam(const glm::ivec4& colorparam) 
{
    m_colorparam = colorparam ;  
}


void Composition::setFlags(std::string flags) 
{
    glm::ivec4 fl = givec4(flags);
    setFlags(fl);
}
void Composition::setFlags(const glm::ivec4& flags) 
{
    m_flags = flags ;  
}


void Composition::setPick(std::string pick) 
{
    glm::ivec4 pk = givec4(pick);
    setPick(pk);
}
void Composition::setPick(const glm::ivec4& pick) 
{
    m_pick = pick ;  
}


void Composition::setParam(std::string param)
{
    glm::vec4 pm = gvec4(param);
    setParam(pm);
}
void Composition::setParam(const glm::vec4& param)
{
    m_param = param ;
}
 
 
void Composition::setTimeDomain(const glm::vec4& td)
{
    m_domain_time = glm::vec4(td.x, td.y, td.z, td.w); 
}

#ifdef GVECTOR
void Composition::setColorDomain(guint4 cd)
{
    m_domain_color = glm::uvec4(cd.x, cd.y, cd.z, cd.w); 
}
#endif


void Composition::setColorDomain(const glm::uvec4& cd)
{
    m_domain_color = glm::uvec4(cd.x, cd.y, cd.z, cd.w); 
}






void Composition::setDomainCenterExtent(const glm::vec4& ce)
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

void Composition::setCenterExtent(const glm::vec4& ce, bool autocam)
{
    // this is invoked by App::uploadGeometry/Scene::setTarget

    m_center_extent.x = ce.x ;
    m_center_extent.y = ce.y ;
    m_center_extent.z = ce.z ;
    m_center_extent.w = ce.w ;

    glm::vec4 ce_(ce.x,ce.y,ce.z,ce.w);
    glm::vec3 sc(ce.w);
    glm::vec3 tr(ce.x, ce.y, ce.z);

    glm::vec3 isc(1.f/ce.w);

    m_model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc); 
    m_world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);   // see tests/CompositionTest.cc


    LOG(info) << "Composition::setCenterExtent"
              << " ce " << gformat(m_center_extent) 
              ;

    LOG(debug) << "Composition::setCenterExtent"
              << " model2world " << gformat(m_model2world)
              << " world2model " << gformat(m_world2model)
              ;

    m_extent = ce.w ; 


    update();

    if(autocam)
    {
        aim(ce_);
    }
}



void Composition::setLookW(std::string lookw)
{
    setLookW(gvec4(lookw));
}
void Composition::setEyeW(std::string eyew)
{
    setEyeW(gvec4(eyew));
}
void Composition::setUpW(std::string upw)
{
    setUpW(gvec4(upw));
}






void Composition::setLookW(const glm::vec4& lookw)
{
    glm::vec4 _lookw(lookw);
    _lookw.w = 1.0f ; 

    glm::vec4 look = m_world2model * _lookw ; 

    LOG(debug) << "Composition::setLookW" 
               << " world2model " << gformat(m_world2model) 
               << " lookw: " << gformat(lookw)
               << " look: " << gformat(look)
               ;
               
    m_view->setLook(look);
}

void Composition::setEyeW(const glm::vec4& eyew)
{
    glm::vec4 _eyew(eyew);
    _eyew.w = 1.0f ; 
    glm::vec4 eye = m_world2model * _eyew ; 

    LOG(debug) << "Composition::setEyeW" 
               << " world2model " << gformat(m_world2model) 
               << " eyew: " << gformat(eyew)
               << " eye: " << gformat(eye)
               ;


    m_view->setEye(eye);
}

void Composition::setUpW(const glm::vec4& upw)
{
    glm::vec4 _upw(upw);
    _upw.w = 0.0f ; 
    _upw *= m_extent ; // scale length of up vector

    glm::vec4 up = m_world2model * _upw ; 
    glm::vec4 upn = glm::normalize(up) ;

    LOG(info)  << "Composition::setUpW" 
               << " world2model " << gformat(m_world2model) 
               << " upw: " << gformat(upw)
               << " up: " << gformat(up)
               << " upn: " << gformat(upn)
               ;


    m_view->setUp(upn);
}









void Composition::setEyeGUI(const glm::vec3& gui)
{
    glm::vec4 eyew ;

    eyew.x = m_extent*gui.x ;  
    eyew.y = m_extent*gui.y ;  
    eyew.z = m_extent*gui.z ;  
    eyew.w = 1.0f ;  

    glm::vec4 eye = m_world2model * eyew ; 

    LOG(info) << "Composition::setEyeGUI extent " << m_extent  ; 
    print(eyew, "eyew");
    print(eye, "eye");

    m_view->setEye(eye); // model frame

}


void Composition::setEyeX(float _x)
{
    m_view->setEyeX(_x);
}
void Composition::setEyeY(float _y)
{
    m_view->setEyeY(_y);
}
void Composition::setEyeZ(float _z)
{
    m_view->setEyeZ(_z);
}




void Composition::aim(glm::vec4& ce, bool verbose)
{
     if(verbose)
     {
         print(ce, "Composition::aim ce (world frame)"); 
         print(m_model2world, "Composition::aim m2w");

         glm::vec4 eye  = m_view->getEye(m_model2world); // world frame
         glm::vec4 look = m_view->getLook(m_model2world);
         glm::vec4 gaze = m_view->getGaze(m_model2world);

         print(eye,  "Composition::aim eye ");
         print(look, "Composition::aim look ");
         print(gaze, "Composition::aim gaze ");
         print(m_gaze, "Composition::aim m_gaze");
     }

     float basis = m_gazelength ;   // gazelength basis matches raygen in the ray tracer, so OptiX and OpenGL renders match
     //float basis = m_extent ; 

     m_camera->aim(basis);
}




bool Composition::getParallel()
{
    return m_camera->getParallel();
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

int* Composition::getPickPtr()  
{
    return glm::value_ptr(m_pick) ;
}
int* Composition::getColorParamPtr()  
{
    return glm::value_ptr(m_colorparam) ;
}




float* Composition::getScanParamPtr()  
{
    return glm::value_ptr(m_scanparam) ;
}
glm::vec4& Composition::getScanParam()  
{
    return m_scanparam  ;
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




bool Composition::hasChangedGeometry()
{
    // used from App::render to determine if a trace needs to be done

    bool view_changed(false) ;

    if(m_view->isTrack())
    {
         TrackView* tv = dynamic_cast<TrackView*>(m_view) ;
         view_changed = tv->hasChanged();
    } 
    else if(m_view->isOrbital())
    {
         OrbitalView* ov = dynamic_cast<OrbitalView*>(m_view) ;
         view_changed = ov->hasChanged();
    }
    else if(m_view->isInterpolated())
    {
         InterpolatedView* iv = dynamic_cast<InterpolatedView*>(m_view) ;
         view_changed = iv->hasChanged();
    }
    else if(m_view->isStandard())
    {
         view_changed = m_view->hasChanged();
    }
    return m_rotator->isActive() || view_changed || m_camera->hasChanged() || m_trackball->hasChanged() ;
}


bool Composition::hasChanged()
{
    return m_rotator->isActive() || m_animator->isActive() || m_view->hasChanged() || m_camera->hasChanged() || m_trackball->hasChanged() ;
}

void Composition::setChanged(bool changed)
{
    m_view->setChanged(changed);
    m_camera->setChanged(changed);
    m_trackball->setChanged(changed);
}





glm::vec4 Composition::transformWorldToEye(const glm::vec4& world) const 
{
    return m_world2eye * world ; 
}


glm::vec4 Composition::transformEyeToWorld(const glm::vec4& eye) const 
{
    //  m_world2eye/m_eye2world incorporates the trackballing and rotation
    return m_eye2world * eye ; 
}
glm::vec4 Composition::getViewpoint()
{
    glm::vec4 viewpoint_eye(0.f, 0.f, 0.f, 1.f ); // viewpoint, ie position of observer (origin in eye frame)
    return transformEyeToWorld(viewpoint_eye);
}
glm::vec4 Composition::getLookpoint()   
{
    glm::vec4 lookpoint_eye(0.f, 0.f, -m_gazelength, 1.f ); // lookpoint observed, looking towards -z in eye frame
    return transformEyeToWorld(lookpoint_eye);
}
glm::vec4 Composition::getUpdir()
{
    glm::vec4 updir_eye(0.f, 1.f, 0.f, 0.f );  // convention: y is up in eye frame, x to the right, z outwards (RHS)
    return transformEyeToWorld(updir_eye);
}


void Composition::commitView()
{
    // This is invoked by pressing shift+number_key (0-9) 
    // whilst currently at that bookmark. 
    // This can be used to update the bookmark
    //
    // fold trackball/rotator changes to viewpoint into the View
    // and home the trackball and rotator
    // without changing the rendered view 
    //
    //  TODO:
    //     updir coming out 0,0,0 and having to change in the .ini
    //     NState::apply view.up change 0.0000,0.0000,1.0000 --> 0.0000,0.0000,0.0000
    //     
    //

    glm::vec4 viewpoint = getViewpoint();
    glm::vec4 lookpoint = getLookpoint();
    glm::vec4 updir = getUpdir();

    LOG(info) << "Composition::commitView " 
              << " viewpoint " << gformat(viewpoint)
              << " lookpoint " << gformat(lookpoint)
              << " updir " << gformat(updir)
              ; 

    setEyeW(viewpoint);
    setLookW(lookpoint);
    setUpW(updir);

    m_trackball->home();
    m_rotator->home();
    //assert(m_lookphi == 0.f );

    update();
}



void Composition::home()
{
    //m_view->home();
    m_trackball->home();
    m_rotator->home();

    setLookAngle(0.f);
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
    // model2world matrix constructed from geometry center and extent
    // is used to construct the lookat matrix 
    //
    //   eye frame : centered on the eye
    //       eye  (0,0,0)
    //       look (0,0,-m_gazelength) 
    //
    //   look frame : centered on where looking
    //       look (0,0,0) 
    //       eye  (0,0,m_gazelength)
    //

    m_view->setChanged(false);
    m_camera->setChanged(false);
    m_trackball->setChanged(false);

    m_viewport = glm::vec4( 0.f, 0.f, getPixelWidth(), getPixelHeight() );

    float pi = boost::math::constants::pi<float>() ;
    m_lookrotation = glm::rotate(glm::mat4(1.f), m_lookphi*float(pi)/180.f , Y );
    m_ilookrotation = glm::transpose(m_lookrotation);



    m_view->getTransforms(m_model2world, m_world2camera, m_camera2world, m_gaze );   // model2world is input, the others are updated

    // the eye2look look2eye pair allows trackball rot to be applied around the look 
    m_gazelength = glm::length(m_gaze);
    m_eye2look = glm::translate( glm::mat4(1.), glm::vec3(0,0,m_gazelength));  
    m_look2eye = glm::translate( glm::mat4(1.), glm::vec3(0,0,-m_gazelength));

    m_trackball->getOrientationMatrices(m_trackballrot, m_itrackballrot);
    m_trackball->getTranslationMatrices(m_trackballtra, m_itrackballtra);

    m_world2eye = m_trackballtra * m_look2eye * m_trackballrot * m_lookrotation * m_eye2look * m_world2camera ;           // ModelView

    m_eye2world = m_camera2world * m_look2eye * m_ilookrotation * m_itrackballrot * m_eye2look * m_itrackballtra ;          // InverseModelView

    m_projection = m_camera->getProjection();

    m_world2clip = m_projection * m_world2eye ;    //  ModelViewProjection


    /*
    LOG(info) << "Composition::update"
              << " m_world2eye " << gformat(m_world2eye)
              << " m_projection " << gformat(m_projection)
              << " m_world2clip " << gformat(m_world2clip)
              ;

    */

    m_world2clip_isnorm = m_world2clip * m_domain_isnorm  ;   // inverse snorm (signed normalization)

    m_clipplane = m_clipper->getClipPlane(m_model2world) ;

    m_light_position = m_light->getPosition(m_model2world);

    m_light_direction = m_light->getDirection(m_model2world);


    //print(m_light_position, "Composition::update m_light_position");


    m_axis_data->setQuad(m_light_position, 0,0 );
    m_axis_data->setQuad(m_axis_x        , 0,1 );
    m_axis_data->setQuad(m_axis_x_color  , 0,2 );

    m_axis_data->setQuad(m_light_position, 1,0 );
    m_axis_data->setQuad(m_axis_y        , 1,1 );
    m_axis_data->setQuad(m_axis_y_color  , 1,2 );

    m_axis_data->setQuad(m_light_position, 2,0 );
    m_axis_data->setQuad(m_axis_z        , 2,1 );
    m_axis_data->setQuad(m_axis_z_color  , 2,2 );



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
    m_axis_attr = new MultiViewNPY("axis_attr");
    //                                              j k l sz   type          norm   iatt  item_from_dim
    m_axis_attr->add(new ViewNPY("vpos",m_axis_data,0,0,0,4,ViewNPY::FLOAT, false, false, 1));     
    m_axis_attr->add(new ViewNPY("vdir",m_axis_data,1,0,0,4,ViewNPY::FLOAT, false, false, 1));     
    m_axis_attr->add(new ViewNPY("vcol",m_axis_data,2,0,0,4,ViewNPY::FLOAT, false, false, 1));     
}


void Composition::dumpAxisData(const char* msg)
{
    AxisNPY ax(m_axis_data);
    ax.dump(msg);
}

glm::vec3 Composition::getNDC(const glm::vec4& position_world) const 
{
    const glm::vec4 position_eye = transformWorldToEye(position_world);
    glm::mat4 projection = m_camera->getProjection();
    glm::vec4 ndc = projection * position_eye ; 
    return glm::vec3(ndc.x/ndc.w, ndc.y/ndc.w, ndc.z/ndc.w) ; 
}

glm::vec3 Composition::getNDC2(const glm::vec4& position_world) const 
{
    glm::vec4 ndc = m_world2clip * position_world ; 
    return glm::vec3(ndc.x/ndc.w, ndc.y/ndc.w, ndc.z/ndc.w) ; 
}


void Composition::dumpFrustum(const char* msg) const 
{
    LOG(info) << msg ; 
    std::vector<glm::vec4> world ;
    std::vector<std::string> labels ;
    m_camera->getFrustumVert(world, labels) ;
    dumpPoints(world, labels);
}

void Composition::dumpCorners(const char* msg) const 
{   
    LOG(info) << msg ; 
    std::vector<glm::vec4> world ;
    std::vector<std::string> labels ;
    getCorners(world, labels);     
    dumpPoints(world, labels);
}


void Composition::getCorners(std::vector<glm::vec4>& corners, std::vector<std::string>& labels) const 
{
    const glm::vec4& ce = m_center_extent ; 
    corners.push_back( glm::vec4( ce.x       , ce.y        , ce.z       , 1.0 ) );
    labels.push_back("center");
    for(unsigned i=0 ; i < 8 ; i++)
    {
        corners.push_back( 
            glm::vec4( 
              i & 1 ? ce.x + ce.w : ce.x - ce.w ,  
              i & 2 ? ce.y + ce.w : ce.y - ce.w ,
              i & 4 ? ce.z + ce.w : ce.z - ce.w ,
              1.f
            ));
        labels.push_back(BStr::concat<int>("corner-", i, NULL));
    }
}




void Composition::dumpPoints(const std::vector<glm::vec4>& world, const std::vector<std::string>& labels) const 
{
   for(unsigned i=0 ; i < world.size() ; i++)
   {
       const glm::vec4& wpos = world[i] ;

       glm::vec3 ndc = getNDC(wpos) ;
       glm::vec3 ndc2 = getNDC2(wpos) ;

       std::cout
               << "(" << std::setw(2) << i << ") " 
               << labels[i] << std::endl 
               << gpresent("world",  wpos )
               << gpresent("model",  m_world2model  * wpos )
               << gpresent("camera", m_world2camera * wpos )
               << gpresent("eye",    m_world2eye * wpos )
               << gpresent("clip",   m_world2clip * wpos )
               << gpresent("ndc",    ndc )
               << gpresent("ndc2",   ndc2 )
               << std::endl
               ;

   }
}












float Composition::getNDCDepth(const glm::vec4& position_world)
{
    const glm::vec4 position_eye = transformWorldToEye(position_world);
    float eyeDist = position_eye.z ;   // from eye frame definition, this is general
    assert(eyeDist <= 0.f ); 

    bool parallel = m_camera->getParallel() ;
    
    // un-homogenizing divides by -z in perspective case

    glm::vec4 zproj ; 
    m_camera->fillZProjection(zproj);

    float ndc_z = parallel ?
                              zproj.z*eyeDist + zproj.w
                           :
                             -zproj.z - zproj.w/eyeDist
                           ;     


    if(0)
    LOG(debug) << "Composition::getNDCDepth"
              << " p_world " << gformat(position_world)
              << " p_eye " << gformat(position_eye)
              << " eyeDist " << eyeDist 
              << " zproj " << gformat(zproj)
              << " ndc_z " << ndc_z
              << " proj " << gformat(m_projection)
              ;




    // range -1:1 for visibles
    return ndc_z ; 
}

float Composition::getClipDepth(const glm::vec4& position_world)
{
    return getNDCDepth(position_world)*0.5f + 0.5f ; 
}





void Composition::getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W, glm::vec4& ZProj )
{
   update();

   float tanYfov = m_camera->getTanYfov();
   float aspect = m_camera->getAspect();

   m_camera->fillZProjection(ZProj); // 3rd row of projection matrix

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

   m_view->getFocalBasis( m_model2world, e,unorm,vnorm, gaze );

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
    glm::vec4 ZProj ;


    printf("test_getEyeUVW\n");
    getEyeUVW(eye,U,V,W,ZProj);

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
    LOG(info) << msg ; 
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


    std::cout << gpresent( "w2e(MV)", m_world2eye ) << std::endl ; 
    std::cout << gpresent( "w2c(MVP)", m_world2clip ) << std::endl ; 

}



