#include <cstdio>
#include <cstring>
#include <sstream>

// brap-
#include "BFile.hh"
#include "BDir.hh"
#include "PLOG.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"

// optickscore-
#include "OpticksConst.hh"
#include "InterpolatedView.hh"
#include "FlightPath.hh"



const char* FlightPath::FILENAME = "flightpath.npy"  ; 

void FlightPath::setCtrl(SCtrl* ctrl)
{
    m_ctrl = ctrl ; 
}

FlightPath::FlightPath(const char* dir)  
    :
    m_flightpathdir(dir ? strdup(dir) : NULL),
    m_flightpath(NULL),
    m_view(NULL),
    m_verbose(false),
    m_ivperiod(128),
    m_ctrl(NULL)
{
}

int* FlightPath::getIVPeriodPtr()
{
    return &m_ivperiod ; 
}
unsigned FlightPath::getNumViews() const 
{
    return m_flightpath->getNumItems(); 
}
void FlightPath::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}
void FlightPath::setInterpolatedViewPeriod(unsigned int ivperiod)
{
    m_ivperiod = ivperiod ; 
}

void FlightPath::load()
{
    std::string path = BFile::FormPath(m_flightpathdir, FILENAME ) ;
    LOG(info) << " path " << path ; 
    delete m_flightpath ; 
    m_flightpath = NPY<float>::load(path.c_str()) ; 
    assert( m_flightpath ) ; 
}


std::string FlightPath::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " << ( m_flightpathdir ? m_flightpathdir : "NULL" )  ; 
    return ss.str();
}

void FlightPath::Summary(const char* msg)
{
    LOG(info) << description(msg);
}

InterpolatedView* FlightPath::makeInterpolatedView()
{
    load(); 
    assert( m_flightpath ) ; 
    return InterpolatedView::MakeFromArray( m_flightpath, m_ivperiod, m_ctrl  ) ; 
}


void FlightPath::refreshInterpolatedView()
{
    delete m_view ; 
    m_view = NULL ; 
}

InterpolatedView* FlightPath::getInterpolatedView()
{
    if(!m_view) m_view = makeInterpolatedView();
    return m_view ;             
}


