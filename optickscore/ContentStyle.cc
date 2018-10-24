#include <cassert>
#include "PLOG.hh"
#include "ContentStyle.hh"

ContentStyle::ContentStyle()
    :
    m_content_style(ASIS),
    m_num_content_style(0),
    m_inst(false),
    m_bbox(false), 
    m_wire(false),
    m_asis(false)
{
}


const char* ContentStyle::ASIS_ = "asis" ; 
const char* ContentStyle::BBOX_ = "bbox" ; 
const char* ContentStyle::NORM_ = "norm" ;   
const char* ContentStyle::NONE_ = "none" ;   
const char* ContentStyle::WIRE_ = "wire" ; 
const char* ContentStyle::NORM_BBOX_ = "norm_bbox" ; 


bool ContentStyle::isInst() const { return m_inst ; }
bool ContentStyle::isBBox() const { return m_bbox ; }
bool ContentStyle::isWire() const { return m_wire ; }
bool ContentStyle::isASIS() const { return m_asis ; }


unsigned int ContentStyle::getNumContentStyle()
{
    return m_num_content_style == 0 ? int(NUM_CONTENT_STYLE) : m_num_content_style ;
}
void ContentStyle::setNumContentStyle(unsigned num_content_style)
{
    m_num_content_style = num_content_style ;
    dumpContentStyles("ContentStyle::setNumContentStyle");
}

void ContentStyle::dumpContentStyles(const char* msg)
{
    const ContentStyle_t style0 = getContentStyle() ;
    ContentStyle_t style = style0 ; 

    LOG(info) << msg << " (ContentStyle::dumpContentStyles) " ; 

    while( style != style0 )
    {
        nextContentStyle();
        style = getContentStyle() ;
    }
    assert( style == style0 );
}

ContentStyle::ContentStyle_t ContentStyle::getContentStyle() const 
{
    return m_content_style ;
}

void ContentStyle::nextContentStyle()
{
    unsigned num_content_style = getNumContentStyle() ;
    int next = (m_content_style + 1) % num_content_style ; 
    setContentStyle( (ContentStyle_t)next );
}


void ContentStyle::command(const char* cmd) 
{ 
    LOG(info) << cmd ; 

    if(strlen(cmd) != 2 ) return ; 
    if( cmd[0] != 'B')    return ; 

    std::string allowed("012345") ; 
    if(allowed.find(cmd[1]) == std::string::npos) return ; 

    
    ContentStyle_t style = ASIS ; 
    switch( cmd[1] )
    {
        case '0': style = ASIS ; break ; 
        case '1': style = BBOX ; break ; 
        case '2': style = NORM ; break ; 
        case '3': style = NONE ; break ; 
        case '4': style = WIRE ; break ; 
        case '5': style = NORM_BBOX ; break ; 
    }
    setContentStyle(style); 
}


void ContentStyle::setContentStyle(ContentStyle_t style)
{
    m_content_style = style ; 
    applyContentStyle();
    LOG(fatal) << desc() ; 
}

const char* ContentStyle::getContentStyleName(ContentStyle::ContentStyle_t style) 
{
   const char* s = NULL ; 
   switch(style)
   {
      case ASIS: s = ASIS_ ; break; 
      case BBOX: s = BBOX_ ; break; 
      case NORM: s = NORM_ ; break; 
      case NONE: s = NONE_ ; break; 
      case WIRE: s = WIRE_ ; break; 
      case NORM_BBOX: s = NORM_BBOX_ ; break; 
      case NUM_CONTENT_STYLE: s = NULL ; break ; 
      default:                s = NULL ; break ; 
   } 
   assert(s); 
   return s ; 
}

const char* ContentStyle::getContentStyleName() const 
{
   return getContentStyleName(m_content_style);
}


// NB this just holds state, the state needs to be acted upon by oglrap.Scene

void ContentStyle::applyContentStyle()  // B:key 
{
    switch(m_content_style)
    {
      case ContentStyle::ASIS:
             m_asis = true ; 
             break; 
      case ContentStyle::BBOX:
             m_asis = false ; 
             m_inst = false ; 
             m_bbox = true ; 
             m_wire = false ; 
             break;
      case ContentStyle::NORM:
             m_asis = false ; 
             m_inst = true ;
             m_bbox = false ; 
             m_wire = false ; 
             break;
      case ContentStyle::NONE:
             m_asis = false ; 
             m_inst = false ;
             m_bbox = false ; 
             m_wire = false ; 
             break;
      case ContentStyle::WIRE:
             m_asis = false ; 
             m_inst = true ;
             m_bbox = false ; 
             m_wire = true ; 
             break;
      case ContentStyle::NORM_BBOX:
             m_asis = false ; 
             m_inst = true ; 
             m_bbox = true ; 
             m_wire = false ; 
             break;
      case ContentStyle::NUM_CONTENT_STYLE:
             assert(0);
             break;
   }
}

std::string ContentStyle::desc() const 
{
     std::stringstream ss ; 
     ss << "ContentStyle "
        << getContentStyleName()
        << " inst " << m_inst 
        << " bbox " << m_bbox 
        << " wire " << m_wire 
        << " asis " << m_asis 
        << " m_num_content_style " << m_num_content_style
        << " NUM_CONTENT_STYLE " << NUM_CONTENT_STYLE 
        ;
 
     return ss.str() ;
}

 
