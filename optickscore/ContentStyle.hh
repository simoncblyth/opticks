#pragma once

#include <string>
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
ContentStyle
===============

Canonical m_content_style instance is ctor resident of Composition

**/

class OKCORE_API ContentStyle {
   public:
        ContentStyle();
   public:
        void nextContentStyle();
        std::string desc() const ; 
        bool isInst() const ; 
        bool isBBox() const ; 
        bool isWire() const ; 
        bool isASIS() const ; 
   public: 
        typedef enum { ASIS, BBOX, NORM, NONE, WIRE, NUM_CONTENT_STYLE, NORM_BBOX } ContentStyle_t ;
        void setNumContentStyle(unsigned num_content_style); // used to disable WIRE style for JUNO
   private:
        // ContentStyle
        static const char* ASIS_ ; 
        static const char* BBOX_ ; 
        static const char* NORM_ ; 
        static const char* NONE_ ; 
        static const char* WIRE_ ; 
        static const char* NORM_BBOX_ ; 

        unsigned getNumContentStyle(); // allows ro override the enum
        void setContentStyle(ContentStyle::ContentStyle_t style);
        ContentStyle::ContentStyle_t getContentStyle() const ; 
        void applyContentStyle();
        static const char* getContentStyleName(ContentStyle::ContentStyle_t style);
        const char* getContentStyleName() const ;
        void dumpContentStyles(const char* msg); 
   private:
        ContentStyle_t  m_content_style ; 
        unsigned int    m_num_content_style ; 
        bool            m_inst ; 
        bool            m_bbox ; 
        bool            m_wire ; 
        bool            m_asis ; 

};


#include "OKCORE_TAIL.hh"


 
