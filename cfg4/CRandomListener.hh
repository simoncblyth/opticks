#pragma once

#include "CFG4_API_EXPORT.hh"


class CFG4_API CRandomListener
{
    public: 
        virtual void preTrack() = 0 ; 
        virtual void postTrack() = 0 ; 
        virtual void postStep() = 0 ; 
        virtual void postpropagate() = 0 ; 
        virtual double flat_instrumented(const char* file, int line) = 0 ;

};
