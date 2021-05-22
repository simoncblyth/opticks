#pragma once

class Opticks ; 
class OpticksEvent ; 
class CRecorder ;
class CRandomEngine ; 
class CMaterialBridge ; 

struct CG4Ctx ; 


/**
CManager
==========

Middle management operating beneath CG4 and G4OpticksRecorder levels 
such that it can be used by both those.

So for example the manager will be what the geant4 actions talk to, 
rather than CG4 which is too high level for reusabliity. 

**/

#include "CFG4_API_EXPORT.hh"

struct CFG4_API CManager
{
    Opticks*        m_ok ; 
    bool            m_dynamic ; 
    CG4Ctx*         m_ctx ; 
    CRandomEngine*  m_engine ; 
    CRecorder*      m_recorder ; 


    void setMaterialBridge(const CMaterialBridge* material_bridge);

    double             flat_instrumented(const char* file, int line);
    CRandomEngine*     getRandomEngine() const ; 
    CRecorder*         getRecorder() const ;
    CG4Ctx&            getCtx() ;
    unsigned long long getSeqHis() const ;

    CManager(Opticks* ok, bool dynamic ); 

    void initEvent(OpticksEvent* evt);

    void preTrack(); 
    void postTrack(); 
    void postStep(); 

    void postpropagate();

    void addRandomNote(const char* note, int value);
    void addRandomCut(const char* ckey, double cvalue);



};


