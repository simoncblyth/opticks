#include "Randomize.hh"
#include "Opticks.hh"

#include "CRandomEngine.hh"
#include "CRecorder.hh"
#include "CG4Ctx.hh"
#include "CManager.hh"



CRecorder* CManager::getRecorder() const 
{ 
    return m_recorder ; 
}
CRandomEngine* CManager::getRandomEngine() const 
{ 
    return m_engine ; 
}
CG4Ctx& CManager::getCtx()
{
    return *m_ctx ; 
}

double CManager::flat_instrumented(const char* file, int line)
{
    return m_engine ? m_engine->flat_instrumented(file, line) : G4UniformRand() ; 
}

unsigned long long CManager::getSeqHis() const { return m_recorder->getSeqHis() ; }


CManager::CManager(Opticks* ok, bool dynamic )
    :
    m_ok(ok),
    m_dynamic(dynamic),
    m_ctx(new CG4Ctx(m_ok)),
    m_engine(m_ok->isAlign() ? new CRandomEngine(this) : NULL  ),   // --align
    m_recorder(new CRecorder(*m_ctx, m_dynamic))
{
}



void CManager::setMaterialBridge(const CMaterialBridge* material_bridge)
{
    m_recorder->setMaterialBridge(material_bridge); 
}


void CManager::initEvent(OpticksEvent* evt)
{
    m_ctx->initEvent(evt);
    m_recorder->initEvent(evt);
}






void CManager::preTrack()
{
    if(m_engine)
    {
        m_engine->preTrack();
    }
}

void CManager::postTrack()
{

    if(m_ctx->_optical)
    {
        m_recorder->postTrack();
    } 
    if(m_engine)
    {
        m_engine->postTrack();
    }
}

void CManager::postStep()
{
    if(m_engine)
    {
        m_engine->postStep();
    }
}


void CManager::postpropagate()
{
    if(m_engine) m_engine->postpropagate();  
}


/**
CManager::addRandomNote
--------------------------

The note is associated with the index of the last random consumption, see boostrap/BLog.cc

**/

void CManager::addRandomNote(const char* note, int value)
{
    assert( m_engine ); 
    m_engine->addNote(note, value); 
}

void CManager::addRandomCut(const char* ckey, double cvalue)
{
    assert( m_engine ); 
    m_engine->addCut(ckey, cvalue); 
}








