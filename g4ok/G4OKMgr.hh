#pragma once

#include <vector>
#include <map>

#include "G4OK_API_EXPORT.hh"

class OpMgr;
class G4Run;
class G4Event; 

class G4OK_API G4OKMgr 
{
  public:
    G4OKMgr();
    ~G4OKMgr();

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

    void addGenstep( float* data, unsigned num_float );

  private:
    OpMgr* m_opmgr;
    std::map<std::string, int> m_mat_g; // geant4 mat name: index
    std::vector<int> m_g2c; // mapping of mat idx: geant4 to opticks
};



