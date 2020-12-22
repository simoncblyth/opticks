#pragma once

#include "G4GDMLReadDefine.hh"  // for G4GDMLMatrix

class X4GDMLMatrix
{
    public:
        X4GDMLMatrix(const G4GDMLMatrix& matrix );
        std::string desc(unsigned edgeitems) const ; 
    private:
        const G4GDMLMatrix& m_matrix ; 

};
