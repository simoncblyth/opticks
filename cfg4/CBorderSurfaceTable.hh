#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

#include "CSurfaceTable.hh"

class CFG4_API CBorderSurfaceTable : public CSurfaceTable {
    public:
         CBorderSurfaceTable();
         void dump(const char* msg="CBorderSurfaceTable::dump");
    private:
         void init();

};

#include "CFG4_TAIL.hh"

