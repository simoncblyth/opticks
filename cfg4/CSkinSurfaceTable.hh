#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
#include "CSurfaceTable.hh"

class CFG4_API CSkinSurfaceTable : public CSurfaceTable {
    public:
         CSkinSurfaceTable();
         void dump(const char* msg="CSkinSurfaceTable::dump");
    private:
         void init();

};

#include "CFG4_TAIL.hh"

