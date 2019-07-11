#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
#include "CSurfaceTable.hh"
#include "plog/Severity.h"

class CFG4_API CSkinSurfaceTable : public CSurfaceTable {
         static const plog::Severity LEVEL ;  
    public:
         CSkinSurfaceTable();
         void dump(const char* msg="CSkinSurfaceTable::dump");
    private:
         void init();
    private:

};

#include "CFG4_TAIL.hh"

