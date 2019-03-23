#pragma once

/**
plog::PlainFormatter
=====================

Controls output formatting of plog logger


**/



#include <plog/Log.h>

namespace plog
{
    class PlainFormatter
    {
    public:
        static util::nstring header() // This method returns a header for a new file. In our case it is empty.
        {
            return util::nstring();
        }

        static util::nstring format(const Record& record) // This method returns a string from a record.
        {

#ifdef OLD_PLOG
            util::nstringstream ss;
            ss << record.getMessage().c_str() << "\n"; // Produce a simple string with a log message.
#else
            util::nostringstream ss;
            ss << record.getMessage() << "\n"; // Produce a simple string with a log message.
#endif

            return ss.str();
        }
    };
}

