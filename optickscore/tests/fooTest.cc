
// NB for this to work with clang 
//    need to switch off visibility and selectively set it to default
//    for the API : making clang behave like msvc
//   
//    https://github.com/SergiusTheBest/plog/issues/20  
//   

#include <plog/Log.h>

// Functions imported form the shared library.
extern "C" void foo_initialize(plog::Severity severity, plog::IAppender* appender);
extern "C" void foo_check();

int main()
{
    plog::init(plog::debug, "/tmp/fooTest.txt"); // Initialize the main logger.

    LOGD << "Hello from app!"; // Write a log message.

    foo_initialize(plog::debug, plog::get()); // Initialize the logger in the shared library. Note that it has its own severity.
    foo_check(); // Call a function from the shared library that produces a log message.

    return 0;
}
