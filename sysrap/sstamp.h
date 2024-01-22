#pragma  once

#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>

struct sstamp
{
    static int64_t Now(); 
    static std::string Format(int64_t t=0, const char* fmt="%FT%T."); 
    static std::string FormatInt(int64_t t, int wid ); 
    static bool LooksLikeStampInt(const char* str); 
    static void sleep(int seconds); 
    static void sleep_us(int microseconds); 
}; 

inline int64_t sstamp::Now()
{
    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds ;
    std::chrono::time_point<Clock> t0 = Clock::now();
    return std::chrono::duration_cast<Unit>(t0.time_since_epoch()).count() ;  
}
/**
stamp::Format
--------------

Time string from uint64_t with the microseconds since UTC epoch, 
t=0 is special cased to give the current time

**/

inline std::string sstamp::Format(int64_t t, const char* fmt)
{
    if(t == 0) t = Now() ; 
    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds  ; 
    std::chrono::time_point<Clock> tp{Unit{t}} ; 

    std::time_t tt = Clock::to_time_t(tp);

    // extract the sub second part from the duration since epoch
    auto subsec = std::chrono::duration_cast<Unit>(tp.time_since_epoch()) % std::chrono::seconds{1};

    std::stringstream ss ; 
    ss 
       << std::put_time(std::localtime(&tt), fmt ) 
       << std::setfill('0') 
       << std::setw(6) << subsec.count() 
       ;

    std::string str = ss.str(); 
    return str ; 
}

inline std::string sstamp::FormatInt(int64_t t, int wid ) // static
{
    std::stringstream ss ; 
    if( t > -1 ) ss << std::setw(wid) << t ;  
    else         ss << std::setw(wid) << "" ; 
    std::string str = ss.str(); 
    return str ; 
}

/**
sstamp::LooksLikeStampInt
--------------------------

Contemporary microsecond uint64_t timestamps since Sept 2001 look like below with 16 digits::

    1700224486350245

::

    In [20]: np.c_[np.array([0,int(1e15),1700224486350245,int(1e16),int(0x7ffffffffffffff) ]).view("datetime64[us]")]
    Out[20]:
    array([[ '1970-01-01T00:00:00.000000'],
           [ '2001-09-09T01:46:40.000000'],
           [ '2023-11-17T12:34:46.350245'],
           [ '2286-11-20T17:46:40.000000'],
           ['20237-04-25T10:45:03.423487']], dtype='datetime64[us]')

*/

inline bool sstamp::LooksLikeStampInt(const char* str) // static
{
    int length = strlen(str) ; 
    int digits = 0 ;  
    for(int i=0 ; i < length ; i++) if(str[i] >= '0' && str[i] <= '9') digits += 1 ;   
    return length == 16 && digits == length  ; 
}

inline void sstamp::sleep(int seconds)
{
    std::chrono::seconds dura(seconds);
    std::this_thread::sleep_for( dura );
}

/**
sstamp::sleep_us
------------------

+--------------+--------------+
| microseconds |   seconds    |
+==============+==============+
| 1,000,000    |  1           |
+--------------+--------------+
|   100,000    |  0.1         |
+--------------+--------------+

**/

inline void sstamp::sleep_us(int us)
{
    std::chrono::microseconds dura(us);
    std::this_thread::sleep_for( dura );
}




