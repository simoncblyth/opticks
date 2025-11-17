#pragma once
/**
sresource.h
==============

* https://man7.org/linux/man-pages/man2/getrlimit.2.html


RLIMIT_MEMLOCK
      This is the maximum number of bytes of memory that may be
      locked into RAM.  This limit is in effect rounded down to
      the nearest multiple of the system page size.  This limit
      affects mlock(2), mlockall(2), and the mmap(2) MAP_LOCKED
      operation.  Since Linux 2.6.9, it also affects the
      shmctl(2) SHM_LOCK operation, where it sets a maximum on
      the total bytes in shared memory segments (see shmget(2))
      that may be locked by the real user ID of the calling
      process.  The shmctl(2) SHM_LOCK locks are accounted for
      separately from the per-process memory locks established by
      mlock(2), mlockall(2), and mmap(2) MAP_LOCKED; a process
      can lock bytes up to this limit in each of these two
      categories.

      Before Linux 2.6.9, this limit controlled the amount of
      memory that could be locked by a privileged process.  Since
      Linux 2.6.9, no limits are placed on the amount of memory
      that a privileged process may lock, and this limit instead
      governs the amount of memory that an unprivileged process
      may lock.




RLIMIT_MEMLOCK Pinned memory required for Async CUDA copies
------------------------------------------------------------

Default soft and hard limits are only 8MB::

    A[blyth@localhost sysrap]$ ulimit -l
    8192

    A[blyth@localhost tests]$ ulimit -H -l
    8192


permanently increase memlocl limit for a user (may need logout/login)
-----------------------------------------------------------------------

To increase the limit for a user::

   sudo vi /etc/security/limits.conf

And add::

    blyth    soft    memlock    unlimited
    blyth    hard    memlock    unlimited



temporarily increase memlock limits for current process
--------------------------------------------------------

::

    A[blyth@localhost sysrap]$ prlimit --memlock=unlimited:unlimited --pid $$
    prlimit: failed to set the MEMLOCK resource limit: Operation not permitted
    A[blyth@localhost sysrap]$ sudo prlimit --memlock=unlimited:unlimited --pid $$
    [sudo] password for blyth:
    A[blyth@localhost sysrap]$ ulimit -l
    unlimited
    A[blyth@localhost sysrap]$ ulimit -H -l
    unlimited
    A[blyth@localhost sysrap]$



**/


#include <cstdint>
#include <string>
#include <sstream>
#include <sys/resource.h>

namespace sresource::memlock {

struct Limit {
    uint64_t bytes;      // 0 = unlimited
    std::string human;
    std::string desc() const
    {
        std::stringstream ss ;
        ss << "[sresource::memlock::Limit .bytes:" << bytes << " .human:" << human << "]" ;
        std::string str = ss.str() ;
        return str ;
    }
};


inline Limit get() {
    struct rlimit rl{};
    if (::getrlimit(RLIMIT_MEMLOCK, &rl) == -1)
        return {0, "error"};

    auto to_bytes = [](rlim_t v) -> uint64_t { return v == RLIM_INFINITY ? 0 : v; };
    uint64_t soft = to_bytes(rl.rlim_cur);
    uint64_t hard = to_bytes(rl.rlim_max);

    auto format = [](uint64_t b, bool full) -> std::string {
        if (!b) return full ? "unlimited" : "inf";
        const char* units[] = {"B", "KiB", "MiB", "GiB"};
        double s = b; int i = 0;
        while (s >= 1024 && i < 3) { s /= 1024; ++i; }
        char buf[32];
        snprintf(buf, sizeof(buf), full ? "%.2f %s" : "%.0f%s", s, units[i]);
        return buf;
    };

    std::string h = format(soft, true);
    if (soft != hard) h += " (max: " + format(hard, false) + ")";
    return {soft, h};
}

} // namespace sresource::memlock


