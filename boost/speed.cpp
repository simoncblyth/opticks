/* 
 *
 * http://tinodidriksen.com/2011/05/28/cpp-convert-string-to-double-speed/
 * http://tinodidriksen.com/uploads/code/cpp/speed-string-to-double.cpp
 * http://www.fftw.org/cycle.h
 * http://www.leapsecond.com/tools/fast_atof.c
 *
 */

#ifdef _MSC_VER
    #define _SECURE_SCL 0
    #define _CRT_SECURE_NO_DEPRECATE 1
    #define WIN32_LEAN_AND_MEAN
    #define VC_EXTRALEAN
    #define NOMINMAX
#endif

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include "cycle.h"

static const size_t N = 100000;
static const size_t R = 7;

void PrintStats(std::vector<double> timings) {
    double fastest = std::numeric_limits<double>::max();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[";
    for (size_t i = 1 ; i<timings.size()-1 ; ++i) {
        fastest = std::min(fastest, timings[i]);
        std::cout << timings[i] << ",";
    }
    std::cout << timings.back();
    std::cout << "]";

    double sum = 0.0;
    for (size_t i = 1 ; i<timings.size() ; ++i) {
        sum += timings[i];
    }
    double avg = sum / static_cast<double>(timings.size()-1);

    sum = 0.0;
    for (size_t i = 1 ; i<timings.size() ; ++i) {
        timings[i] = pow(timings[i]-avg, 2);
        sum += timings[i];
    }
    double var = sum/(timings.size()-2);
    double sdv = sqrt(var);

    std::cout << " with fastest " << fastest << ", average " << avg << ", stddev " << sdv;
}

double naive(const char *p) {
    double r = 0.0;
    bool neg = false;
    if (*p == '-') {
        neg = true;
        ++p;
    }
    while (*p >= '0' && *p <= '9') {
        r = (r*10.0) + (*p - '0');
        ++p;
    }
    if (*p == '.') {
        double f = 0.0;
        int n = 0;
        ++p;
        while (*p >= '0' && *p <= '9') {
            f = (f*10.0) + (*p - '0');
            ++p;
            ++n;
        }
        r += f / std::pow(10.0, n);
    }
    if (neg) {
        r = -r;
    }
    return r;
}

int main() {
    std::vector<std::string> nums;
    nums.reserve(N);
    for (size_t i=0 ; i<N ; ++i) {
        std::string y;
        if (i & 1) {
            y += '-';
        }
        y += boost::lexical_cast<std::string>(i);
        y += '.';
        y += boost::lexical_cast<std::string>(i);
        nums.push_back(y);
    }

    {
        double tsum = 0.0;
        std::vector<double> timings;
        timings.reserve(R);
        for (size_t r=0 ; r<R ; ++r) {
            ticks start = getticks();
            for (size_t i=0 ; i<nums.size() ; ++i) {
                double x = naive(nums[i].c_str());
                tsum += x;
            }
            ticks end = getticks();
            double timed = elapsed(end, start);
            timings.push_back(timed);
        }

        std::cout << "naive: ";
        PrintStats(timings);
        std::cout << std::endl;
        std::cout << tsum << std::endl;
    }

    {
        double tsum = 0.0;
        std::vector<double> timings;
        timings.reserve(R);
        for (size_t r=0 ; r<R ; ++r) {
            ticks start = getticks();
            for (size_t i=0 ; i<nums.size() ; ++i) {
                double x = atof(nums[i].c_str());
                tsum += x;
            }
            ticks end = getticks();
            double timed = elapsed(end, start);
            timings.push_back(timed);
        }

        std::cout << "atof(): ";
        PrintStats(timings);
        std::cout << std::endl;
        std::cout << tsum << std::endl;
    }

    {
        double tsum = 0.0;
        std::vector<double> timings;
        timings.reserve(R);
        for (size_t r=0 ; r<R ; ++r) {
            ticks start = getticks();
            for (size_t i=0 ; i<nums.size() ; ++i) {
                double x = strtod(nums[i].c_str(), 0);
                tsum += x;
            }
            ticks end = getticks();
            double timed = elapsed(end, start);
            timings.push_back(timed);
        }

        std::cout << "strtod(): ";
        PrintStats(timings);
        std::cout << std::endl;
        std::cout << tsum << std::endl;
    }

    {
        double tsum = 0.0;
        std::vector<double> timings;
        timings.reserve(R);
        for (size_t r=0 ; r<R ; ++r) {
            ticks start = getticks();
            for (size_t i=0 ; i<nums.size() ; ++i) {
                double x = 0.0;
                sscanf(nums[i].c_str(), "%lf", &x);
                tsum += x;
            }
            ticks end = getticks();
            double timed = elapsed(end, start);
            timings.push_back(timed);
        }

        std::cout << "sscanf(): ";
        PrintStats(timings);
        std::cout << std::endl;
        std::cout << tsum << std::endl;
    }

    {
        double tsum = 0.0;
        std::vector<double> timings;
        timings.reserve(R);
        for (size_t r=0 ; r<R ; ++r) {
            ticks start = getticks();
            for (size_t i=0 ; i<nums.size() ; ++i) {
                double x = boost::lexical_cast<double>(nums[i]);
                tsum += x;
            }
            ticks end = getticks();
            double timed = elapsed(end, start);
            timings.push_back(timed);
        }

        std::cout << "lexical_cast: ";
        PrintStats(timings);
        std::cout << std::endl;
        std::cout << tsum << std::endl;
    }

    {
        using boost::spirit::qi::double_;
        using boost::spirit::qi::parse;
        double tsum = 0.0;
        std::vector<double> timings;
        timings.reserve(R);
        for (size_t r=0 ; r<R ; ++r) {
            ticks start = getticks();
            for (size_t i=0 ; i<nums.size() ; ++i) {
                double x = 0.0;
                char const *str = nums[i].c_str();
                parse(str, &str[nums[i].size()], double_, x);
                tsum += x;
            }
            ticks end = getticks();
            double timed = elapsed(end, start);
            timings.push_back(timed);
        }

        std::cout << "spirit qi: ";
        PrintStats(timings);
        std::cout << std::endl;
        std::cout << tsum << std::endl;
    }

    {
        double tsum = 0.0;
        std::vector<double> timings;
        timings.reserve(R);
        for (size_t r=0 ; r<R ; ++r) {
            ticks start = getticks();
            for (size_t i=0 ; i<nums.size() ; ++i) {
                std::istringstream ss(nums[i]);
                double x = 0.0;
                ss >> x;
                tsum += x;
            }
            ticks end = getticks();
            double timed = elapsed(end, start);
            timings.push_back(timed);
        }

        std::cout << "stringstream: ";
        PrintStats(timings);
        std::cout << std::endl;
        std::cout << tsum << std::endl;
    }

    {
        double tsum = 0.0;
        std::vector<double> timings;
        timings.reserve(R);
        for (size_t r=0 ; r<R ; ++r) {
            ticks start = getticks();
            std::istringstream ss;
            for (size_t i=0 ; i<nums.size() ; ++i) {
                ss.str(nums[i]);
                ss.clear();
                double x = 0.0;
                ss >> x;
                tsum += x;
            }
            ticks end = getticks();
            double timed = elapsed(end, start);
            timings.push_back(timed);
        }

        std::cout << "stringstream reused: ";
        PrintStats(timings);
        std::cout << std::endl;
        std::cout << tsum << std::endl;
    }
}
