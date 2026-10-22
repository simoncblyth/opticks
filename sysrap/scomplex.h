#pragma once
/**
scomplex.h
===========

* CAUTION Untested generated code 
* aiming to replace thrust::complex (and its heavy headers) in device code (especially optix kernels)

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define SCOMPLEX_METHOD __host__ __device__
#else
   #define SCOMPLEX_METHOD 
   #include <cmath>
#endif 


namespace opticks {

template <typename T>
struct scomplex {
    T _real;
    T _imag;

    SCOMPLEX_METHOD constexpr scomplex(T r = T(0), T i = T(0)) : _real(r), _imag(i) {}

    SCOMPLEX_METHOD constexpr T real() const { return _real; }
    SCOMPLEX_METHOD constexpr T imag() const { return _imag; }

    SCOMPLEX_METHOD void real(const T& r) { _real = r ; }
    SCOMPLEX_METHOD void imag(const T& i) { _imag = i ; }


    SCOMPLEX_METHOD scomplex& operator+=(const scomplex& o) {
        _real += o._real; 
        _imag += o._imag; 
        return *this;
    }
    SCOMPLEX_METHOD scomplex& operator-=(const scomplex& o) {
        _real -= o._real; 
        _imag -= o._imag; 
        return *this;
    }
    SCOMPLEX_METHOD scomplex& operator*=(const scomplex& o) {
        T r = _real * o._real - _imag * o._imag;
        T i = _real * o._imag + _imag * o._real;
        _real = r; 
        _imag = i;
        return *this;
    }
    SCOMPLEX_METHOD scomplex& operator/=(const scomplex& o) {
        T denom = o._real * o._real + o._imag * o._imag;
        T r = (_real * o._real + _imag * o._imag) / denom;
        T i = (_imag * o._real - _real * o._imag) / denom;
        _real = r; 
        _imag = i;
        return *this;
    }














};

// Binary operators
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator+(scomplex<T> lhs, const scomplex<T>& rhs) { return lhs += rhs; }
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator-(scomplex<T> lhs, const scomplex<T>& rhs) { return lhs -= rhs; }
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator*(scomplex<T> lhs, const scomplex<T>& rhs) { return lhs *= rhs; }
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator/(scomplex<T> lhs, const scomplex<T>& rhs) { return lhs /= rhs; }









// Unary Operators
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator-(const scomplex<T>& c) { return scomplex<T>(-c.real(), -c.imag()); }

// Mixed Scalar (T) and Complex Operators
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator+(T lhs, const scomplex<T>& rhs) { return scomplex<T>(lhs + rhs.real(), rhs.imag()); }
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator+(const scomplex<T>& lhs, T rhs) { return scomplex<T>(lhs.real() + rhs, lhs.imag()); }
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator-(T lhs, const scomplex<T>& rhs) { return scomplex<T>(lhs - rhs.real(), -rhs.imag());}
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator-(const scomplex<T>& lhs, T rhs) { return scomplex<T>(lhs.real() - rhs, lhs.imag()); }
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator*(T lhs, const scomplex<T>& rhs) { return scomplex<T>(lhs * rhs.real(), lhs * rhs.imag());}
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator*(const scomplex<T>& lhs, T rhs) { return scomplex<T>(lhs.real() * rhs, lhs.imag() * rhs);}
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator/(const scomplex<T>& lhs, T rhs) { return scomplex<T>(lhs.real() / rhs, lhs.imag() / rhs); }
template <typename T>
SCOMPLEX_METHOD inline scomplex<T> operator/(T lhs, const scomplex<T>& rhs) {
    T denom = rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
    return scomplex<T>((lhs * rhs.real()) / denom, (-lhs * rhs.imag()) / denom);
}




// TMM Support Functions
template <typename T>
SCOMPLEX_METHOD inline T norm(const scomplex<T>& c) {
    return c.real() * c.real() + c.imag() * c.imag();
}

template <typename T>
SCOMPLEX_METHOD inline scomplex<T> conj(const scomplex<T>& c) {
    return scomplex<T>(c.real(), -c.imag());
}

template <typename T>
SCOMPLEX_METHOD inline scomplex<T> exp(const scomplex<T>& c) {
    // e^(a + bi) = e^a * (cos(b) + i sin(b))
    T ea = ::exp(c.real());
    return scomplex<T>(ea * ::cos(c.imag()), ea * ::sin(c.imag()));
}

template <typename T>
SCOMPLEX_METHOD inline scomplex<T> sqrt(const scomplex<T>& c) {
    T mag = ::sqrt(norm(c));
    T real_part = ::sqrt(T(0.5) * (mag + c.real()));
    T imag_part = ::sqrt(T(0.5) * (mag - c.real()));
    if (c.imag() < T(0)) imag_part = -imag_part;
    return scomplex<T>(real_part, imag_part);
}

template <typename T>
SCOMPLEX_METHOD inline scomplex<T> sin(const scomplex<T>& c) {
    // sin(x + iy) = sin(x)cosh(y) + i cos(x)sinh(y)
    T x = c.real();
    T y = c.imag();
    return scomplex<T>(::sin(x) * ::cosh(y), ::cos(x) * ::sinh(y));
}

template <typename T>
SCOMPLEX_METHOD inline scomplex<T> cos(const scomplex<T>& c) {
    // cos(x + iy) = cos(x)cosh(y) - i sin(x)sinh(y)
    T x = c.real();
    T y = c.imag();
    return scomplex<T>(::cos(x) * ::cosh(y), -::sin(x) * ::sinh(y));
}

} // namespace opticks
