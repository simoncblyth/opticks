/*
The MIT License(MIT)

Copyright(c) 2015 Alexandre Avenel

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files(the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#ifndef MORTON2D_H
#define MORTON2D_H

#include <cstdint>
#include <array>
#include <algorithm>
#include <assert.h>
#include <immintrin.h>

/*
BMI2(Bit Manipulation Instruction Set 2) is a special set of instructions available for intel core i5, i7(since Haswell architecture) and Xeon E3.
Some instructions are not available for Microsoft Visual Studio older than 2013.
*/
#if _MSC_VER
#define USE_BMI2
#endif

#ifndef USE_BMI2
//mortonkey(x+1) = (mortonkey(x) - MAXMORTONKEY) & MAXMORTONKEY
static const uint32_t morton2dLUT[256] =
{ 0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015,
  0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055,
  0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115,
  0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155,
  0x0400, 0x0401, 0x0404, 0x0405, 0x0410, 0x0411, 0x0414, 0x0415,
  0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455,
  0x0500, 0x0501, 0x0504, 0x0505, 0x0510, 0x0511, 0x0514, 0x0515,
  0x0540, 0x0541, 0x0544, 0x0545, 0x0550, 0x0551, 0x0554, 0x0555,
  0x1000, 0x1001, 0x1004, 0x1005, 0x1010, 0x1011, 0x1014, 0x1015,
  0x1040, 0x1041, 0x1044, 0x1045, 0x1050, 0x1051, 0x1054, 0x1055,
  0x1100, 0x1101, 0x1104, 0x1105, 0x1110, 0x1111, 0x1114, 0x1115,
  0x1140, 0x1141, 0x1144, 0x1145, 0x1150, 0x1151, 0x1154, 0x1155,
  0x1400, 0x1401, 0x1404, 0x1405, 0x1410, 0x1411, 0x1414, 0x1415,
  0x1440, 0x1441, 0x1444, 0x1445, 0x1450, 0x1451, 0x1454, 0x1455,
  0x1500, 0x1501, 0x1504, 0x1505, 0x1510, 0x1511, 0x1514, 0x1515,
  0x1540, 0x1541, 0x1544, 0x1545, 0x1550, 0x1551, 0x1554, 0x1555,
  0x4000, 0x4001, 0x4004, 0x4005, 0x4010, 0x4011, 0x4014, 0x4015,
  0x4040, 0x4041, 0x4044, 0x4045, 0x4050, 0x4051, 0x4054, 0x4055,
  0x4100, 0x4101, 0x4104, 0x4105, 0x4110, 0x4111, 0x4114, 0x4115,
  0x4140, 0x4141, 0x4144, 0x4145, 0x4150, 0x4151, 0x4154, 0x4155,
  0x4400, 0x4401, 0x4404, 0x4405, 0x4410, 0x4411, 0x4414, 0x4415,
  0x4440, 0x4441, 0x4444, 0x4445, 0x4450, 0x4451, 0x4454, 0x4455,
  0x4500, 0x4501, 0x4504, 0x4505, 0x4510, 0x4511, 0x4514, 0x4515,
  0x4540, 0x4541, 0x4544, 0x4545, 0x4550, 0x4551, 0x4554, 0x4555,
  0x5000, 0x5001, 0x5004, 0x5005, 0x5010, 0x5011, 0x5014, 0x5015,
  0x5040, 0x5041, 0x5044, 0x5045, 0x5050, 0x5051, 0x5054, 0x5055,
  0x5100, 0x5101, 0x5104, 0x5105, 0x5110, 0x5111, 0x5114, 0x5115,
  0x5140, 0x5141, 0x5144, 0x5145, 0x5150, 0x5151, 0x5154, 0x5155,
  0x5400, 0x5401, 0x5404, 0x5405, 0x5410, 0x5411, 0x5414, 0x5415,
  0x5440, 0x5441, 0x5444, 0x5445, 0x5450, 0x5451, 0x5454, 0x5455,
  0x5500, 0x5501, 0x5504, 0x5505, 0x5510, 0x5511, 0x5514, 0x5515,
  0x5540, 0x5541, 0x5544, 0x5545, 0x5550, 0x5551, 0x5554, 0x5555
};
#endif

const uint64_t x2_mask = 0xAAAAAAAAAAAAAAAA; //0b...10101010
const uint64_t y2_mask = 0x5555555555555555; //0b...01010101

template<class T = uint64_t>
struct morton2d
{
public:
	T key;

public:

	inline morton2d() : key(0) {};
	inline explicit morton2d(T _key) : key(_key) {};

	/* If BMI2 intrinsics are not available, we rely on a look up table of precomputed morton codes. */
	inline morton2d(const uint32_t x, const uint32_t y) : key(0) {

#ifdef USE_BMI2
		key = static_cast<T>(_pdep_u64(y, y2_mask) | _pdep_u64(x, x2_mask));
#else
		key = morton2dLUT[(x >> 24) & 0xFF] << 1 |
			morton2dLUT[(y >> 24) & 0xFF];
		key = key << 16 |
			morton2dLUT[(x >> 16) & 0xFF] << 1 |
			morton2dLUT[(y >> 16) & 0xFF];
		key = key << 16 |
			morton2dLUT[(x >> 8) & 0xFF] << 1 |
			morton2dLUT[(y >> 8) & 0xFF];
		key = key << 16 |
			morton2dLUT[x & 0xFF] << 1 |
			morton2dLUT[y & 0xFF];
#endif
	}

	inline void decode(uint64_t& x, uint64_t& y) const
	{
#ifdef USE_BMI2
		x = _pext_u64(this->key, x2_mask);
		y = _pext_u64(this->key, y2_mask);
#else
		x = compactBits(this->key >> 1);
		y = compactBits(this->key);
#endif
	}

	//Binary operators
	inline bool operator==(const morton2d m1) const
	{
		return this->key == m1.key;
	}

	inline bool operator!=(const morton2d m1) const
	{
		return !operator==(m1);
	}

	inline morton2d operator|(const morton2d m1) const
	{
		return morton2d<T>(this->key | m1.key);
	}

	inline morton2d operator&(const morton2d m1) const
	{
		return morton2d<T>(this->key & m1.key);
	}

	inline morton2d operator >> (const uint64_t d) const
	{
		return morton2d<T>(this->key >> (2 * d));
	}

	inline morton2d operator<<(const uint64_t d) const
	{
		return morton2d<T>(this->key << (2 * d));
	}

	inline void operator+=(const morton2d<T> rhs)
	{
		T x_sum = (this->key | y2_mask) + (rhs.key & x2_mask);
		T y_sum = (this->key | x2_mask) + (rhs.key & y2_mask);
		this->key = (x_sum & x2_mask) | (y_sum & y2_mask);
	}

	inline void operator-=(const morton2d<T> rhs)
	{
		T x_diff = (this->key & x2_mask) - (rhs.key & x2_mask);
		T y_diff = (this->key & y2_mask) - (rhs.key & y2_mask);
		this->key = (x_diff & x2_mask) | (y_diff & y2_mask);
	}

	/* Increment X part of a morton2 code (xy interleaving)
	morton2(4,5).incX() == morton2(5,5);

	Ref : http://bitmath.blogspot.fr/2012/11/tesseral-arithmetic-useful-snippets.html */
	inline morton2d incX() const
	{
		const T x_sum = static_cast<T>((this->key | y2_mask) + 2);
		return morton2d<T>((x_sum & x2_mask) | (this->key & y2_mask));
	}

	inline morton2d incY() const
	{
		const T y_sum = static_cast<T>((this->key | x2_mask) + 1);
		return morton2d<T>((y_sum & y2_mask) | (this->key & x2_mask));
	}

	inline morton2d decX() const
	{
		const T x_diff = static_cast<T>((this->key & x2_mask) - 2);
		return morton2d<T>((x_diff & x2_mask) | (this->key & y2_mask));
	}

	inline morton2d decY() const
	{
		const T y_diff = static_cast<T>((this->key & y2_mask) - 1);
		return morton2d<T>((y_diff & y2_mask) | (this->key & x2_mask));
	}

	/*
	  min(morton2(4,5), morton2(8,3)) == morton2(4,3);
	  Ref : http://asgerhoedt.dk/?p=276
	*/
	static inline morton2d min(const morton2d lhs, const morton2d rhs)
	{
		T lhsX = lhs.key & x2_mask;
		T rhsX = rhs.key & x2_mask;
		T lhsY = lhs.key & y2_mask;
		T rhsY = rhs.key & y2_mask;
		return morton2d<T>(std::min(lhsX, rhsX) + std::min(lhsY, rhsY));
	}

	/*
	  max(morton2(4,5), morton2(8,3)) == morton2(8,5);
	*/
	static inline morton2d max(const morton2d lhs, const morton2d rhs)
	{
		T lhsX = lhs.key & x2_mask;
		T rhsX = rhs.key & x2_mask;
		T lhsY = lhs.key & y2_mask;
		T rhsY = rhs.key & y2_mask;
		return morton2d<T>(std::max(lhsX, rhsX) + std::max(lhsY, rhsY));
	}

#ifndef USE_BMI2

	/* Fast encode of morton2 code when BMI2 instructions aren't available.
	This does not work for values greater than 256.

	This function takes roughly the same time as a full encode (64 bits) using BMI2 intrinsic.*/
	static inline morton2d morton2d_256(const uint32_t x, const uint32_t y)
	{
		assert(x < 256 && y < 256);
		T key = morton2dLUT[x] << 1 |
			morton2dLUT[y];
		return morton2d(key);
	}

private:
	inline uint64_t compactBits(uint64_t n) const
	{
		n &= 0x5555555555555555;
		n = (n ^ (n >> 1)) & 0x3333333333333333;
		n = (n ^ (n >> 2)) & 0x0f0f0f0f0f0f0f0f;
		n = (n ^ (n >> 4)) & 0x00ff00ff00ff00ff;
		n = (n ^ (n >> 8)) & 0x0000ffff0000ffff;
		n = (n ^ (n >> 16)) & 0x00000000ffffffff;
		return n;
	}
#endif

};

/* Add two morton keys (xy interleaving)
morton2(4,5) + morton3(1,2) == morton2(5,7); */
template<class T>
inline morton2d<T> operator+(const morton2d<T> lhs, const morton2d<T> rhs)
{
	T x_sum = (lhs.key | y2_mask) + (rhs.key & x2_mask);
	T y_sum = (lhs.key | x2_mask) + (rhs.key & y2_mask);
	return morton2d<T>((x_sum & x2_mask) | (y_sum & y2_mask));
}

/* Substract two mortons keys (xy interleaving)
  morton2(4,5) - morton2(1,2) == morton2(3,3); */
template<class T>
inline morton2d<T> operator-(const morton2d<T> lhs, const morton2d<T> rhs)
{
	T x_diff = (lhs.key & x2_mask) - (rhs.key & x2_mask);
	T y_diff = (lhs.key & y2_mask) - (rhs.key & y2_mask);
	return morton2d<T>((x_diff & x2_mask) | (y_diff & y2_mask));
}

template<class T>
inline bool operator< (const morton2d<T>& lhs, const morton2d<T>& rhs)
{
	return (lhs.key) < (rhs.key);
}

template<class T>
inline bool operator> (const morton2d<T>& lhs, const morton2d<T>& rhs)
{
	return (lhs.key) > (rhs.key);
}

template<class T>
inline bool operator>= (const morton2d<T>& lhs, const morton2d<T>& rhs)
{
	return (lhs.key) >= (rhs.key);
}

template<class T>
inline bool operator<= (const morton2d<T>& lhs, const morton2d<T>& rhs)
{
	return (lhs.key) <= (rhs.key);
}

template<class T>
std::ostream& operator<<(std::ostream& os, const morton2d<T>& m)
{
	uint64_t x, y;
	m.decode(x, y);
	os << m.key << ": " << x << ", " << y;
	return os;
}

typedef morton2d<> morton2;

#endif
