/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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

#ifndef MORTON3D_H
#define MORTON3D_H

#include <cstdint>
#include <array>
#include <assert.h>

#if _MSC_VER
#include <immintrin.h>
#endif

#if __GNUC__
#include <x86intrin.h>
#endif


/*
BMI2 (Bit Manipulation Instruction Set 2) is a special set of instructions available for intel core i5, i7 (since Haswell architecture) and Xeon E3.
Some instructions are not available for Microsoft Visual Studio older than 2013.
*/
#if _MSC_VER
#define USE_BMI2
#endif

#ifndef USE_BMI2
//mortonkey(x+1) = (mortonkey(x) - MAXMORTONKEY) & MAXMORTONKEY
static const uint32_t morton3dLUT[256] =
{
  0x00000000, 0x00000001, 0x00000008, 0x00000009, 0x00000040, 0x00000041, 0x00000048, 0x00000049,
  0x00000200, 0x00000201, 0x00000208, 0x00000209, 0x00000240, 0x00000241, 0x00000248, 0x00000249,
  0x00001000, 0x00001001, 0x00001008, 0x00001009, 0x00001040, 0x00001041, 0x00001048, 0x00001049,
  0x00001200, 0x00001201, 0x00001208, 0x00001209, 0x00001240, 0x00001241, 0x00001248, 0x00001249,
  0x00008000, 0x00008001, 0x00008008, 0x00008009, 0x00008040, 0x00008041, 0x00008048, 0x00008049,
  0x00008200, 0x00008201, 0x00008208, 0x00008209, 0x00008240, 0x00008241, 0x00008248, 0x00008249,
  0x00009000, 0x00009001, 0x00009008, 0x00009009, 0x00009040, 0x00009041, 0x00009048, 0x00009049,
  0x00009200, 0x00009201, 0x00009208, 0x00009209, 0x00009240, 0x00009241, 0x00009248, 0x00009249,
  0x00040000, 0x00040001, 0x00040008, 0x00040009, 0x00040040, 0x00040041, 0x00040048, 0x00040049,
  0x00040200, 0x00040201, 0x00040208, 0x00040209, 0x00040240, 0x00040241, 0x00040248, 0x00040249,
  0x00041000, 0x00041001, 0x00041008, 0x00041009, 0x00041040, 0x00041041, 0x00041048, 0x00041049,
  0x00041200, 0x00041201, 0x00041208, 0x00041209, 0x00041240, 0x00041241, 0x00041248, 0x00041249,
  0x00048000, 0x00048001, 0x00048008, 0x00048009, 0x00048040, 0x00048041, 0x00048048, 0x00048049,
  0x00048200, 0x00048201, 0x00048208, 0x00048209, 0x00048240, 0x00048241, 0x00048248, 0x00048249,
  0x00049000, 0x00049001, 0x00049008, 0x00049009, 0x00049040, 0x00049041, 0x00049048, 0x00049049,
  0x00049200, 0x00049201, 0x00049208, 0x00049209, 0x00049240, 0x00049241, 0x00049248, 0x00049249,
  0x00200000, 0x00200001, 0x00200008, 0x00200009, 0x00200040, 0x00200041, 0x00200048, 0x00200049,
  0x00200200, 0x00200201, 0x00200208, 0x00200209, 0x00200240, 0x00200241, 0x00200248, 0x00200249,
  0x00201000, 0x00201001, 0x00201008, 0x00201009, 0x00201040, 0x00201041, 0x00201048, 0x00201049,
  0x00201200, 0x00201201, 0x00201208, 0x00201209, 0x00201240, 0x00201241, 0x00201248, 0x00201249,
  0x00208000, 0x00208001, 0x00208008, 0x00208009, 0x00208040, 0x00208041, 0x00208048, 0x00208049,
  0x00208200, 0x00208201, 0x00208208, 0x00208209, 0x00208240, 0x00208241, 0x00208248, 0x00208249,
  0x00209000, 0x00209001, 0x00209008, 0x00209009, 0x00209040, 0x00209041, 0x00209048, 0x00209049,
  0x00209200, 0x00209201, 0x00209208, 0x00209209, 0x00209240, 0x00209241, 0x00209248, 0x00209249,
  0x00240000, 0x00240001, 0x00240008, 0x00240009, 0x00240040, 0x00240041, 0x00240048, 0x00240049,
  0x00240200, 0x00240201, 0x00240208, 0x00240209, 0x00240240, 0x00240241, 0x00240248, 0x00240249,
  0x00241000, 0x00241001, 0x00241008, 0x00241009, 0x00241040, 0x00241041, 0x00241048, 0x00241049,
  0x00241200, 0x00241201, 0x00241208, 0x00241209, 0x00241240, 0x00241241, 0x00241248, 0x00241249,
  0x00248000, 0x00248001, 0x00248008, 0x00248009, 0x00248040, 0x00248041, 0x00248048, 0x00248049,
  0x00248200, 0x00248201, 0x00248208, 0x00248209, 0x00248240, 0x00248241, 0x00248248, 0x00248249,
  0x00249000, 0x00249001, 0x00249008, 0x00249009, 0x00249040, 0x00249041, 0x00249048, 0x00249049,
  0x00249200, 0x00249201, 0x00249208, 0x00249209, 0x00249240, 0x00249241, 0x00249248, 0x00249249
};
#endif

const uint64_t x3_mask = 0x4924924924924924; // 0b...00100100
const uint64_t y3_mask = 0x2492492492492492; // 0b...10010010
const uint64_t z3_mask = 0x9249249249249249; // 0b...01001001
const uint64_t xy3_mask = x3_mask | y3_mask;
const uint64_t xz3_mask = x3_mask | z3_mask;
const uint64_t yz3_mask = y3_mask | z3_mask;

template<class T = uint64_t>
struct morton3d
{
public:
	T key;

public:

	inline explicit morton3d() : key(0) {};
	inline explicit morton3d(const T _key) : key(_key) {};

	/* If BMI2 intrinsics are not available, we rely on a look up table of precomputed morton codes.
	Ref : http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/ */
	inline morton3d(const uint32_t x, const uint32_t y, const uint32_t z) : key(0) {
#ifdef USE_BMI2
		key = static_cast<T>(_pdep_u64(z, z3_mask) | _pdep_u64(y, y3_mask) | _pdep_u64(x, x3_mask));
#else
		key = morton3dLUT[(x >> 16) & 0xFF] << 2 |
			morton3dLUT[(y >> 16) & 0xFF] << 1 |
			morton3dLUT[(z >> 16) & 0xFF];
		key = key << 24 |
			morton3dLUT[(x >> 8) & 0xFF] << 2 |
			morton3dLUT[(y >> 8) & 0xFF] << 1 |
			morton3dLUT[(z >> 8) & 0xFF];
		key = key << 24 |
			morton3dLUT[x & 0xFF] << 2 |
			morton3dLUT[y & 0xFF] << 1 |
			morton3dLUT[z & 0xFF];
#endif
	}

	inline void decode(uint64_t& x, uint64_t& y, uint64_t& z) const
	{
#ifdef USE_BMI2
		x = _pext_u64(this->key, x3_mask);
		y = _pext_u64(this->key, y3_mask);
		z = _pext_u64(this->key, z3_mask);
#else
		x = compactBits(this->key >> 2);
		y = compactBits(this->key >> 1);
		z = compactBits(this->key);
#endif
	}


	//Binary operators
	inline bool operator==(const morton3d m1) const
	{
		return this->key == m1.key;
	}

	inline bool operator!=(const morton3d m1) const
	{
		return !operator==(m1);
	}

	inline morton3d operator|(const morton3d m1) const
	{
		return morton3d<T>(this->key | m1.key);
	}

	inline morton3d operator&(const morton3d m1) const
	{
		return morton3d<T>(this->key & m1.key);
	}

	inline morton3d operator >> (const uint64_t d) const
	{
		assert(d < 22);
		return morton3d<T>(this->key >> (3 * d));
	}

	inline morton3d operator<<(const uint64_t d) const
	{
		assert(d < 22);
		return morton3d<T>(this->key << (3 * d));
	}

	inline void operator+=(const morton3d<T> m1)
	{
		T x_sum = (this->key | yz3_mask) + (m1.key & x3_mask);
		T y_sum = (this->key | xz3_mask) + (m1.key & y3_mask);
		T z_sum = (this->key | xy3_mask) + (m1.key & z3_mask);
		this->key = ((x_sum & x3_mask) | (y_sum & y3_mask) | (z_sum & z3_mask));
	}

	inline void operator-=(const morton3d<T> m1)
	{
		T x_diff = (this->key & x3_mask) - (m1.key & x3_mask);
		T y_diff = (this->key & y3_mask) - (m1.key & y3_mask);
		T z_diff = (this->key & z3_mask) - (m1.key & z3_mask);
		this->key = ((x_diff & x3_mask) | (y_diff & y3_mask) | (z_diff & z3_mask));
	}

#ifndef USE_BMI2
	/* Fast encode of morton3 code when BMI2 instructions aren't available.
	This does not work for values greater than 256.

	This function takes roughly the same time as a full encode (64 bits) using BMI2 intrinsic.*/
	static inline morton3d morton3d_256(const uint32_t x, const uint32_t y, const uint32_t z)
	{
		assert(x < 256 && y < 256 && z < 256);
		uint64_t key = morton3dLUT[x] << 2 |
			morton3dLUT[y] << 1 |
			morton3dLUT[z];
		return morton3d(key);
	}
#endif

	/* Increment X part of a morton3 code (xyz interleaving)
	   morton3(4,5,6).incX() == morton3(5,5,6);

	   Ref : http://bitmath.blogspot.fr/2012/11/tesseral-arithmetic-useful-snippets.html */
	inline morton3d incX() const
	{
		const T x_sum = static_cast<T>((this->key | yz3_mask) + 4);
		return morton3d<T>((x_sum & x3_mask) | (this->key & yz3_mask));
	}

	inline morton3d incY() const
	{
		const T y_sum = static_cast<T>((this->key | xz3_mask) + 2);
		return morton3d<T>((y_sum & y3_mask) | (this->key & xz3_mask));
	}

	inline morton3d incZ() const
	{
		const T z_sum = static_cast<T>((this->key | xy3_mask) + 1);
		return morton3d<T>((z_sum & z3_mask) | (this->key & xy3_mask));
	}

	/* Decrement X part of a morton3 code (xyz interleaving)
	   morton3(4,5,6).decX() == morton3(3,5,6); */
	inline morton3d decX() const
	{
		const T x_diff = (this->key & x3_mask) - 4;
		return morton3d<T>((x_diff & x3_mask) | (this->key & yz3_mask));
	}

	inline morton3d decY() const
	{
		const T y_diff = (this->key & y3_mask) - 2;
		return morton3d<T>((y_diff & y3_mask) | (this->key & xz3_mask));
	}

	inline morton3d decZ() const
	{
		const T z_diff = (this->key & z3_mask) - 1;
		return morton3d<T>((z_diff & z3_mask) | (this->key & xy3_mask));
	}


	/*
	  min(morton3(4,5,6), morton3(8,3,7)) == morton3(4,3,6);
	  Ref : http://asgerhoedt.dk/?p=276
	*/
	static inline morton3d min(const morton3d lhs, const morton3d rhs)
	{
		T lhsX = lhs.key & x3_mask;
		T rhsX = rhs.key & x3_mask;
		T lhsY = lhs.key & y3_mask;
		T rhsY = rhs.key & y3_mask;
		T lhsZ = lhs.key & z3_mask;
		T rhsZ = rhs.key & z3_mask;
		return morton3d<T>(std::min(lhsX, rhsX) + std::min(lhsY, rhsY) + std::min(lhsZ, rhsZ));
	}

	/*
	  max(morton3(4,5,6), morton3(8,3,7)) == morton3(8,5,7);
	*/
	static inline morton3d max(const morton3d lhs, const morton3d rhs)
	{
		T lhsX = lhs.key & x3_mask;
		T rhsX = rhs.key & x3_mask;
		T lhsY = lhs.key & y3_mask;
		T rhsY = rhs.key & y3_mask;
		T lhsZ = lhs.key & z3_mask;
		T rhsZ = rhs.key & z3_mask;
		return morton3d<T>(std::max(lhsX, rhsX) + std::max(lhsY, rhsY) + std::max(lhsZ, rhsZ));
	}

private:
	inline uint64_t compactBits(uint64_t n) const
	{
		n &= 0x1249249249249249;
		n = (n ^ (n >> 2)) & 0x30c30c30c30c30c3;
		n = (n ^ (n >> 4)) & 0xf00f00f00f00f00f;
		n = (n ^ (n >> 8)) & 0x00ff0000ff0000ff;
		n = (n ^ (n >> 16)) & 0x00ff00000000ffff;
		n = (n ^ (n >> 32)) & 0x1fffff;
		return n;
	}

};


/* Add two morton keys (xyz interleaving)
  morton3(4,5,6) + morton3(1,2,3) == morton3(5,7,9);*/
template<class T>
inline morton3d<T> operator+(const morton3d<T> m1, const morton3d<T> m2)
{
	T x_sum = (m1.key | yz3_mask) + (m2.key & x3_mask);
	T y_sum = (m1.key | xz3_mask) + (m2.key & y3_mask);
	T z_sum = (m1.key | xy3_mask) + (m2.key & z3_mask);
	return morton3d<T>((x_sum & x3_mask) | (y_sum & y3_mask) | (z_sum & z3_mask));
}

/* Substract two morton keys (xyz interleaving)
   morton3(4,5,6) - morton3(1,2,3) == morton3(3,3,3);*/
template<class T>
inline morton3d<T> operator-(const morton3d<T> m1, const morton3d<T> m2)
{
	T x_diff = (m1.key & x3_mask) - (m2.key & x3_mask);
	T y_diff = (m1.key & y3_mask) - (m2.key & y3_mask);
	T z_diff = (m1.key & z3_mask) - (m2.key & z3_mask);
	return morton3d<T>((x_diff & x3_mask) | (y_diff & y3_mask) | (z_diff & z3_mask));
}

template<class T>
inline bool operator< (const morton3d<T>& lhs, const morton3d<T>& rhs)
{
	return (lhs.key) < (rhs.key);
}

template<class T>
inline bool operator> (const morton3d<T>& lhs, const morton3d<T>& rhs)
{
	return (lhs.key) > (rhs.key);
}

template<class T>
inline bool operator>= (const morton3d<T>& lhs, const morton3d<T>& rhs)
{
	return (lhs.key) >= (rhs.key);
}

template<class T>
inline bool operator<= (const morton3d<T>& lhs, const morton3d<T>& rhs)
{
	return (lhs.key) <= (rhs.key);
}

template<class T>
std::ostream& operator<<(std::ostream& os, const morton3d<T>& m)
{
	uint64_t x, y, z;
	m.decode(x, y, z);
	os << m.key << ": " << x << ", " << y << ", " << z;
	return os;
}

typedef morton3d<> morton3;

#endif
