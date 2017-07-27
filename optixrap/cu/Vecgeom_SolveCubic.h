
typedef float T ; 

unsigned int SolveCubic(T a, T b, T c, T *x) 
{
  // Find real solutions of the cubic equation : x^3 + a*x^2 + b*x + c = 0
  // Input: a,b,c
  // Output: x[3] real solutions
  // Returns number of real solutions (1 or 3)
  const T ott        = 1. / 3.; 
  const T sq3        = Sqrt(3.);
  const T inv6sq3    = 1. / (6. * sq3);
  unsigned int ireal = 1;
  T p                = b - a * a * ott;
  T q                = c - a * b * ott + 2. * a * a * a * ott * ott * ott;
  T delta            = 4 * p * p * p + 27. * q * q;
  T t, u;
  if (delta >= 0) {
    delta = Sqrt(delta);
    t     = (-3 * q * sq3 + delta) * inv6sq3;
    u     = (3 * q * sq3 + delta) * inv6sq3;
    x[0]  = CopySign(1., t) * Cbrt(Abs(t)) - CopySign(1., u) * Cbrt(Abs(u)) - a * ott;
  } else {
    delta = Sqrt(-delta);
    t     = -0.5 * q;
    u     = delta * inv6sq3;
    x[0]  = 2. * Pow(t * t + u * u, 0.5 * ott) * cos(ott * ATan2(u, t));
    x[0] -= a * ott;
  }

  t     = x[0] * x[0] + a * x[0] + b;
  u     = a + x[0];
  delta = u * u - 4. * t;
  if (delta >= 0) {
    ireal = 3;
    delta = Sqrt(delta);
    x[1]  = 0.5 * (-u - delta);
    x[2]  = 0.5 * (-u + delta);
  }
  return ireal;
}

