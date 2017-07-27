/*
Adapting polynomial solvers from ./volumes/kernel/TorusImplementation2.h
attempting to get operational inside OptiX 
*/

__device__ __host__
static unsigned SolveCubic(float a, float b, float c, float *x) 
{
    // Find real solutions of the cubic equation : x^3 + a*x^2 + b*x + c = 0
    // Input: a,b,c
    // Output: x[3] real solutions
    // Returns number of real solutions (1 or 3)

    const float ott        = 1.f / 3.f; 
    const float sq3        = sqrt(3.f);
    const float inv6sq3    = 1.f / (6.f * sq3);
    unsigned int ireal = 1;
    float p                = b - a * a * ott;
    float q                = c - a * b * ott + 2.f * a * a * a * ott * ott * ott;
    float delta            = 4.f * p * p * p + 27.f * q * q;

    float t, u;

    if (delta >= 0.f) 
    {
        delta = sqrt(delta);
        t     = (-3.f * q * sq3 + delta) * inv6sq3;
        u     = (3.f * q * sq3 + delta) * inv6sq3;

        x[0]  = copysign(1.f, t) * cbrt(abs(t)) - copysign(1.f, u) * cbrt(abs(u)) - a * ott;
    } 
    else 
    {
        delta = sqrt(-delta);
        t     = -0.5f * q;
        u     = delta * inv6sq3;
        x[0]  = 2.f * pow(t * t + u * u, 0.5f * ott) * cos(ott * atan2(u, t));
        x[0] -= a * ott;
    }

    t     = x[0] * x[0] + a * x[0] + b;
    u     = a + x[0];
    delta = u * u - 4.f * t;

    if (delta >= 0.f) 
    {
        ireal = 3;
        delta = sqrt(delta);
        x[1]  = 0.5f * (-u - delta);
        x[2]  = 0.5f * (-u + delta);
    }
    return ireal;
}


__device__ __host__
static int SolveQuartic(float a, float b, float c, float d, float *x)
{
    // Find real solutions of the quartic equation : x^4 + a*x^3 + b*x^2 + c*x + d = 0
    // Input: a,b,c,d
    // Output: x[4] - real solutions
    // Returns number of real solutions (0 to 3)
    float e     = b - 3.f * a * a / 8.f;
    float f     = c + a * a * a / 8.f - 0.5f * a * b;
    float g     = d - 3.f * a * a * a * a / 256.f + a * a * b / 16.f - a * c / 4.f;
    float xx[4] = { 1e10f, 1e10f, 1e10f, 1e10f };
    float delta;
    float h = 0.f;
    unsigned ireal = 0;

    // special case when f is zero
    if (abs(f) < 1e-6f) 
    {
        delta = e * e - 4.f * g;
        if (delta < 0.f) return 0;
        delta = sqrt(delta);
        h  = 0.5f * (-e - delta);
        if (h >= 0.f)
        {
            h = sqrt(h);
            x[ireal++] = -h - 0.25f * a;
            x[ireal++] = h - 0.25f * a;
        }
        h = 0.5f * (-e + delta);
        if (h >= 0.f) 
        {
            h = sqrt(h);
            x[ireal++] = -h - 0.25f * a;
            x[ireal++] = h - 0.25f * a;
        }
        //Sort4(x);
        return ireal;
    }

    if (abs(g) < 1e-6f) 
    {
        x[ireal++] = -0.25f * a;
        // this actually wants to solve a second order equation
        // we should specialize if it happens often
        unsigned ncubicroots = SolveCubic(0, e, f, xx);
        // this loop is not nice
        for (unsigned i = 0; i < ncubicroots; i++) x[ireal++] = xx[i] - 0.25f * a;
        //Sort4(x); // could be Sort3
        return ireal;
    }

    ireal = SolveCubic(2.f * e, e * e - 4.f * g, -f * f, xx);

    if (ireal == 1) 
    {
        if (xx[0] <= 0.f) return 0;
        h = sqrt(xx[0]);
    } 
    else 
    {
        // 3 real solutions of the cubic
        for (unsigned i = 0; i < 3; i++) 
        {
            h = xx[i];
            if (h >= 0.f) break;
        }
        if (h <= 0.f) return 0;
        h = sqrt(h);
    }
    float j = 0.5f * (e + h * h - f / h);
    ireal = 0;

    delta = h * h - 4.f * j;
    if (delta >= 0.f) 
    {
        delta = sqrt(delta);
        x[ireal++] = 0.5f * (-h - delta) - 0.25f * a;
        x[ireal++] = 0.5f * (-h + delta) - 0.25f * a;
    }
    delta = h * h - 4.f * g / j;
    if (delta >= 0.f) 
    {
        delta = sqrt(delta);
        x[ireal++] = 0.5f * (h - delta) - 0.25f * a;
        x[ireal++] = 0.5f * (h + delta) - 0.25f * a;
    }
    //Sort4(x);
    return ireal;
}


