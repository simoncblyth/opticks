// wavelengthToSRGB.glsl

float cie1931_x(float wave) {
    float t1 = (wave - 442.0) * ((wave < 442.0) ? 0.0624 : 0.0374);
    float t2 = (wave - 599.8) * ((wave < 599.8) ? 0.0264 : 0.0323);
    float t3 = (wave - 501.1) * ((wave < 501.1) ? 0.0490 : 0.0382);
    return 0.362 * exp(-0.5 * t1 * t1) + 1.056 * exp(-0.5 * t2 * t2) - 0.065 * exp(-0.5 * t3 * t3);
}

float cie1931_y(float wave) {
    float t1 = (wave - 568.8) * ((wave < 568.8) ? 0.0213 : 0.0247);
    float t2 = (wave - 530.9) * ((wave < 530.9) ? 0.0613 : 0.0322);
    return 0.821 * exp(-0.5 * t1 * t1) + 0.286 * exp(-0.5 * t2 * t2);
}

float cie1931_z(float wave) {
    float t1 = (wave - 437.0) * ((wave < 437.0) ? 0.0845 : 0.0278);
    float t2 = (wave - 459.0) * ((wave < 459.0) ? 0.0385 : 0.0725);
    return 1.217 * exp(-0.5 * t1 * t1) + 0.681 * exp(-0.5 * t2 * t2);
}

vec3 cie1931_XYZ(float wave) {
    return vec3(cie1931_x(wave), cie1931_y(wave), cie1931_z(wave));
}


vec3 linearToSRGB(vec3 rgbLinear) {

    // Human visual perception of brightness is non-linear
    // with far more sensitivity to relative differences in dark tones than in bright tones.
    // Below piecewise function follows IEC 61966-2-1 sRGB standard commonly used for monitor display

    vec3 linearSegment = rgbLinear * 12.92;
    vec3 gammaSegment  = 1.055 * pow(max(rgbLinear, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;

    // Select component-wise based on threshold 0.0031308
    return mix(gammaSegment, linearSegment, step(rgbLinear, vec3(0.0031308)));
}

vec3 xyzToSRGB(vec3 XYZ) {
    // CIE XYZ to Linear sRGB transformation matrix (D65 white point)
    mat3 XYZ_TO_sRGB = mat3(
         3.2404542, -0.9692660,  0.0556434,
        -1.5371385,  1.8760108, -0.2040259,
        -0.4985314,  0.0415560,  1.0572252
    );

    vec3 rgbLinear = XYZ_TO_sRGB * XYZ;

    // Desaturate out-of-gamut values by shifting baseline
    float minVal = min(0.0, min(rgbLinear.r, min(rgbLinear.g, rgbLinear.b)));
    rgbLinear -= vec3(minVal);

    return clamp(linearToSRGB(rgbLinear), 0.0, 1.0);
}

vec3 wavelengthToSRGB(float wavelength_nm)
{
    vec3 XYZ = cie1931_XYZ(wavelength_nm);
    vec3 rgb = xyzToSRGB(XYZ);
    return rgb ;
}


