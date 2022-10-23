#include <metal_stdlib>
#include <metal_math>
using namespace metal;

float starter(float ecc, float M, float ome) {
    const float FACTOR1 = 3 * M_PI_F / (M_PI_F - 6 / M_PI_F);
    const float FACTOR2 = 1.6 / (M_PI_F - 6 / M_PI_F);
    float M2 = M * M;
    float alpha = FACTOR1 + FACTOR2 * (M_PI_F - M) / (1 + ecc);
    float d = 3 * ome + alpha * ecc;
    float alphad = alpha * d;
    float r = (3 * alphad * (d - ome) + M2) * M;
    float q = 2 * alphad * ome - M2;
    float q2 = q * q;
    float w = pow(fabs(r) + sqrt(q2 * q + r * r), 2.0 / 3.0);
    return (2 * r * w / (w * w + w * q + q2) + M) / d;
}

float refine(float M, float ecc, float ome, float E) {
    float cE;
    float sE = sincos(E, cE);
    sE = E - sE;
    cE = 1 - cE;

    float f_0 = ecc * sE + E * ome - M;
    float f_1 = ecc * cE + ome;
    float f_2 = ecc * (E - sE);
    float f_3 = 1 - f_1;
    float d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1);
    float d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6);
    float d_42 = d_4 * d_4;
    float dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24);

    return E + dE;
}


kernel void work_on_arrays(device const float* ecc,
                           device const float* mean_anom,
                           device float* ecc_anom,
                           uint index [[thread_position_in_grid]])
{
    float ome = 1.0 - ecc[index];
    float start = starter(ecc[index], mean_anom[index], ome);
    ecc_anom[index] = refine(mean_anom[index], ecc[index], ome, start);
}

