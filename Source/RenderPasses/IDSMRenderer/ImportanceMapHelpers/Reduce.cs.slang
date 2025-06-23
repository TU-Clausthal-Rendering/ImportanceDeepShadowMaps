#include "../SharedShadowDefines.slang"

#ifndef COUNT_LIGHTS
#define COUNT_LIGHTS 1
#endif

#define WK_SIZE 16
#define WK_TOTAL WK_SIZE * WK_SIZE

Texture2D<uint> gSrc[COUNT_LIGHTS];
RWTexture2D<uint> gDst[COUNT_LIGHTS];

cbuffer CB {
    uint2 gDstSize;
}

groupshared uint sdata[WK_TOTAL];

// "Explodes" an integer, i.e. inserts a 0 between each bit.  Takes inputs up to 16 bit wide.
//      For example, 0b11111111 -> 0b1010101010101010
uint integerExplode(uint x)
{
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

// Reverse of integerExplode, i.e. takes every other bit in the integer and compresses
// those bits into a dense bit field. Takes 32-bit inputs, produces 16-bit outputs.
//    For example, 0b'abcdefgh' -> 0b'0000bdfh'
uint integerCompact(uint x)
{
    x = (x & 0x11111111) | ((x & 0x44444444) >> 1);
    x = (x & 0x03030303) | ((x & 0x30303030) >> 2);
    x = (x & 0x000F000F) | ((x & 0x0F000F00) >> 4);
    x = (x & 0x000000FF) | ((x & 0x00FF0000) >> 8);
    return x;
}

// Converts a 2D position to a linear index following a Z-curve pattern.
uint zCurveToLinearIndex(uint2 xy)
{
    return integerExplode(xy[0]) | (integerExplode(xy[1]) << 1);
}

// Converts a linear to a 2D position following a Z-curve pattern.
uint2 linearIndexToZCurve(uint index)
{
    return uint2(
        integerCompact(index),
        integerCompact(index >> 1));
}

/*
    Parallel Reduction. Calculates the Total Importance for the Importance Map
    The 2D index of the texture is converted to 1D using a Z-curve Pattern.
    Each Workgroup calculates the sum of the whole group.
    gDst is the Mip-Level corresponding to the sum Region. (+5 Mip Levels)
*/
[numthreads(WK_SIZE, WK_SIZE, 1)]
void main(uint3 dispatchID: SV_DispatchThreadID, uint3 localInvocationID: SV_GroupThreadID, uint3 workGroupID: SV_GroupID)
{
    const uint2 pixel = dispatchID.xy;
    const uint lightIdx = dispatchID.z;
    if (any(pixel > gDstSize))
        return;

    // Get Sum of 2x2 field
    uint2 startIdx = pixel * 2u;
    uint startSum = 0;
    startSum += gSrc[lightIdx][startIdx];
    startSum += gSrc[lightIdx][startIdx + uint2(1, 0)];
    startSum += gSrc[lightIdx][startIdx + uint2(0, 1)];
    startSum += gSrc[lightIdx][startIdx + uint2(1, 1)];
    
    //In case this was dispatched from the second hightest MIP
    if (gDstSize.x == 1) {
        gDst[lightIdx][workGroupID.xy] = startSum;
        return;
    }

    //Load into shared memory
    uint tid = zCurveToLinearIndex(localInvocationID.xy);
    sdata[tid] = startSum;
    GroupMemoryBarrierWithGroupSync();

    // Reduce with Sequential Addressing
    const uint maxLoop = min(WK_TOTAL, gDstSize.x * gDstSize.y); //Dispatch can theoretically be smaller than Workgroup size
    for (uint s = maxLoop / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // TODO last iterations of the loop could be implemented using WaveOP (WaveActiveSum; WavePrefixSum)
    // https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12#wave-reduction
    // However, due to z curve adressing it seems that [0,1, ... , 32] is not in the same Wave, which results in an incorrect Total Importance

    //First thread writes. WorkGroupID is identical to the MIP bound to gDst
    if (tid == 0)
        gDst[lightIdx][workGroupID.xy] = sdata[0];
}
