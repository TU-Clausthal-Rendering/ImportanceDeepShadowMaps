#include "../SharedShadowDefines.slang"

#ifndef COUNT_LIGHTS
#define COUNT_LIGHTS 1
#endif

Texture2D<uint> gSrc[COUNT_LIGHTS];
RWTexture2D<uint> gDst[COUNT_LIGHTS];

cbuffer CB {
    uint2 gDstSize;
}

/* Reduce using a Mip-Chain. Calculates the Total Importance for the Importance Map.
   gSrc is the lower Mip, gDst is one Mip higher.
   The dispatch dimensions are derived from gDst. 
*/

[numthreads(16, 16, 1)]
void main(uint3 thdID: SV_DispatchThreadID)
{
    const uint2 pixel = thdID.xy;
    if (any(pixel > gDstSize))
        return;

    uint2 srcIdx = pixel * 2;

    uint sum = 0;
    sum += gSrc[thdID.z][uint2(srcIdx.x, srcIdx.y)];
    sum += gSrc[thdID.z][uint2(srcIdx.x + 1, srcIdx.y)];
    sum += gSrc[thdID.z][uint2(srcIdx.x, srcIdx.y + 1)];
    sum += gSrc[thdID.z][uint2(srcIdx.x + 1, srcIdx.y + 1)];

    gDst[thdID.z][pixel] = sum;
}
