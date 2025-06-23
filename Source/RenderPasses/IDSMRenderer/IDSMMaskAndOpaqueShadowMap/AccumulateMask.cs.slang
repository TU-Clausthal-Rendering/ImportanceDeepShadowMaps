#include "../SharedShadowDefines.slang"

#ifndef COUNT_LIGHTS
#define COUNT_LIGHTS 1
#endif

Texture2DArray<float> gCurrentMask;
RWTexture2DArray<float> gTemporalMask;

cbuffer CB {
    uint3 gDispatchDims;    //Dispatch Dimension
    uint gCurrentBit;   //Current Bit
}

/** Makes the non-opaque objects mask temporal by storing the newest addition in the lowest bit.
* The mask is later compaired to 1/255 to delay everything by a frame (as the shadow in itself is delayed by 1 frame)
*/

[numthreads(16, 16, 1)]
void main(uint3 thdID: SV_DispatchThreadID)
{
    if (any(thdID > gDispatchDims))
        return;

    uint currentMask = uint(round(gCurrentMask[thdID])); //0 or 1
    uint temporalMask = uint(round(gTemporalMask[thdID] * 255.0)); //Uint [0,255]

    //Bitshif everything to the left
    temporalMask = (temporalMask >> 1) & 0x7F;
    temporalMask |= currentMask << 7;

    gTemporalMask[thdID] = temporalMask / 255.0; //Back to [0,1]
}
