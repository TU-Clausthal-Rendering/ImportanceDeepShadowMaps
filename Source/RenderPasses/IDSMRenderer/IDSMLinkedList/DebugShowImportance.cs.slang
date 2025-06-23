cbuffer CB
{
    uint2 gDispatchSize; // Target Dim
    uint2 gSMRes;        // Render Dim
    float gScaleFactorIM;  // Scale Factor
    float gScaleFactorSD;  // Scale Factor
}

// Inputs
Texture2D<int> gSampleDistribution;
Texture2D<uint> gImportanceMap;
Texture2D<float> gMask;

// Outputs
RWTexture2D<float4> gDebug;

static const float4 kClearColor = float4(0, 0, 0, 1);
static const float4 kBorderColor = float4(0, 1, 0, 1);

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    //Out of bounds
    if (any(pixel >= gDispatchSize))
        return;

    uint2 renderArea1 = uint2(gSMRes.x * 2 + 1, gSMRes.y);
    uint2 renderArea2 = uint2(gSMRes.x, gSMRes.y * 2 + 1);

    // Clear
    if (any(pixel > renderArea1) && any(pixel > renderArea2)) {
        gDebug[pixel] = kClearColor;
        return;
    }

    // Border
    if (any(pixel == uint2(0)) || any(pixel == (gSMRes + 1)) || any(pixel == renderArea1) || any(pixel == renderArea2)) {
        gDebug[pixel] = kBorderColor;
        return;
    }

    bool isSampleDistribution = pixel.x > gSMRes.x;
    bool isMask = pixel.y > gSMRes.y;

    // Copy and scale
    float importanceValue = 0.0;
    if (isMask) {
        uint2 fetchPixel = uint2(pixel.x - 1, pixel.y - gSMRes.y - 2);
        importanceValue = gMask[fetchPixel];
    }
    else if (isSampleDistribution) {
        uint2 fetchPixel = uint2(pixel.x - gSMRes.x - 2, pixel.y - 1);
        importanceValue = gSampleDistribution[fetchPixel] / gScaleFactorSD;
    }
    else {
        uint2 fetchPixel = pixel - 1;
        importanceValue = gImportanceMap[fetchPixel] / gScaleFactorIM;
    }
       
    gDebug[pixel] = float4(importanceValue, importanceValue, importanceValue, 1.0);
}
