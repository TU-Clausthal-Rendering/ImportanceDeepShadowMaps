#pragma once
#include "../SharedShadowDefines.slang"

#ifndef COUNT_LIGHTS
#define COUNT_LIGHTS 1
#endif

#ifndef USE_ONE_BUFFER_FOR_ALL_LIGHTS
#define USE_ONE_BUFFER_FOR_ALL_LIGHTS 0
#endif

Texture2D<uint> gImportance[COUNT_LIGHTS];
Texture2D<uint> gTotalImportance[COUNT_LIGHTS]; //Highest Mip (Size: 1x1) with the total importance
RWTexture2D<float> gSampleBudget[COUNT_LIGHTS];   //Distributed Sample (Rays) texture
Texture2D<float> gTotalSampleBudget[COUNT_LIGHTS];

Buffer<uint> gNodeCount;         // Last frame node buffer element count
RWBuffer<uint> gLastFrameRayBudget; //Max number of distributed samples

static const uint kMaxAtomicWeight = IMPORTANCE_MAX_ATOMIC_WEIGHT; //Needs to be the same as in the shader where the atomic counter is increased.
static const uint kMinSamples = 4096;                              // Min 64x64 samples
static const uint kMaxSamples = MAX_RAY_SAMPLES; //Cannot distribute more than this per shadow map
static const bool kUseOneBufferForAllLights = USE_ONE_BUFFER_FOR_ALL_LIGHTS;

cbuffer CB {
    uint2 gDstSize;
    int gMaxNodeSize;                   //Total size of availible nodes
    float gChangePercentageIncrease;    //(0,1] how much can the ray dispatch can increase
    float gChangePercentageDecrease;    //(0,1] how much can the ray dispatch can decrease
    bool gDetermineTotalSampleBudget; // Should only be true on the max mip level
}

[numthreads(16, 16, 1)]
void main(uint3 thdID: SV_DispatchThreadID)
{
    const uint2 pixel = thdID.xy;
    if (any(pixel > gDstSize))
        return;

    // Determine the Total Sample Budget for this frame
    if (gDetermineTotalSampleBudget) {
        if (thdID.z > 0 || thdID.x > 0 || thdID.y > 0) //Only one thread is calculating the following
            return;

        // Get importance and ray count from each light
        uint rayCounts[COUNT_LIGHTS];
        float importances[COUNT_LIGHTS];
        float totalRayCount = 0;
        float totalImportance = 0.0;
        int elementCount = kUseOneBufferForAllLights ? gNodeCount[0] : 0;
        [unroll]
        for (uint i = 0; i < COUNT_LIGHTS; i++) {
            rayCounts[i] = gLastFrameRayBudget[i];
            // Fallback
            if (rayCounts[i] == 0)
                rayCounts[i] = gSampleBudget[i][uint2(0)];
            totalRayCount += rayCounts[i];
            importances[i] = gImportance[i][uint2(0)] / float(kMaxAtomicWeight);
            totalImportance += importances[i];
            if (!kUseOneBufferForAllLights)
                elementCount += gNodeCount[i];
        }

        // adjust ray count based on the fullness of the buffer
        int countDiff = gMaxNodeSize - elementCount;
        float diffPercentage = float(countDiff) / gMaxNodeSize;
        float rayCountChange = 0.0;
      
        rayCountChange = totalRayCount * diffPercentage * gChangePercentageDecrease;

        totalRayCount += rayCountChange;

        // Write new ray counts to textures        
        #if COUNT_LIGHTS == 1    
            float w = importances[0] / totalImportance;
            float newCount = min(floor(w * totalRayCount), kMaxSamples);
            if (importances[0] > 0)
                newCount = max(newCount, kMinSamples); // Force an minimum amount of samples TODO this can push total count over maximum, maybe balance this
            gSampleBudget[0][uint2(0)] = newCount;     // Write new total sample budget
            gLastFrameRayBudget[0] = uint(newCount);
        #else
        //For Multiple lights, check first if a light has not enough importance. Reduce the ray count from the total
        float newCounts[COUNT_LIGHTS];
        float newRayCount = totalRayCount;
        [unroll]
        for (uint i = 0; i < COUNT_LIGHTS; i++) {
            float w = importances[i] / totalImportance;
            newCounts[i] = w * totalRayCount;
            if (importances[i] > 0 && (newCounts[i] < kMinSamples)) {
                newRayCount -= (kMinSamples - newCounts[i]);
                newCounts[i] = -1.0;
            }
        }
        //Calculate ray counts with new adjusted budget
        [unroll]
        for (uint i = 0; i < COUNT_LIGHTS; i++) {
            float w = importances[i] / totalImportance;
            if (newCounts[i] > 0) {
                newCounts[i] = min(floor(w * newRayCount), kMaxSamples);
            } else if (newCounts[i] < 0) {
                newCounts[i] = kMinSamples;
            }
            gSampleBudget[i][uint2(0)] = newCounts[i]; // Write new total sample budget
            gLastFrameRayBudget[i] = uint(newCounts[i]);
        }
        #endif
       
    }
    //Distribute Total Sample Budget
    else {
        uint totalImportance = gTotalImportance[thdID.z][uint2(0)];         //Get full importance from the highest MIP
        float totalSampleCount = gTotalSampleBudget[thdID.z][uint2(0)];     //Get full Sample Budget from the highest MIP
        float sampleImportance = float(gImportance[thdID.z][pixel]);        //Get importance for the current sample

        float samples = 0.0;
        if (totalImportance > 0) {
            float weight = sampleImportance / (totalImportance);            
            samples = totalSampleCount * weight;
        }

        gSampleBudget[thdID.z][pixel] = samples;
    }
}
