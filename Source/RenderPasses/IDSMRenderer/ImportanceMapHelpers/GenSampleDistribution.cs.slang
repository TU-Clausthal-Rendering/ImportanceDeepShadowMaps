#include "Utils/Math/MathConstants.slangh"

#ifndef COUNT_LIGHTS
#define COUNT_LIGHTS 1
#endif

#ifndef CLEAR_COUNTER
#define CLEAR_COUNTER 0
#endif

RWTexture2D<float> gSampleDistribution0[COUNT_LIGHTS];
RWTexture2D<float> gSampleDistribution1[COUNT_LIGHTS];

RWBuffer<uint> gCounter; // Can be nullptr;

static const bool kClearCounter = CLEAR_COUNTER;
cbuffer CB {
    uint2 gDispatchSize;
}

/*
    Compute Shader that calculates the Sample Distribution from the Sample Budget using the Rules from Chapter 3.1.3 in the Paper.
    The shader always works on the current Mipmap level and the one below, starting from level 1.
    The incoming Sample Budget is adjusted according to three cases:
    - Case (a):
        The sum of the Sample Budget is smaller than the number of valid budgets (budget > 0).
        All budgets are set to 0 and the sum is stored for the next iteration
        We do this, as we want to guarantee that each pixel that got importance is also traced
    - Case (b):
        The sum of the Sample Budget is bigger or equal the number of valid samples, guaranteeing that each pixel gets at least 1 ray
        If a sample has less than 1 budget it is increased to 1. The remaining weight is distributed proportional to the original weight.
        Every sample that has a remaining budget >1, the Grid requirement is applied (next lowest square; Eq.3 in paper).
        The grid requirement allows for uniform subdivision of the base Importance Map resolution.
    - Case (c):
        If a sample was previously balanced by case (b), it cannot be changed anymore as otherwise sampling for the corresponding elements below it would break.
        Balanced samples are marked with the sign bit. However, it may occur that some of the samples are marked, and others are not. In that case, we apply the grid requirement on
        all non-marked samples. If the sample is smaller than 1, it is increased to 1 to guarantee that at least one sample is distributed.

        
*/             
[numthreads(16, 16, 1)]
void main(uint3 thdID: SV_DispatchThreadID)
{
    const uint2 pixel = thdID.xy;
    const uint lightID = thdID.z;

    //In case WorkGroup size does not fit the dispatch size
    if (any(pixel >= gDispatchSize))
        return;

    // Load the 4 pixels from the mip before
    const uint2 pixelLower = pixel * 2;
    float4 values = float4(0);
    values[0] = gSampleDistribution0[lightID][pixelLower];
    values[1] = gSampleDistribution0[lightID][uint2(pixelLower.x + 1, pixelLower.y)];
    values[2] = gSampleDistribution0[lightID][uint2(pixelLower.x, pixelLower.y + 1)];
    values[3] = gSampleDistribution0[lightID][uint2(pixelLower.x + 1, pixelLower.y + 1)];

    //Get Properties from the 4 samples
    float sumOfSamples = 0.0;  //Summed Budget
    uint numValidSamples = 0;       //Samples with Budget > 0
    bool hasPrevBalanced = any(values < 0); //True if a pixel was balanced in the last iteration

    for (uint i = 0; i < 4; i++) {
        if (values[i] > 0) {
            numValidSamples++;
        }
        sumOfSamples += abs(values[i]);
    }

    // Check if balancing is possible (case b)
    bool balance = true;
    if (sumOfSamples < numValidSamples || hasPrevBalanced)
        balance = false;

    //Case (b)
    if (balance) {
        // Increase all weight smaller than 0 to 1
        float remainingWeight = sumOfSamples;
        float totalWeightOverOne = 0.0;
        for (uint i = 0; i < 4; i++) {
            if (values[i] > 0 && values[i] < 1.0) {
                remainingWeight -= 1.0;
                values[i] = 1.0;
            } else if (values[i] > 0) {
                remainingWeight -= 1.0;
                totalWeightOverOne += values[i] - 1.0;
            }
        }

        //Balance all remaining weights greater than 1
        sumOfSamples = 0; //Recalculate Sum
        for (uint i = 0; i < 4 && totalWeightOverOne > 0; i++) {
            if (values[i] > 1.0) {
                float rebalancedVal = 1.0 + max(((values[i] - 1.0) / totalWeightOverOne) * remainingWeight, 0.0);

                // Grid Requirement
                rebalancedVal = floor(sqrt(rebalancedVal));
                rebalancedVal *= rebalancedVal;

                values[i] = rebalancedVal;
            }
            sumOfSamples += values[i];
        }
    }
    //Case (c) Already balanced samples cannot be changed. Therefore apply the Grid requirement on all remaining samples
    else if (hasPrevBalanced) {
        sumOfSamples = 0; // Recalculate Sum
        [unroll]
        for (uint i = 0; i < 4; i++) {
            //Apply the grid requirement on all samples that where not changed until now
            if (values[i] > 0.0) {
                //Guarantee that each pixel that got importance is traced
                values[i] = max(values[i], 1.0);    
                // Grid Requirement
                values[i] = floor(sqrt(values[i]));
                values[i] *= values[i];
            }
            //Copy all previous balanced values, as they cannot be changed (as this would mess up traversal in the generation step)
            else if (values[i] < 0) {
                values[i] = abs(values[i]);
            }
            sumOfSamples  += values[i];
        }
    }
    //Case (a) Sum of all samples is less than the number of samples greater 0. 
    else {
        [unroll]
        for (uint i = 0; i < 4; i++) {
            values[i] = 0;
        }
    }

    //Write changed sample distribution values
    {
        gSampleDistribution0[lightID][pixelLower] = asfloat(uint(values[0]));
        gSampleDistribution0[lightID][uint2(pixelLower.x + 1, pixelLower.y)] = asfloat(uint(values[1]));
        gSampleDistribution0[lightID][uint2(pixelLower.x, pixelLower.y + 1)] = asfloat(uint(values[2]));
        gSampleDistribution0[lightID][uint2(pixelLower.x + 1, pixelLower.y + 1)] = asfloat(uint(values[3]));
    }

    // Store the sum of the samples for the next iteration
    // The sum is negative if it was balanced or previously balanced
    if (all(gDispatchSize == 1)) {
        gSampleDistribution1[lightID][pixel] = asfloat(uint(sumOfSamples));
    } else {
        bool storeAsNegative = balance || hasPrevBalanced;
        //storeAsNegative &= all(gDispatchSize > 1); // Last iteration, store positive
        gSampleDistribution1[lightID][pixel] = storeAsNegative ? -sumOfSamples : sumOfSamples;
    }
    

    // Clear for the head buffer of the linked list
    if (kClearCounter && all(gDispatchSize == 1))
        gCounter[lightID] = uint(sumOfSamples);
}
