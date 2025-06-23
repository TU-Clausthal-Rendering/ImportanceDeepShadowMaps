#include "Utils/Math/MathConstants.slangh"

import Scene.SceneTypes;
import Utils.Sampling.SampleGenerator;
import ParticleDataTypes;

cbuffer CB
{
    bool gReset;                //Should simulation be reseted
    uint gFrameCount;           //Frame Count for sample generator
    uint gParticleBufferOffset; // Offset in the global particle point buffer
    uint gNumParticles;     //Number of particles for this particle spawner

    float3 gRestPosition; // Position for inactive particles
    float gBaseRadius;    // Default radius for every active/inactive particle

    float gDeltaT;
    float3 gInitialPosition;

    float3 gInitialVelocity;
    float gVelocityRandom;

    float gMaxLifetime;
    float3 gWind;               //Wind
        
    float gGravity;
    float gGravityRandom;
    float gSpawnRadius;
    float gSpreadAngle;

    float gRadiusOffset; // Random Radius offset
    float gRandomLifetime; // Random Lifetime offset
    float gShrinkAtEnd; //Particle reduces size. Is based on the max possible radius
    uint _pad;
}

RWStructuredBuffer<ParticlePointDesc> gParticlePointDesc;
RWStructuredBuffer<ParticleAnimateData> gParticleAnimateData; //Extra per point data

static const float kMinRadius = 1e-7;

/** Generates a random spread direction with polar coordinates
    https://stackoverflow.com/questions/72006596/how-to-expand-particle-system-angle-spread-to-3-dimensions
*/
float3 rnd_dir(float3 t, float spread, inout SampleGenerator sg) {
    float a, r, x, y, z;
    a = sampleNext1D(sg) * spread * 0.5; // random angle within spread [0,PI]
    if (a > M_PI_2) {
        a = M_PI - a;
        a = sqrt(a * M_2_PI) *  M_PI_2;
        a = M_PI - a;
    } else {
        a = sqrt(a * M_2_PI) * M_PI_2;
    }
    x = cos(a);                         // convert angle to main axis coordinate
    r = sqrt(1.0 - (x * x));            // max radius of cone cap still inside unit sphere
    a = sampleNext1D(sg) * 2.0 * M_PI;  // random polar angle inside the cone cap [rad]
    y = r * cos(a);
    z = r * sin(a);

    //TBN Matrix 
    float3 b, n;
    t = normalize(t);
    n = float3(1, 0, 0);
    if (abs(dot(n, t)) > 0.75) n = float3(0, 1, 0);
    n = normalize(cross(n, t));
    b = normalize(cross(n, t));
    return (t * x) + (b * y) + (n * z);
}

/** Returns a random value in [-value, value]
*/
float getRandomValue(inout SampleGenerator sg, float value) {
    float rnd = (sampleNext1D(sg) * 2.0) - 1.0;
    return value * rnd;
}


[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint prtIdx = dispatchThreadId.x;
    if (prtIdx >= gNumParticles)
        return;

    SampleGenerator sg = SampleGenerator(uint2(prtIdx, 0), gFrameCount); // TODO add seed in pixel.y ?
    ParticlePointDesc particlePoint = gParticlePointDesc[gParticleBufferOffset + prtIdx];
    ParticleAnimateData particleData = gParticleAnimateData[gParticleBufferOffset + prtIdx];

    // Add delta time to lifetime
    particleData.lifetime -= gDeltaT;
    
    // Respawn particle
    if (gReset || !particleData.isValid) {
        particlePoint.position = gRestPosition;
        particleData.velocity = float3(0);
        particleData.isValid = false;
    }

    //Spawn new particle
    if (particleData.lifetime < 0) {
        particleData.lifetime += gMaxLifetime + getRandomValue(sg, gRandomLifetime);

        if (particleData.lifetime > 0) {
            float3 startPosition = gInitialPosition;
            float3 rnd = sampleNext3D(sg);
            rnd = (rnd * 2.0) - 1.0; // From [0,1] to [-1,1]
            particlePoint.position = startPosition + rnd * gSpawnRadius;
            particlePoint.radius = max(gBaseRadius + getRandomValue(sg, gRadiusOffset), kMinRadius);
            particleData.velocity = gInitialVelocity;
            particleData.isValid = true;
            if (gSpreadAngle > 0) {
                float lengthVel = length(particleData.velocity); // Get power of velocity
                float3 rndDir = rnd_dir(particleData.velocity, gSpreadAngle, sg);
                particleData.velocity = rndDir * lengthVel;
            }
            // Apply random velocity
            if (gVelocityRandom > 0.) {
                float velRnd = getRandomValue(sg, gVelocityRandom);
                particleData.velocity += particleData.velocity * velRnd;
            }
        } else {
            particlePoint.position = gRestPosition;
            particleData.velocity = float3(0);
            particleData.isValid = false;
        } 
    }

    // Update position and apply gravity
    if (particleData.isValid) {
        float gravRnd = 0.0;
        if (gGravityRandom > 0.0)
            gravRnd = getRandomValue(sg, gGravityRandom);

        particlePoint.position += particleData.velocity * gDeltaT;
        particleData.velocity.y -= (gGravity + gravRnd) * gDeltaT;
        particleData.velocity += gWind * gDeltaT;
        if (particleData.lifetime < gShrinkAtEnd) {
            float maxRadius = max(gBaseRadius + gRadiusOffset, kMinRadius);
            float normTimeDiff = (gShrinkAtEnd - particleData.lifetime) / gShrinkAtEnd;
            float newRad = lerp(maxRadius, kMinRadius, normTimeDiff);
            particlePoint.radius = min(particlePoint.radius, newRad);
        }
    }

    gParticlePointDesc[gParticleBufferOffset + prtIdx] = particlePoint;
    gParticleAnimateData[gParticleBufferOffset +prtIdx] = particleData;
}
