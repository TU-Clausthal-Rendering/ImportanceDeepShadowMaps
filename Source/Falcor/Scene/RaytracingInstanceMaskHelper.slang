
//Defines that can be used for the instance mask

static const uint kInstanceMaskBitNormalGeometry = 1; //Instance Mask Bit with all geometry without special properties
static const uint kInstanceMaskBitCastShadow = 2; //Instance Mask Bit with all geometry that should not cast a shadow (exclude for shadow ray)
static const uint kInstanceMaskBitParticleCamera = 4; // Instance Mask Bit with the particles facing the active camera
static const uint kInstanceMaskBitParticleXY = 8;     // Instance Mask Bit with the particles in XY-World Plane
static const uint kInstanceMaskBitParticleYZ = 16;    // Instance Mask Bit with the particles in YZ-World Plane
static const uint kInstanceMaskBitParticleXZ = 32;    // Instance Mask Bit with the particles in XZ-World Plane

static const uint kShadowRayInstanceMask = 0xFD; // Instance Mask without castShadow geometry (legacy)

static const uint kInstanceNoParticle = 0x3;        //Normal and Cast shadow
static const uint kInstanceCameraParticle = 0x07;   //Camera Particle
static const uint kInstanceShadowParticleXY = 0x09; // Particle Shadow Ray XY Plane
static const uint kInstanceShadowParticleYZ = 0x11; // Particle Shadow Ray YZ Plane
static const uint kInstanceShadowParticleXZ = 0x21; // Particle Shadow Ray XZ Plane

/** Helper to get the best fitting shadow plane from a normalized direction
* \param[in] normalized direction
* \return instance mask escluding non shadowable geometry and all particles except the selcted plane
*/
uint getBestParticleShadowPlane(float3 direction, bool addShadowBit = false) {
    uint instanceMask = kInstanceMaskBitNormalGeometry;
    if(addShadowBit)
        instanceMask |= kInstanceMaskBitCastShadow;
    uint particleMask = kInstanceMaskBitParticleXY;

    float xy = abs(dot(direction, float3(0, 0, 1)));
    float yz = abs(dot(direction, float3(1, 0, 0)));
    float xz = abs(dot(direction, float3(0, 1, 0)));

    if (yz > xy) {
        xy = yz;
        particleMask = kInstanceMaskBitParticleYZ;
    }
    if (xz > xy)
        particleMask = kInstanceMaskBitParticleXZ;

    return instanceMask | particleMask;
}
