#define MAX_ELEMENTS 1024 //Max number transparent hits
#define IMPORTANCE_MAX_ATOMIC_WEIGHT 63
#define MIN_OPACITY 1e-7
#define MAX_LOWEST_LEVEL_IMPORTANCE IMPORTANCE_MAX_ATOMIC_WEIGHT * 16384 //Avoid extreme outliners
//Compaire value for the optional mask. Value can be changed to 0.004 (~ 1./255) to delay 1 frame or 0.008 (~ 2./255) to delay 2 frames
#define TRANSPARENCY_MASK_COMPAIRE_VALUE 0.0 //0.004 0.008

//Mask Flags
#define TRANSPARENCY_MASK_FLAG_NONE 0
#define TRANSPARENCY_MASK_FLAG_USE_MASK 1
#define TRANSPARENCY_MASK_FLAG_USE_SM 2
