/**
 * FLIP Fluid Simulation - Configuration
 *
 * Defines materials, their physical properties, and simulation constants.
 */

// =============================================================================
// GRID CONFIGURATION
// =============================================================================

export const GRID = {
    WIDTH: 256,
    HEIGHT: 256,
    WORKGROUP_SIZE: 8,
};

// =============================================================================
// MATERIAL TYPES
// =============================================================================

export const MATERIAL = {
    EMPTY: 0,      // Air/void
    WALL: 1,       // Solid immovable
    SAND: 2,       // Granular particle
    WATER: 3,      // Low viscosity liquid
    LAVA: 4,       // High viscosity liquid + heat source
    STEAM: 5,      // Gas, rises
    ROCK: 6,       // Cooled lava
};

// =============================================================================
// MATERIAL PROPERTIES
// =============================================================================

/**
 * Physical properties for each material.
 * These drive the simulation behavior.
 */
export const PROPERTIES = {
    [MATERIAL.EMPTY]: {
        name: 'Air',
        density: 0.001,        // Very light
        viscosity: 0.0,        // No resistance
        temperature: 20,       // Ambient temp (°C)
        isFluid: true,
        isSolid: false,
        heatCapacity: 1.0,     // How much heat it can hold
        heatConductivity: 0.1, // How fast heat spreads
    },
    [MATERIAL.WALL]: {
        name: 'Wall',
        density: 999,          // Immovable
        viscosity: 999,        // Doesn't flow
        temperature: 20,
        isFluid: false,
        isSolid: true,
        heatCapacity: 2.0,
        heatConductivity: 0.5,
    },
    [MATERIAL.SAND]: {
        name: 'Sand',
        density: 2.5,          // Sinks in water
        viscosity: 0,          // Particle, not fluid
        temperature: 20,
        isFluid: false,
        isSolid: false,        // Can be displaced
        heatCapacity: 0.8,
        heatConductivity: 0.3,
    },
    [MATERIAL.WATER]: {
        name: 'Water',
        density: 1.0,          // Baseline
        viscosity: 0.001,      // Flows freely
        temperature: 20,
        isFluid: true,
        isSolid: false,
        heatCapacity: 4.2,     // Water has high heat capacity
        heatConductivity: 0.6,
        boilTemp: 100,         // Becomes steam
        freezeTemp: 0,         // Becomes ice (future)
    },
    [MATERIAL.LAVA]: {
        name: 'Lava',
        density: 3.0,          // Heavy
        viscosity: 0.8,        // Very sticky
        temperature: 1200,     // Very hot!
        isFluid: true,
        isSolid: false,
        heatCapacity: 1.0,
        heatConductivity: 0.8,
        coolTemp: 700,         // Becomes rock below this
        emitsHeat: true,       // Constantly radiates heat
    },
    [MATERIAL.STEAM]: {
        name: 'Steam',
        density: 0.0006,       // Very light, rises
        viscosity: 0.0,        // Flows freely
        temperature: 100,
        isFluid: true,
        isSolid: false,
        heatCapacity: 2.0,
        heatConductivity: 0.2,
        condenseTemp: 100,     // Becomes water below this
    },
    [MATERIAL.ROCK]: {
        name: 'Rock',
        density: 2.8,
        viscosity: 999,        // Solid
        temperature: 400,      // Warm from cooling
        isFluid: false,
        isSolid: true,
        heatCapacity: 0.8,
        heatConductivity: 0.4,
        meltTemp: 1000,        // Becomes lava above this
    },
};

// =============================================================================
// COLORS (RGB 0-1)
// =============================================================================

export const COLORS = {
    [MATERIAL.EMPTY]: [0.05, 0.05, 0.1],
    [MATERIAL.WALL]: [0.4, 0.4, 0.45],
    [MATERIAL.SAND]: [0.9, 0.78, 0.4],
    [MATERIAL.WATER]: [0.2, 0.5, 0.9],
    [MATERIAL.LAVA]: [1.0, 0.3, 0.0],
    [MATERIAL.STEAM]: [0.8, 0.8, 0.85],
    [MATERIAL.ROCK]: [0.35, 0.3, 0.3],
};

// =============================================================================
// SIMULATION CONSTANTS
// =============================================================================

export const SIM = {
    // Pressure solver
    PRESSURE_ITERATIONS: 40,   // More = more accurate, slower
    OVERRELAXATION: 1.9,       // Speeds up convergence (1.0-2.0)

    // Fluid dynamics
    GRAVITY: 9.8,
    DT: 1/60,                  // Time step
    PARTICLES_PER_CELL: 9,
    PARTICLES_PER_CELL_SIDE: 3,
    FLIP_RATIO: 0.9,
    RENDER_THRESHOLD: 0.5,
    INTERACT_THRESHOLD: 0.05,
    REST_DENSITY: 1.0,
    DRIFT_SCALE: 0.8,
    PARTICLE_RADIUS: 0.15,
    MAX_CELL_PARTICLES: 32,
    SEPARATION_ITERS: 1,
    SEPARATION_STRENGTH: 0.2,

    // Temperature
    AMBIENT_TEMP: 20,
    HEAT_DIFFUSION: 0.1,       // How fast heat spreads
    THERMAL_BUOYANCY: 0.001,   // How much hot air rises

    // Mass
    MAX_MASS: 1.0,             // Max fluid mass per cell
    MIN_MASS: 0.01,            // Below this, cell becomes empty
};

// =============================================================================
// UI DEFAULTS
// =============================================================================

export const DEFAULTS = {
    simSpeed: 1,
    brushSize: 5,
    selectedMaterial: MATERIAL.WATER,
    isPaused: false,
    showPressure: false,
    showVelocity: false,
    showTemperature: false,
};
