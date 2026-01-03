/**
 * FLIP Fluid Simulation - WGSL Shaders (Particles + Grid)
 */

import { GRID, MATERIAL, COLORS, SIM } from './config.js';

const commonCode = `
const WIDTH: i32 = ${GRID.WIDTH};
const HEIGHT: i32 = ${GRID.HEIGHT};
const U_WIDTH: i32 = WIDTH + 1;
const U_HEIGHT: i32 = HEIGHT;
const V_WIDTH: i32 = WIDTH;
const V_HEIGHT: i32 = HEIGHT + 1;

const PARTICLES_PER_CELL: u32 = ${SIM.PARTICLES_PER_CELL}u;
const PARTICLES_PER_CELL_SIDE: i32 = ${SIM.PARTICLES_PER_CELL_SIDE};
const RENDER_THRESHOLD: f32 = ${SIM.RENDER_THRESHOLD};
const INTERACT_THRESHOLD: f32 = ${SIM.INTERACT_THRESHOLD};
const REST_DENSITY: f32 = ${SIM.REST_DENSITY};
const DRIFT_SCALE: f32 = ${SIM.DRIFT_SCALE};
const PARTICLE_RADIUS: f32 = ${SIM.PARTICLE_RADIUS};
const MAX_CELL_PARTICLES: u32 = ${SIM.MAX_CELL_PARTICLES}u;
const SEPARATION_STRENGTH: f32 = ${SIM.SEPARATION_STRENGTH};

const FIXED_SCALE_F: f32 = 65536.0;
const FIXED_SCALE_I: i32 = 65536;
const FIXED_SCALE_U: u32 = 65536u;

const EMPTY: u32 = ${MATERIAL.EMPTY}u;
const WALL: u32 = ${MATERIAL.WALL}u;
const SAND: u32 = ${MATERIAL.SAND}u;
const WATER: u32 = ${MATERIAL.WATER}u;
const LAVA: u32 = ${MATERIAL.LAVA}u;
const STEAM: u32 = ${MATERIAL.STEAM}u;
const ROCK: u32 = ${MATERIAL.ROCK}u;

const MAX_PARTICLES: u32 = u32(WIDTH * HEIGHT) * PARTICLES_PER_CELL;

fn inBounds(pos: vec2i) -> bool {
    return pos.x >= 0 && pos.x < WIDTH && pos.y >= 0 && pos.y < HEIGHT;
}

fn inBoundsU(pos: vec2i) -> bool {
    return pos.x >= 0 && pos.x < U_WIDTH && pos.y >= 0 && pos.y < U_HEIGHT;
}

fn inBoundsV(pos: vec2i) -> bool {
    return pos.x >= 0 && pos.x < V_WIDTH && pos.y >= 0 && pos.y < V_HEIGHT;
}

fn cellIndex(pos: vec2i) -> u32 {
    return u32(pos.y * WIDTH + pos.x);
}

fn isSolid(t: u32) -> bool {
    return t == WALL || t == ROCK || t == SAND;
}

fn isFluidMaterial(t: u32) -> bool {
    return t == WATER || t == LAVA || t == STEAM;
}
`;

// =============================================================================
// FREE LIST SHADERS
// =============================================================================

export const clearFreeListShader = `
${commonCode}

@group(0) @binding(0) var<storage, read_write> freeCount: atomic<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x == 0u) {
        atomicStore(&freeCount, 0u);
    }
}
`;

export const buildFreeListShader = `
${commonCode}

@group(0) @binding(0) var<storage, read_write> freeCount: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> freeList: array<u32>;
@group(0) @binding(2) var<storage, read> particleMass: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= MAX_PARTICLES) { return; }

    if (particleMass[idx] <= 0.0) {
        let slot = atomicAdd(&freeCount, 1u);
        if (slot < MAX_PARTICLES) {
            freeList[slot] = idx;
        }
    }
}
`;

// =============================================================================
// SPAWN SHADER - Updates material and spawns/removes water particles
// =============================================================================

export const spawnParticlesShader = `
${commonCode}

@group(0) @binding(0) var spawnTex: texture_storage_2d<r32uint, read>;
@group(0) @binding(1) var materialIn: texture_storage_2d<r32uint, read>;
@group(0) @binding(2) var materialOut: texture_storage_2d<r32uint, write>;
@group(0) @binding(3) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> particleMass: array<f32>;
@group(0) @binding(6) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> cellParticles: array<u32>;
@group(0) @binding(8) var<storage, read_write> freeCount: atomic<u32>;
@group(0) @binding(9) var<storage, read> freeList: array<u32>;

fn slotOffset(slot: u32) -> vec2f {
    let row = i32(slot) / PARTICLES_PER_CELL_SIDE;
    let col = i32(slot) % PARTICLES_PER_CELL_SIDE;
    let fx = (f32(col) + 0.5) / f32(PARTICLES_PER_CELL_SIDE);
    let fy = (f32(row) + 0.5) / f32(PARTICLES_PER_CELL_SIDE);
    return vec2f(fx, fy);
}

fn popFreeIndex() -> i32 {
    var old = atomicLoad(&freeCount);
    loop {
        if (old == 0u) { return -1; }
        let res = atomicCompareExchangeWeak(&freeCount, old, old - 1u);
        if (res.exchanged) {
            return i32(freeList[old - 1u]);
        }
        old = res.old_value;
    }
}

fn clearCellParticles(cell: vec2i) {
    let cIdx = cellIndex(cell);
    let count = min(atomicLoad(&cellCounts[cIdx]), MAX_CELL_PARTICLES);
    for (var i: u32 = 0u; i < count; i++) {
        let idx = cellParticles[cIdx * MAX_CELL_PARTICLES + i];
        particleMass[idx] = 0.0;
        particleVel[idx] = vec2f(0.0, 0.0);
    }
}

@compute @workgroup_size(${GRID.WORKGROUP_SIZE}, ${GRID.WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pos = vec2i(i32(id.x), i32(id.y));
    if (!inBounds(pos)) { return; }

    let spawn = textureLoad(spawnTex, pos).r;
    let currentMat = textureLoad(materialIn, pos).r;
    var resultMat = currentMat;

    if (spawn == 255u) {
        resultMat = EMPTY;
        clearCellParticles(pos);
    } else if (spawn == WALL || spawn == SAND || spawn == ROCK) {
        resultMat = spawn;
        clearCellParticles(pos);
    } else if (spawn == WATER) {
        if (!isSolid(currentMat)) {
            let cIdx = cellIndex(pos);
            let existing = min(atomicLoad(&cellCounts[cIdx]), PARTICLES_PER_CELL);
            let toSpawn = PARTICLES_PER_CELL - existing;
            let base = vec2f(f32(pos.x), f32(pos.y));
            let mass = 1.0 / f32(PARTICLES_PER_CELL);
            for (var i: u32 = 0u; i < toSpawn; i++) {
                let idx = popFreeIndex();
                if (idx < 0) { break; }
                let offset = slotOffset(i);
                particlePos[u32(idx)] = base + offset;
                particleVel[u32(idx)] = vec2f(0.0, 0.0);
                particleMass[u32(idx)] = mass;
            }
        }
    }

    textureStore(materialOut, pos, vec4u(resultMat));
}
`;

// =============================================================================
// INTEGRATE PARTICLES SHADER
// =============================================================================

export const integrateParticlesShader = `
${commonCode}

struct Params {
    dt: f32,
    gravity: f32,
    viscosityScale: f32,
    flipRatio: f32,
    frame: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var materialTex: texture_storage_2d<r32uint, read>;
@group(0) @binding(2) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> particleMass: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= MAX_PARTICLES) { return; }

    let mass = particleMass[idx];
    if (mass <= 0.0) { return; }

    var pos = particlePos[idx];
    var vel = particleVel[idx];
    let oldPos = pos;

    vel.y += params.gravity * params.dt;

    let damping = 1.0 / (1.0 + params.viscosityScale * params.dt);
    vel *= damping;

    pos += vel * params.dt;

    if (pos.x < 0.001) {
        pos.x = 0.001;
        vel.x = 0.0;
    }
    if (pos.x > f32(WIDTH) - 0.001) {
        pos.x = f32(WIDTH) - 0.001;
        vel.x = 0.0;
    }
    if (pos.y < 0.001) {
        pos.y = 0.001;
        vel.y = 0.0;
    }
    if (pos.y > f32(HEIGHT) - 0.001) {
        pos.y = f32(HEIGHT) - 0.001;
        vel.y = 0.0;
    }

    let cell = vec2i(i32(floor(pos.x)), i32(floor(pos.y)));
    if (inBounds(cell)) {
        let mat = textureLoad(materialTex, cell).r;
        if (isSolid(mat)) {
            pos = oldPos;
            vel = vec2f(0.0, 0.0);
        }
    }

    particlePos[idx] = pos;
    particleVel[idx] = vel;
}
`;

// =============================================================================
// CLEAR PARTICLE GRID (CELL COUNTS)
// =============================================================================

export const clearParticleGridShader = `
${commonCode}

@group(0) @binding(0) var<storage, read_write> cellCounts: array<atomic<u32>>;

const CELL_COUNT: u32 = u32(WIDTH * HEIGHT);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= CELL_COUNT) { return; }
    atomicStore(&cellCounts[idx], 0u);
}
`;

// =============================================================================
// BIN PARTICLES INTO CELLS
// =============================================================================

export const binParticlesShader = `
${commonCode}

@group(0) @binding(0) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cellParticles: array<u32>;
@group(0) @binding(2) var<storage, read> particlePos: array<vec2f>;
@group(0) @binding(3) var<storage, read> particleMass: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= MAX_PARTICLES) { return; }

    let mass = particleMass[idx];
    if (mass <= 0.0) { return; }

    let pos = particlePos[idx];
    let cell = vec2i(
        clamp(i32(floor(pos.x)), 0, WIDTH - 1),
        clamp(i32(floor(pos.y)), 0, HEIGHT - 1)
    );

    let cIdx = cellIndex(cell);
    let slot = atomicAdd(&cellCounts[cIdx], 1u);
    if (slot < MAX_CELL_PARTICLES) {
        cellParticles[cIdx * MAX_CELL_PARTICLES + slot] = idx;
    }
}
`;

// =============================================================================
// SEPARATE PARTICLES (REDUCE CLUMPING)
// =============================================================================

export const separateParticlesShader = `
${commonCode}

@group(0) @binding(0) var materialTex: texture_storage_2d<r32uint, read>;
@group(0) @binding(1) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cellParticles: array<u32>;
@group(0) @binding(3) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(4) var<storage, read> particleMass: array<f32>;

fn getMaterial(pos: vec2i) -> u32 {
    if (!inBounds(pos)) { return WALL; }
    return textureLoad(materialTex, pos).r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= MAX_PARTICLES) { return; }

    let mass = particleMass[idx];
    if (mass <= 0.0) { return; }

    var pos = particlePos[idx];
    let baseCell = vec2i(i32(floor(pos.x)), i32(floor(pos.y)));
    let restDist = 2.0 * PARTICLE_RADIUS;
    let restDist2 = restDist * restDist;

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let cell = baseCell + vec2i(dx, dy);
            if (!inBounds(cell)) { continue; }
            let cIdx = cellIndex(cell);
            let count = min(atomicLoad(&cellCounts[cIdx]), MAX_CELL_PARTICLES);
            for (var i: u32 = 0u; i < count; i++) {
                let other = cellParticles[cIdx * MAX_CELL_PARTICLES + i];
                if (other == idx) { continue; }
                let otherPos = particlePos[other];
                let delta = pos - otherPos;
                let dist2 = dot(delta, delta);
                if (dist2 > 0.0 && dist2 < restDist2) {
                    let dist = sqrt(dist2);
                    let push = (restDist - dist) * 0.5 * SEPARATION_STRENGTH;
                    pos += (delta / dist) * push;
                }
            }
        }
    }

    pos.x = clamp(pos.x, 0.001, f32(WIDTH) - 0.001);
    pos.y = clamp(pos.y, 0.001, f32(HEIGHT) - 0.001);

    let cell = vec2i(i32(floor(pos.x)), i32(floor(pos.y)));
    if (inBounds(cell) && !isSolid(getMaterial(cell))) {
        particlePos[idx] = pos;
    }
}
`;

// =============================================================================
// CLEAR GRID ACCUMULATORS SHADER
// =============================================================================

export const clearGridShader = `
${commonCode}

@group(0) @binding(0) var<storage, read_write> densitySum: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> uSum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> uWeight: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> vSum: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> vWeight: array<atomic<u32>>;

const DENSITY_SIZE: u32 = u32(WIDTH * HEIGHT);
const U_SIZE: u32 = u32(U_WIDTH * U_HEIGHT);
const V_SIZE: u32 = u32(V_WIDTH * V_HEIGHT);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx < DENSITY_SIZE) {
        atomicStore(&densitySum[idx], 0u);
    }
    if (idx < U_SIZE) {
        atomicStore(&uSum[idx], 0);
        atomicStore(&uWeight[idx], 0u);
    }
    if (idx < V_SIZE) {
        atomicStore(&vSum[idx], 0);
        atomicStore(&vWeight[idx], 0u);
    }
}
`;

// =============================================================================
// DEPOSIT PARTICLES TO GRID (DENSITY + VELOCITY)
// =============================================================================

export const depositParticlesShader = `
${commonCode}

@group(0) @binding(0) var<storage, read_write> densitySum: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> uSum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> uWeight: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> vSum: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> vWeight: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(6) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(7) var<storage, read_write> particleMass: array<f32>;

fn depositDensity(pos: vec2f, mass: f32) {
    let x = pos.x - 0.5;
    let y = pos.y - 0.5;
    let x0 = clamp(i32(floor(x)), 0, WIDTH - 1);
    let y0 = clamp(i32(floor(y)), 0, HEIGHT - 1);
    let x1 = clamp(x0 + 1, 0, WIDTH - 1);
    let y1 = clamp(y0 + 1, 0, HEIGHT - 1);

    let fx = x - f32(x0);
    let fy = y - f32(y0);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let m00 = u32(round(w00 * mass * FIXED_SCALE_F));
    let m10 = u32(round(w10 * mass * FIXED_SCALE_F));
    let m01 = u32(round(w01 * mass * FIXED_SCALE_F));
    let m11 = u32(round(w11 * mass * FIXED_SCALE_F));

    let idx00 = u32(y0 * WIDTH + x0);
    let idx10 = u32(y0 * WIDTH + x1);
    let idx01 = u32(y1 * WIDTH + x0);
    let idx11 = u32(y1 * WIDTH + x1);

    atomicAdd(&densitySum[idx00], m00);
    atomicAdd(&densitySum[idx10], m10);
    atomicAdd(&densitySum[idx01], m01);
    atomicAdd(&densitySum[idx11], m11);
}

fn depositU(pos: vec2f, vel: f32, mass: f32) {
    let x = pos.x;
    let y = pos.y - 0.5;
    let x0 = clamp(i32(floor(x)), 0, U_WIDTH - 2);
    let y0 = clamp(i32(floor(y)), 0, U_HEIGHT - 2);
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - f32(x0);
    let fy = y - f32(y0);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let w00u = u32(round(w00 * mass * FIXED_SCALE_F));
    let w10u = u32(round(w10 * mass * FIXED_SCALE_F));
    let w01u = u32(round(w01 * mass * FIXED_SCALE_F));
    let w11u = u32(round(w11 * mass * FIXED_SCALE_F));

    let v00i = i32(round(vel * f32(w00u)));
    let v10i = i32(round(vel * f32(w10u)));
    let v01i = i32(round(vel * f32(w01u)));
    let v11i = i32(round(vel * f32(w11u)));

    let idx00 = u32(y0 * U_WIDTH + x0);
    let idx10 = u32(y0 * U_WIDTH + x1);
    let idx01 = u32(y1 * U_WIDTH + x0);
    let idx11 = u32(y1 * U_WIDTH + x1);

    atomicAdd(&uWeight[idx00], w00u);
    atomicAdd(&uWeight[idx10], w10u);
    atomicAdd(&uWeight[idx01], w01u);
    atomicAdd(&uWeight[idx11], w11u);

    atomicAdd(&uSum[idx00], v00i);
    atomicAdd(&uSum[idx10], v10i);
    atomicAdd(&uSum[idx01], v01i);
    atomicAdd(&uSum[idx11], v11i);
}

fn depositV(pos: vec2f, vel: f32, mass: f32) {
    let x = pos.x - 0.5;
    let y = pos.y;
    let x0 = clamp(i32(floor(x)), 0, V_WIDTH - 2);
    let y0 = clamp(i32(floor(y)), 0, V_HEIGHT - 2);
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - f32(x0);
    let fy = y - f32(y0);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let w00u = u32(round(w00 * mass * FIXED_SCALE_F));
    let w10u = u32(round(w10 * mass * FIXED_SCALE_F));
    let w01u = u32(round(w01 * mass * FIXED_SCALE_F));
    let w11u = u32(round(w11 * mass * FIXED_SCALE_F));

    let v00i = i32(round(vel * f32(w00u)));
    let v10i = i32(round(vel * f32(w10u)));
    let v01i = i32(round(vel * f32(w01u)));
    let v11i = i32(round(vel * f32(w11u)));

    let idx00 = u32(y0 * V_WIDTH + x0);
    let idx10 = u32(y0 * V_WIDTH + x1);
    let idx01 = u32(y1 * V_WIDTH + x0);
    let idx11 = u32(y1 * V_WIDTH + x1);

    atomicAdd(&vWeight[idx00], w00u);
    atomicAdd(&vWeight[idx10], w10u);
    atomicAdd(&vWeight[idx01], w01u);
    atomicAdd(&vWeight[idx11], w11u);

    atomicAdd(&vSum[idx00], v00i);
    atomicAdd(&vSum[idx10], v10i);
    atomicAdd(&vSum[idx01], v01i);
    atomicAdd(&vSum[idx11], v11i);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= MAX_PARTICLES) { return; }

    let mass = particleMass[idx];
    if (mass <= 0.0) { return; }

    let pos = particlePos[idx];
    let vel = particleVel[idx];

    depositDensity(pos, mass);
    depositU(pos, vel.x, mass);
    depositV(pos, vel.y, mass);
}
`;

// =============================================================================
// NORMALIZE GRID (DENSITY + VELOCITY)
// =============================================================================

export const normalizeGridShader = `
${commonCode}

@group(0) @binding(0) var<storage, read_write> densitySum: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> uSum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> uWeight: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> vSum: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> vWeight: array<atomic<u32>>;
@group(0) @binding(5) var densityOut: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var velUPrevOut: texture_storage_2d<r32float, write>;
@group(0) @binding(7) var velVPrevOut: texture_storage_2d<r32float, write>;
@group(0) @binding(8) var velUOut: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var velVOut: texture_storage_2d<r32float, write>;

const DENSITY_SIZE: u32 = u32(WIDTH * HEIGHT);
const U_SIZE: u32 = u32(U_WIDTH * U_HEIGHT);
const V_SIZE: u32 = u32(V_WIDTH * V_HEIGHT);

@compute @workgroup_size(${GRID.WORKGROUP_SIZE}, ${GRID.WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pos = vec2i(i32(id.x), i32(id.y));

    if (pos.x < WIDTH && pos.y < HEIGHT) {
        let idx = u32(pos.y * WIDTH + pos.x);
        let mass = f32(atomicLoad(&densitySum[idx])) / FIXED_SCALE_F;
        textureStore(densityOut, pos, vec4f(mass));
    }

    if (pos.x < U_WIDTH && pos.y < U_HEIGHT) {
        let idx = u32(pos.y * U_WIDTH + pos.x);
        let w = f32(atomicLoad(&uWeight[idx]));
        var u = 0.0;
        if (w > 0.0) {
            let sum = f32(atomicLoad(&uSum[idx]));
            u = sum / w;
        }
        textureStore(velUPrevOut, pos, vec4f(u));
        textureStore(velUOut, pos, vec4f(u));
    }

    if (pos.x < V_WIDTH && pos.y < V_HEIGHT) {
        let idx = u32(pos.y * V_WIDTH + pos.x);
        let w = f32(atomicLoad(&vWeight[idx]));
        var v = 0.0;
        if (w > 0.0) {
            let sum = f32(atomicLoad(&vSum[idx]));
            v = sum / w;
        }
        textureStore(velVPrevOut, pos, vec4f(v));
        textureStore(velVOut, pos, vec4f(v));
    }
}
`;

// =============================================================================
// PRESSURE SOLVE SHADER - Jacobi iteration (uses density)
// =============================================================================

export const pressureShader = `
${commonCode}

@group(0) @binding(0) var materialTex: texture_storage_2d<r32uint, read>;
@group(0) @binding(1) var densityTex: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var velUTex: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var velVTex: texture_storage_2d<r32float, read>;
@group(0) @binding(4) var pressureIn: texture_storage_2d<r32float, read>;
@group(0) @binding(5) var pressureOut: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var<storage, read_write> cellCounts: array<atomic<u32>>;

fn getMaterial(pos: vec2i) -> u32 {
    if (!inBounds(pos)) { return WALL; }
    return textureLoad(materialTex, pos).r;
}

fn getDensity(pos: vec2i) -> f32 {
    if (!inBounds(pos)) { return 0.0; }
    return textureLoad(densityTex, pos).r;
}

fn getU(pos: vec2i) -> f32 {
    if (!inBoundsU(pos)) { return 0.0; }
    return textureLoad(velUTex, pos).r;
}

fn getV(pos: vec2i) -> f32 {
    if (!inBoundsV(pos)) { return 0.0; }
    return textureLoad(velVTex, pos).r;
}

fn getP(pos: vec2i) -> f32 {
    if (!inBounds(pos)) { return 0.0; }
    return textureLoad(pressureIn, pos).r;
}

fn getCount(pos: vec2i) -> u32 {
    if (!inBounds(pos)) { return 0u; }
    let idx = cellIndex(pos);
    return atomicLoad(&cellCounts[idx]);
}

@compute @workgroup_size(${GRID.WORKGROUP_SIZE}, ${GRID.WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pos = vec2i(i32(id.x), i32(id.y));
    if (!inBounds(pos)) { return; }

    let mat = getMaterial(pos);
    let density = getDensity(pos);
    let isFluid = getCount(pos) > 0u;

    if (isSolid(mat) || !isFluid) {
        textureStore(pressureOut, pos, vec4f(0.0));
        return;
    }

    let matL = getMaterial(pos + vec2i(-1, 0));
    let matR = getMaterial(pos + vec2i(1, 0));
    let matU = getMaterial(pos + vec2i(0, -1));
    let matD = getMaterial(pos + vec2i(0, 1));

    let uRight = select(0.0, getU(pos + vec2i(1, 0)), !isSolid(matR));
    let uLeft = select(0.0, getU(pos), !isSolid(matL));
    let vDown = select(0.0, getV(pos + vec2i(0, 1)), !isSolid(matD));
    let vUp = select(0.0, getV(pos), !isSolid(matU));
    var divergence = uRight - uLeft + vDown - vUp;
    divergence -= DRIFT_SCALE * (density - REST_DENSITY);

    let fluidL = getCount(pos + vec2i(-1, 0)) > 0u && !isSolid(matL);
    let fluidR = getCount(pos + vec2i(1, 0)) > 0u && !isSolid(matR);
    let fluidU = getCount(pos + vec2i(0, -1)) > 0u && !isSolid(matU);
    let fluidD = getCount(pos + vec2i(0, 1)) > 0u && !isSolid(matD);

    var s = 0.0;
    if (!isSolid(matL)) { s += 1.0; }
    if (!isSolid(matR)) { s += 1.0; }
    if (!isSolid(matU)) { s += 1.0; }
    if (!isSolid(matD)) { s += 1.0; }

    if (s == 0.0) {
        textureStore(pressureOut, pos, vec4f(0.0));
        return;
    }

    let pL = select(0.0, getP(pos + vec2i(-1, 0)), fluidL);
    let pR = select(0.0, getP(pos + vec2i(1, 0)), fluidR);
    let pU = select(0.0, getP(pos + vec2i(0, -1)), fluidU);
    let pD = select(0.0, getP(pos + vec2i(0, 1)), fluidD);

    let pNew = (pL + pR + pU + pD - divergence) / s;
    textureStore(pressureOut, pos, vec4f(pNew));
}
`;

// =============================================================================
// APPLY PRESSURE SHADER (MAC)
// =============================================================================

export const applyPressureShader = `
${commonCode}

@group(0) @binding(0) var materialTex: texture_storage_2d<r32uint, read>;
@group(0) @binding(1) var densityTex: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var pressureTex: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var velUIn: texture_storage_2d<r32float, read>;
@group(0) @binding(4) var velVIn: texture_storage_2d<r32float, read>;
@group(0) @binding(5) var velUOut: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var velVOut: texture_storage_2d<r32float, write>;

fn getMaterial(pos: vec2i) -> u32 {
    if (!inBounds(pos)) { return WALL; }
    return textureLoad(materialTex, pos).r;
}

fn getDensity(pos: vec2i) -> f32 {
    if (!inBounds(pos)) { return 0.0; }
    return textureLoad(densityTex, pos).r;
}

fn getP(pos: vec2i) -> f32 {
    if (!inBounds(pos)) { return 0.0; }
    return textureLoad(pressureTex, pos).r;
}

@compute @workgroup_size(${GRID.WORKGROUP_SIZE}, ${GRID.WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pos = vec2i(i32(id.x), i32(id.y));

    if (pos.x < U_WIDTH && pos.y < U_HEIGHT) {
        let left = vec2i(pos.x - 1, pos.y);
        let right = vec2i(pos.x, pos.y);
        let matL = getMaterial(left);
        let matR = getMaterial(right);
        let densL = getDensity(left);
        let densR = getDensity(right);
        var u = textureLoad(velUIn, pos).r;

        if (pos.x == 0 || pos.x == WIDTH || isSolid(matL) || isSolid(matR)) {
            u = 0.0;
        } else {
            let pL = getP(left);
            let pR = getP(right);
            u -= (pR - pL);
        }

        textureStore(velUOut, pos, vec4f(u));
    }

    if (pos.x < V_WIDTH && pos.y < V_HEIGHT) {
        let up = vec2i(pos.x, pos.y - 1);
        let down = vec2i(pos.x, pos.y);
        let matU = getMaterial(up);
        let matD = getMaterial(down);
        let densU = getDensity(up);
        let densD = getDensity(down);
        var v = textureLoad(velVIn, pos).r;

        if (pos.y == 0 || pos.y == HEIGHT || isSolid(matU) || isSolid(matD)) {
            v = 0.0;
        } else {
            let pU = getP(up);
            let pD = getP(down);
            v -= (pD - pU);
        }

        textureStore(velVOut, pos, vec4f(v));
    }
}
`;

// =============================================================================
// TRANSFER GRID VELOCITY TO PARTICLES (PIC/FLIP)
// =============================================================================

export const transferParticlesShader = `
${commonCode}

struct Params {
    dt: f32,
    gravity: f32,
    viscosityScale: f32,
    flipRatio: f32,
    frame: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var velUTex: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var velVTex: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var velUPrev: texture_storage_2d<r32float, read>;
@group(0) @binding(4) var velVPrev: texture_storage_2d<r32float, read>;
@group(0) @binding(5) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(6) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(7) var<storage, read_write> particleMass: array<f32>;

fn sampleU(tex: texture_storage_2d<r32float, read>, worldPos: vec2f) -> f32 {
    let x = clamp(worldPos.x, 0.0, f32(WIDTH));
    let y = clamp(worldPos.y - 0.5, 0.0, f32(HEIGHT - 1));

    let x0 = clamp(i32(floor(x)), 0, U_WIDTH - 1);
    let y0 = clamp(i32(floor(y)), 0, U_HEIGHT - 1);
    let x1 = clamp(x0 + 1, 0, U_WIDTH - 1);
    let y1 = clamp(y0 + 1, 0, U_HEIGHT - 1);

    let fx = x - f32(x0);
    let fy = y - f32(y0);

    let u00 = textureLoad(tex, vec2i(x0, y0)).r;
    let u10 = textureLoad(tex, vec2i(x1, y0)).r;
    let u01 = textureLoad(tex, vec2i(x0, y1)).r;
    let u11 = textureLoad(tex, vec2i(x1, y1)).r;

    let ux0 = mix(u00, u10, fx);
    let ux1 = mix(u01, u11, fx);
    return mix(ux0, ux1, fy);
}

fn sampleV(tex: texture_storage_2d<r32float, read>, worldPos: vec2f) -> f32 {
    let x = clamp(worldPos.x - 0.5, 0.0, f32(WIDTH - 1));
    let y = clamp(worldPos.y, 0.0, f32(HEIGHT));

    let x0 = clamp(i32(floor(x)), 0, V_WIDTH - 1);
    let y0 = clamp(i32(floor(y)), 0, V_HEIGHT - 1);
    let x1 = clamp(x0 + 1, 0, V_WIDTH - 1);
    let y1 = clamp(y0 + 1, 0, V_HEIGHT - 1);

    let fx = x - f32(x0);
    let fy = y - f32(y0);

    let v00 = textureLoad(tex, vec2i(x0, y0)).r;
    let v10 = textureLoad(tex, vec2i(x1, y0)).r;
    let v01 = textureLoad(tex, vec2i(x0, y1)).r;
    let v11 = textureLoad(tex, vec2i(x1, y1)).r;

    let vx0 = mix(v00, v10, fx);
    let vx1 = mix(v01, v11, fx);
    return mix(vx0, vx1, fy);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= MAX_PARTICLES) { return; }

    let mass = particleMass[idx];
    if (mass <= 0.0) { return; }

    let pos = particlePos[idx];
    let vPic = vec2f(sampleU(velUTex, pos), sampleV(velVTex, pos));
    let vPrev = vec2f(sampleU(velUPrev, pos), sampleV(velVPrev, pos));
    let vOld = particleVel[idx];

    let vFlip = vOld + (vPic - vPrev);
    let vNew = mix(vPic, vFlip, params.flipRatio);

    particleVel[idx] = vNew;
}
`;

// =============================================================================
// RENDER SHADER
// =============================================================================

export const renderShader = `
${commonCode}

@group(0) @binding(0) var materialTex: texture_storage_2d<r32uint, read>;
@group(0) @binding(1) var densityTex: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var pressureTex: texture_storage_2d<r32float, read>;
@group(0) @binding(3) var velUTex: texture_storage_2d<r32float, read>;
@group(0) @binding(4) var velVTex: texture_storage_2d<r32float, read>;
@group(0) @binding(5) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<uniform> displayMode: u32;
@group(0) @binding(7) var<storage, read_write> cellCounts: array<atomic<u32>>;

const COLORS = array<vec3f, 7>(
    vec3f(${COLORS[0].join(', ')}),
    vec3f(${COLORS[1].join(', ')}),
    vec3f(${COLORS[2].join(', ')}),
    vec3f(${COLORS[3].join(', ')}),
    vec3f(${COLORS[4].join(', ')}),
    vec3f(${COLORS[5].join(', ')}),
    vec3f(${COLORS[6].join(', ')})
);

@compute @workgroup_size(${GRID.WORKGROUP_SIZE}, ${GRID.WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pos = vec2i(i32(id.x), i32(id.y));
    if (pos.x >= WIDTH || pos.y >= HEIGHT) { return; }

    let mat = textureLoad(materialTex, pos).r;
    let density = textureLoad(densityTex, pos).r;
    let pressure = textureLoad(pressureTex, pos).r;
    let count = atomicLoad(&cellCounts[cellIndex(pos)]);

    let u = 0.5 * (textureLoad(velUTex, vec2i(pos.x, pos.y)).r +
        textureLoad(velUTex, vec2i(pos.x + 1, pos.y)).r);
    let v = 0.5 * (textureLoad(velVTex, vec2i(pos.x, pos.y)).r +
        textureLoad(velVTex, vec2i(pos.x, pos.y + 1)).r);

    var color = COLORS[EMPTY];

    if (displayMode == 1u) {
        let p = clamp((pressure + 1.0) / 2.0, 0.0, 1.0);
        color = vec3f(p, 0.2, 1.0 - p);
    } else if (displayMode == 2u) {
        let speed = sqrt(u * u + v * v);
        let s = clamp(speed / 10.0, 0.0, 1.0);
        color = vec3f(s, 1.0 - s, 0.5);
    } else {
        if (isSolid(mat)) {
            color = COLORS[mat];
        } else if (count > 0u && density >= INTERACT_THRESHOLD) {
            let level = clamp(density / RENDER_THRESHOLD, 0.0, 1.0);
            color = COLORS[WATER] * (0.5 + 0.5 * level);
        } else {
            color = COLORS[EMPTY];
        }
    }

    textureStore(outputTex, pos, vec4f(color, 1.0));
}
`;
