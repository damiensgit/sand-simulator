/**
 * FLIP Fluid Simulation - GPU Particles + Grid
 * Water uses particles internally and writes back to a grid for rendering.
 */

import { GRID, MATERIAL, SIM, DEFAULTS } from './config.js';
import {
    clearFreeListShader,
    buildFreeListShader,
    spawnParticlesShader,
    integrateParticlesShader,
    clearParticleGridShader,
    binParticlesShader,
    separateParticlesShader,
    sandStepShader,
    evictParticlesShader,
    clearGridShader,
    buildMarkerShader,
    reseedParticlesShader,
    depositParticlesShader,
    normalizeGridShader,
    maskSolidVelShader,
    vorticityShader,
    applyVorticityShader,
    ballForceShader,
    dampVelocityShader,
    pressureShader,
    applyPressureShader,
    extrapolateVelocityShader,
    transferParticlesShader,
    renderShader,
} from './shaders.js';

const WIDTH = GRID.WIDTH;
const HEIGHT = GRID.HEIGHT;
const WORKGROUP_SIZE = GRID.WORKGROUP_SIZE;
const PARTICLES_PER_CELL = SIM.PARTICLES_PER_CELL;
const MAX_CELL_PARTICLES = SIM.MAX_CELL_PARTICLES;
const MAX_PARTICLES = WIDTH * HEIGHT * PARTICLES_PER_CELL;

const MATERIAL_BY_NAME = {
    sand: MATERIAL.SAND,
    water: MATERIAL.WATER,
    lava: MATERIAL.LAVA,
    steam: MATERIAL.STEAM,
    wall: MATERIAL.WALL,
    eraser: 255,
};

const state = {
    paused: DEFAULTS.isPaused,
    gravity: SIM.GRAVITY,
    viscosityScale: 0.03,
    velocityDamping: SIM.VELOCITY_DAMPING,
    gridDamping: SIM.GRID_DAMPING,
    flipRatio: SIM.FLIP_RATIO,
    separationStrength: SIM.SEPARATION_STRENGTH,
    separationIters: SIM.SEPARATION_ITERS,
    vorticityStrength: SIM.VORTICITY_STRENGTH,
    ballEnabled: false,
    ballRadius: 12,
    simSpeed: DEFAULTS.simSpeed,
    brushSize: DEFAULTS.brushSize,
    selectedMaterial: DEFAULTS.selectedMaterial,
    selectedMaterialName: 'water',
    fluidity: 50,
    displayMode: 0,
    airMixing: SIM.AIR_MIXING,
    isDrawing: false,
    frame: 0,
    lastFpsTime: 0,
    fpsFrameCount: 0,
};

const ballState = {
    pos: { x: -100, y: -100 },
    vel: { x: 0, y: 0 },
    dragging: false,
    lastTime: 0,
    lastPos: { x: -100, y: -100 },
};

function showError(message) {
    const el = document.getElementById('error');
    el.textContent = message;
    el.style.display = 'block';
}

async function init() {
    if (!navigator.gpu) {
        showError('WebGPU not supported! Use Chrome 113+ or Edge 113+');
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        showError('Failed to get GPU adapter');
        return;
    }

    const requiredStorageTextures = 8;
    if (adapter.limits.maxStorageTexturesPerShaderStage < requiredStorageTextures) {
        showError(
            `GPU supports only ${adapter.limits.maxStorageTexturesPerShaderStage} storage textures per shader stage; need ${requiredStorageTextures}.`
        );
        return;
    }

    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageTexturesPerShaderStage: requiredStorageTextures,
        },
    });

    const canvas = document.getElementById('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = `${WIDTH * 2}px`;
    canvas.style.height = `${HEIGHT * 2}px`;

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    const textureUsage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;

    const velUWidth = WIDTH + 1;
    const velVHeight = HEIGHT + 1;

    function createTexture(format, width, height) {
        return device.createTexture({
            size: { width, height },
            format,
            usage: textureUsage,
        });
    }

    const materialTex = [createTexture('r32uint', WIDTH, HEIGHT), createTexture('r32uint', WIDTH, HEIGHT)];
    const sandMetaTex = [createTexture('r32uint', WIDTH, HEIGHT), createTexture('r32uint', WIDTH, HEIGHT)];
    const solidVelXTex = [createTexture('r32float', WIDTH, HEIGHT), createTexture('r32float', WIDTH, HEIGHT)];
    const solidVelYTex = [createTexture('r32float', WIDTH, HEIGHT), createTexture('r32float', WIDTH, HEIGHT)];
    const densityTex = createTexture('r32float', WIDTH, HEIGHT);
    const pressureTex = [createTexture('r32float', WIDTH, HEIGHT), createTexture('r32float', WIDTH, HEIGHT)];
    const velUTex = [createTexture('r32float', velUWidth, HEIGHT), createTexture('r32float', velUWidth, HEIGHT)];
    const velVTex = [createTexture('r32float', WIDTH, velVHeight), createTexture('r32float', WIDTH, velVHeight)];
    const velUPrevTex = createTexture('r32float', velUWidth, HEIGHT);
    const velVPrevTex = createTexture('r32float', WIDTH, velVHeight);
    const vorticityTex = createTexture('r32float', WIDTH, HEIGHT);

    const spawnTex = device.createTexture({
        size: { width: WIDTH, height: HEIGHT },
        format: 'r32uint',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });

    const outputTex = device.createTexture({
        size: { width: WIDTH, height: HEIGHT },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    const paramsBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const ballBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const displayModeBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const simSettingsBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const particlePosBuffer = device.createBuffer({
        size: MAX_PARTICLES * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const particleVelBuffer = device.createBuffer({
        size: MAX_PARTICLES * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const particleMassBuffer = device.createBuffer({
        size: MAX_PARTICLES * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const cellCountsBuffer = device.createBuffer({
        size: WIDTH * HEIGHT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const cellParticlesBuffer = device.createBuffer({
        size: WIDTH * HEIGHT * MAX_CELL_PARTICLES * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const freeCountBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const freeListBuffer = device.createBuffer({
        size: MAX_PARTICLES * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const densitySumBuffer = device.createBuffer({
        size: WIDTH * HEIGHT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const markerBuffer = device.createBuffer({
        size: WIDTH * HEIGHT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const uSumBuffer = device.createBuffer({
        size: velUWidth * HEIGHT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const uWeightBuffer = device.createBuffer({
        size: velUWidth * HEIGHT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const vSumBuffer = device.createBuffer({
        size: WIDTH * velVHeight * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const vWeightBuffer = device.createBuffer({
        size: WIDTH * velVHeight * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const spawnData = new Uint32Array(WIDTH * HEIGHT);
    const clearCellFloatData = new Float32Array(WIDTH * HEIGHT);
    const clearSandMetaData = new Uint32Array(WIDTH * HEIGHT);
    const velUBytesPerRow = Math.ceil(velUWidth * 4 / 256) * 256;
    const velUClearData = new Uint8Array(velUBytesPerRow * HEIGHT);
    const velVClearData = new Float32Array(WIDTH * velVHeight);
    const clearParticleData = new Uint8Array(MAX_PARTICLES * 8);
    const clearParticleMass = new Uint8Array(MAX_PARTICLES * 4);

    function buildMaterialState() {
        const material = new Uint32Array(WIDTH * HEIGHT);
        for (let y = 0; y < HEIGHT; y++) {
            for (let x = 0; x < WIDTH; x++) {
                const idx = y * WIDTH + x;
                if (x === 0 || y === 0 || x === WIDTH - 1 || y === HEIGHT - 1) {
                    material[idx] = MATERIAL.WALL;
                } else {
                    material[idx] = MATERIAL.EMPTY;
                }
            }
        }
        return material;
    }

    function seedWaterBlock() {
        const minX = Math.max(1, Math.floor(WIDTH * 0.2));
        const maxX = Math.min(WIDTH - 2, Math.floor(WIDTH * 0.5));
        const minY = Math.max(1, Math.floor(HEIGHT * 0.2));
        const maxY = Math.min(HEIGHT - 2, Math.floor(HEIGHT * 0.6));

        for (let y = minY; y <= maxY; y++) {
            for (let x = minX; x <= maxX; x++) {
                spawnData[y * WIDTH + x] = MATERIAL.WATER;
            }
        }
    }

    function clearSimulation(withWater = false) {
        const material = buildMaterialState();
        for (let i = 0; i < 2; i++) {
            device.queue.writeTexture(
                { texture: materialTex[i] },
                material,
                { bytesPerRow: WIDTH * 4 },
                { width: WIDTH, height: HEIGHT }
            );
            device.queue.writeTexture(
                { texture: sandMetaTex[i] },
                clearSandMetaData,
                { bytesPerRow: WIDTH * 4 },
                { width: WIDTH, height: HEIGHT }
            );
            device.queue.writeTexture(
                { texture: solidVelXTex[i] },
                clearCellFloatData,
                { bytesPerRow: WIDTH * 4 },
                { width: WIDTH, height: HEIGHT }
            );
            device.queue.writeTexture(
                { texture: solidVelYTex[i] },
                clearCellFloatData,
                { bytesPerRow: WIDTH * 4 },
                { width: WIDTH, height: HEIGHT }
            );
            device.queue.writeTexture(
                { texture: pressureTex[i] },
                clearCellFloatData,
                { bytesPerRow: WIDTH * 4 },
                { width: WIDTH, height: HEIGHT }
            );
            device.queue.writeTexture(
                { texture: velUTex[i] },
                velUClearData,
                { bytesPerRow: velUBytesPerRow },
                { width: velUWidth, height: HEIGHT }
            );
            device.queue.writeTexture(
                { texture: velVTex[i] },
                velVClearData,
                { bytesPerRow: WIDTH * 4 },
                { width: WIDTH, height: velVHeight }
            );
        }

        device.queue.writeTexture(
            { texture: densityTex },
            clearCellFloatData,
            { bytesPerRow: WIDTH * 4 },
            { width: WIDTH, height: HEIGHT }
        );
        device.queue.writeTexture(
            { texture: velUPrevTex },
            velUClearData,
            { bytesPerRow: velUBytesPerRow },
            { width: velUWidth, height: HEIGHT }
        );
        device.queue.writeTexture(
            { texture: velVPrevTex },
            velVClearData,
            { bytesPerRow: WIDTH * 4 },
            { width: WIDTH, height: velVHeight }
        );

        device.queue.writeBuffer(particlePosBuffer, 0, clearParticleData);
        device.queue.writeBuffer(particleVelBuffer, 0, clearParticleData);
        device.queue.writeBuffer(particleMassBuffer, 0, clearParticleMass);

        spawnData.fill(0);
        if (withWater) {
            seedWaterBlock();
        }
    }

    clearSimulation(true);

    const clearFreeListPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: clearFreeListShader }), entryPoint: 'main' },
    });
    const buildFreeListPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: buildFreeListShader }), entryPoint: 'main' },
    });
    const spawnPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: spawnParticlesShader }), entryPoint: 'main' },
    });
    const integratePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: integrateParticlesShader }), entryPoint: 'main' },
    });
    const clearParticleGridPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: clearParticleGridShader }), entryPoint: 'main' },
    });
    const binParticlesPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: binParticlesShader }), entryPoint: 'main' },
    });
    const separateParticlesPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: separateParticlesShader }), entryPoint: 'main' },
    });
    const sandStepPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: sandStepShader }), entryPoint: 'main' },
    });
    const evictParticlesPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: evictParticlesShader }), entryPoint: 'main' },
    });
    const clearGridPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: clearGridShader }), entryPoint: 'main' },
    });
    const buildMarkerPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: buildMarkerShader }), entryPoint: 'main' },
    });
    const reseedPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: reseedParticlesShader }), entryPoint: 'main' },
    });
    const depositPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: depositParticlesShader }), entryPoint: 'main' },
    });
    const normalizePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: normalizeGridShader }), entryPoint: 'main' },
    });
    const maskSolidVelPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: maskSolidVelShader }), entryPoint: 'main' },
    });
    const vorticityPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: vorticityShader }), entryPoint: 'main' },
    });
    const applyVorticityPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: applyVorticityShader }), entryPoint: 'main' },
    });
    const ballForcePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: ballForceShader }), entryPoint: 'main' },
    });
    const dampVelocityPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: dampVelocityShader }), entryPoint: 'main' },
    });
    const pressurePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: pressureShader }), entryPoint: 'main' },
    });
    const applyPressurePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: applyPressureShader }), entryPoint: 'main' },
    });
    const extrapolatePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: extrapolateVelocityShader }), entryPoint: 'main' },
    });
    const transferPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: transferParticlesShader }), entryPoint: 'main' },
    });
    const renderPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: renderShader }), entryPoint: 'main' },
    });

    const spawnGroups = [
        device.createBindGroup({
            layout: spawnPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: spawnTex.createView() },
                { binding: 1, resource: materialTex[0].createView() },
                { binding: 2, resource: materialTex[1].createView() },
                { binding: 3, resource: { buffer: particlePosBuffer } },
                { binding: 4, resource: { buffer: particleVelBuffer } },
                { binding: 5, resource: { buffer: particleMassBuffer } },
                { binding: 6, resource: { buffer: cellCountsBuffer } },
                { binding: 7, resource: { buffer: cellParticlesBuffer } },
                { binding: 8, resource: { buffer: freeCountBuffer } },
                { binding: 9, resource: { buffer: freeListBuffer } },
                { binding: 10, resource: sandMetaTex[0].createView() },
                { binding: 11, resource: sandMetaTex[1].createView() },
            ],
        }),
        device.createBindGroup({
            layout: spawnPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: spawnTex.createView() },
                { binding: 1, resource: materialTex[1].createView() },
                { binding: 2, resource: materialTex[0].createView() },
                { binding: 3, resource: { buffer: particlePosBuffer } },
                { binding: 4, resource: { buffer: particleVelBuffer } },
                { binding: 5, resource: { buffer: particleMassBuffer } },
                { binding: 6, resource: { buffer: cellCountsBuffer } },
                { binding: 7, resource: { buffer: cellParticlesBuffer } },
                { binding: 8, resource: { buffer: freeCountBuffer } },
                { binding: 9, resource: { buffer: freeListBuffer } },
                { binding: 10, resource: sandMetaTex[1].createView() },
                { binding: 11, resource: sandMetaTex[0].createView() },
            ],
        }),
    ];

    const integrateGroups = [
        device.createBindGroup({
            layout: integratePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: materialTex[0].createView() },
                { binding: 2, resource: { buffer: particlePosBuffer } },
                { binding: 3, resource: { buffer: particleVelBuffer } },
                { binding: 4, resource: { buffer: particleMassBuffer } },
                { binding: 5, resource: densityTex.createView() },
            ],
        }),
        device.createBindGroup({
            layout: integratePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: materialTex[1].createView() },
                { binding: 2, resource: { buffer: particlePosBuffer } },
                { binding: 3, resource: { buffer: particleVelBuffer } },
                { binding: 4, resource: { buffer: particleMassBuffer } },
                { binding: 5, resource: densityTex.createView() },
            ],
        }),
    ];

    const clearParticleGridGroup = device.createBindGroup({
        layout: clearParticleGridPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellCountsBuffer } },
        ],
    });

    const clearFreeListGroup = device.createBindGroup({
        layout: clearFreeListPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: freeCountBuffer } },
        ],
    });

    const buildFreeListGroup = device.createBindGroup({
        layout: buildFreeListPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: freeCountBuffer } },
            { binding: 1, resource: { buffer: freeListBuffer } },
            { binding: 2, resource: { buffer: particleMassBuffer } },
        ],
    });

    const binParticlesGroup = device.createBindGroup({
        layout: binParticlesPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellCountsBuffer } },
            { binding: 1, resource: { buffer: cellParticlesBuffer } },
            { binding: 2, resource: { buffer: particlePosBuffer } },
            { binding: 3, resource: { buffer: particleMassBuffer } },
        ],
    });

    const separateGroups = [
        device.createBindGroup({
            layout: separateParticlesPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: materialTex[0].createView() },
                { binding: 1, resource: { buffer: cellCountsBuffer } },
                { binding: 2, resource: { buffer: cellParticlesBuffer } },
                { binding: 3, resource: { buffer: particlePosBuffer } },
                { binding: 4, resource: { buffer: particleMassBuffer } },
                { binding: 5, resource: { buffer: paramsBuffer } },
            ],
        }),
        device.createBindGroup({
            layout: separateParticlesPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: materialTex[1].createView() },
                { binding: 1, resource: { buffer: cellCountsBuffer } },
                { binding: 2, resource: { buffer: cellParticlesBuffer } },
                { binding: 3, resource: { buffer: particlePosBuffer } },
                { binding: 4, resource: { buffer: particleMassBuffer } },
                { binding: 5, resource: { buffer: paramsBuffer } },
            ],
        }),
    ];

    const sandGroups = Array.from({ length: 2 }, () => Array(2));
    const evictGroups = [
        device.createBindGroup({
            layout: evictParticlesPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: materialTex[0].createView() },
                { binding: 1, resource: { buffer: particlePosBuffer } },
                { binding: 2, resource: { buffer: particleVelBuffer } },
                { binding: 3, resource: { buffer: particleMassBuffer } },
                { binding: 4, resource: solidVelXTex[0].createView() },
                { binding: 5, resource: solidVelYTex[0].createView() },
            ],
        }),
        device.createBindGroup({
            layout: evictParticlesPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: materialTex[1].createView() },
                { binding: 1, resource: { buffer: particlePosBuffer } },
                { binding: 2, resource: { buffer: particleVelBuffer } },
                { binding: 3, resource: { buffer: particleMassBuffer } },
                { binding: 4, resource: solidVelXTex[1].createView() },
                { binding: 5, resource: solidVelYTex[1].createView() },
            ],
        }),
    ];

    const clearGridGroup = device.createBindGroup({
        layout: clearGridPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: densitySumBuffer } },
            { binding: 1, resource: { buffer: uSumBuffer } },
            { binding: 2, resource: { buffer: uWeightBuffer } },
            { binding: 3, resource: { buffer: vSumBuffer } },
            { binding: 4, resource: { buffer: vWeightBuffer } },
        ],
    });

    const depositGroup = device.createBindGroup({
        layout: depositPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: densitySumBuffer } },
            { binding: 1, resource: { buffer: uSumBuffer } },
            { binding: 2, resource: { buffer: uWeightBuffer } },
            { binding: 3, resource: { buffer: vSumBuffer } },
            { binding: 4, resource: { buffer: vWeightBuffer } },
            { binding: 5, resource: { buffer: particlePosBuffer } },
            { binding: 6, resource: { buffer: particleVelBuffer } },
            { binding: 7, resource: { buffer: particleMassBuffer } },
        ],
    });

    const normalizeGroup = device.createBindGroup({
        layout: normalizePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: densitySumBuffer } },
            { binding: 1, resource: { buffer: uSumBuffer } },
            { binding: 2, resource: { buffer: uWeightBuffer } },
            { binding: 3, resource: { buffer: vSumBuffer } },
            { binding: 4, resource: { buffer: vWeightBuffer } },
            { binding: 5, resource: densityTex.createView() },
            { binding: 6, resource: velUPrevTex.createView() },
            { binding: 7, resource: velVPrevTex.createView() },
            { binding: 8, resource: velUTex[0].createView() },
            { binding: 9, resource: velVTex[0].createView() },
        ],
    });

    const markerGroups = [
        device.createBindGroup({
            layout: buildMarkerPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: materialTex[0].createView() },
                { binding: 1, resource: densityTex.createView() },
                { binding: 2, resource: { buffer: cellCountsBuffer } },
                { binding: 3, resource: { buffer: markerBuffer } },
            ],
        }),
        device.createBindGroup({
            layout: buildMarkerPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: materialTex[1].createView() },
                { binding: 1, resource: densityTex.createView() },
                { binding: 2, resource: { buffer: cellCountsBuffer } },
                { binding: 3, resource: { buffer: markerBuffer } },
            ],
        }),
    ];

    const reseedGroup = device.createBindGroup({
        layout: reseedPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellCountsBuffer } },
            { binding: 1, resource: { buffer: cellParticlesBuffer } },
            { binding: 2, resource: { buffer: particlePosBuffer } },
            { binding: 3, resource: { buffer: particleVelBuffer } },
            { binding: 4, resource: velUTex[0].createView() },
            { binding: 5, resource: velVTex[0].createView() },
            { binding: 6, resource: { buffer: markerBuffer } },
        ],
    });

    const maskVelGroups = Array.from({ length: 2 }, () => Array(2));
    for (let mi = 0; mi < 2; mi++) {
        for (let vi = 0; vi < 2; vi++) {
            maskVelGroups[mi][vi] = device.createBindGroup({
                layout: maskSolidVelPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: materialTex[mi].createView() },
                    { binding: 1, resource: solidVelXTex[mi].createView() },
                    { binding: 2, resource: solidVelYTex[mi].createView() },
                    { binding: 3, resource: velUTex[vi].createView() },
                    { binding: 4, resource: velVTex[vi].createView() },
                ],
            });
        }
    }

    const vorticityGroups = Array.from({ length: 2 }, (_, vi) => (
        device.createBindGroup({
            layout: vorticityPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: velUTex[vi].createView() },
                { binding: 1, resource: velVTex[vi].createView() },
                { binding: 2, resource: vorticityTex.createView() },
            ],
        })
    ));

    const applyVorticityGroups = Array.from({ length: 2 }, (_, vi) => (
        device.createBindGroup({
            layout: applyVorticityPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: vorticityTex.createView() },
                { binding: 2, resource: velUTex[vi].createView() },
                { binding: 3, resource: velVTex[vi].createView() },
            ],
        })
    ));

    const ballForceGroups = Array.from({ length: 2 }, (_, vi) => (
        device.createBindGroup({
            layout: ballForcePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: ballBuffer } },
                { binding: 1, resource: velUTex[vi].createView() },
                { binding: 2, resource: velVTex[vi].createView() },
            ],
        })
    ));

    const dampVelGroups = Array.from({ length: 2 }, (_, vi) => (
        device.createBindGroup({
            layout: dampVelocityPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: velUTex[vi].createView() },
                { binding: 2, resource: velVTex[vi].createView() },
            ],
        })
    ));

    const pressureGroups = Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => Array(2)));
    const applyGroups = Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => Array(2)));
    const extrapolateGroups = Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => Array(2)));
    const renderGroups = Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => Array(2)));
    const transferGroups = Array.from({ length: 2 }, () => Array(2));

    for (let mi = 0; mi < 2; mi++) {
        for (let vi = 0; vi < 2; vi++) {
            transferGroups[mi][vi] = device.createBindGroup({
                layout: transferPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: paramsBuffer } },
                    { binding: 1, resource: velUTex[vi].createView() },
                    { binding: 2, resource: velVTex[vi].createView() },
                    { binding: 3, resource: velUPrevTex.createView() },
                    { binding: 4, resource: velVPrevTex.createView() },
                    { binding: 5, resource: materialTex[mi].createView() },
                    { binding: 6, resource: densityTex.createView() },
                    { binding: 7, resource: { buffer: particlePosBuffer } },
                    { binding: 8, resource: { buffer: particleVelBuffer } },
                    { binding: 9, resource: { buffer: particleMassBuffer } },
                ],
            });

            sandGroups[mi][vi] = device.createBindGroup({
                layout: sandStepPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: paramsBuffer } },
                    { binding: 1, resource: materialTex[mi].createView() },
                    { binding: 2, resource: materialTex[1 - mi].createView() },
                    { binding: 3, resource: densityTex.createView() },
                    { binding: 4, resource: velUTex[vi].createView() },
                    { binding: 5, resource: sandMetaTex[mi].createView() },
                    { binding: 6, resource: sandMetaTex[1 - mi].createView() },
                    { binding: 7, resource: solidVelXTex[1 - mi].createView() },
                    { binding: 8, resource: solidVelYTex[1 - mi].createView() },
                ],
            });

            for (let wi = 0; wi < 2; wi++) {
                extrapolateGroups[mi][vi][wi] = device.createBindGroup({
                    layout: extrapolatePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: velUTex[vi].createView() },
                        { binding: 1, resource: velVTex[vi].createView() },
                        { binding: 2, resource: velUTex[wi].createView() },
                        { binding: 3, resource: velVTex[wi].createView() },
                        { binding: 4, resource: { buffer: markerBuffer } },
                        { binding: 5, resource: { buffer: simSettingsBuffer } },
                    ],
                });
            }

            for (let pi = 0; pi < 2; pi++) {
                pressureGroups[mi][vi][pi] = device.createBindGroup({
                    layout: pressurePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: materialTex[mi].createView() },
                        { binding: 1, resource: densityTex.createView() },
                        { binding: 2, resource: velUTex[vi].createView() },
                        { binding: 3, resource: velVTex[vi].createView() },
                        { binding: 4, resource: pressureTex[pi].createView() },
                        { binding: 5, resource: pressureTex[1 - pi].createView() },
                        { binding: 6, resource: { buffer: markerBuffer } },
                        { binding: 7, resource: { buffer: simSettingsBuffer } },
                        { binding: 8, resource: solidVelXTex[mi].createView() },
                        { binding: 9, resource: solidVelYTex[mi].createView() },
                    ],
                });

                applyGroups[mi][vi][pi] = device.createBindGroup({
                    layout: applyPressurePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: materialTex[mi].createView() },
                        { binding: 1, resource: pressureTex[pi].createView() },
                        { binding: 2, resource: velUTex[vi].createView() },
                        { binding: 3, resource: velVTex[vi].createView() },
                        { binding: 4, resource: velUTex[1 - vi].createView() },
                        { binding: 5, resource: velVTex[1 - vi].createView() },
                    ],
                });

                renderGroups[mi][pi][vi] = device.createBindGroup({
                    layout: renderPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: materialTex[mi].createView() },
                        { binding: 1, resource: densityTex.createView() },
                        { binding: 2, resource: pressureTex[pi].createView() },
                        { binding: 3, resource: velUTex[vi].createView() },
                        { binding: 4, resource: velVTex[vi].createView() },
                        { binding: 5, resource: outputTex.createView() },
                        { binding: 6, resource: { buffer: displayModeBuffer } },
                        { binding: 7, resource: { buffer: cellCountsBuffer } },
                        { binding: 8, resource: { buffer: simSettingsBuffer } },
                        { binding: 9, resource: sandMetaTex[mi].createView() },
                        { binding: 10, resource: { buffer: ballBuffer } },
                    ],
                });
            }
        }
    }

    const fullscreenShader = `
        @group(0) @binding(0) var gridTex: texture_2d<f32>;
        @group(0) @binding(1) var gridSampler: sampler;

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) uv: vec2f,
        }

        @vertex
        fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
            var pos = array<vec2f, 4>(
                vec2f(-1.0, -1.0),
                vec2f(1.0, -1.0),
                vec2f(-1.0, 1.0),
                vec2f(1.0, 1.0)
            );
            var uvs = array<vec2f, 4>(
                vec2f(0.0, 1.0),
                vec2f(1.0, 1.0),
                vec2f(0.0, 0.0),
                vec2f(1.0, 0.0)
            );
            var out: VertexOutput;
            out.position = vec4f(pos[idx], 0.0, 1.0);
            out.uv = uvs[idx];
            return out;
        }

        @fragment
        fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            return textureSample(gridTex, gridSampler, uv);
        }
    `;

    const fullscreenModule = device.createShaderModule({ code: fullscreenShader });
    const sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    const screenPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: fullscreenModule, entryPoint: 'vs' },
        fragment: { module: fullscreenModule, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-strip' },
    });

    const screenBindGroup = device.createBindGroup({
        layout: screenPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: outputTex.createView() },
            { binding: 1, resource: sampler },
        ],
    });

    const paramsData = new ArrayBuffer(64);
    const paramsView = new DataView(paramsData);
    const ballData = new ArrayBuffer(32);
    const ballView = new DataView(ballData);

    function updateParams() {
        paramsView.setFloat32(0, SIM.DT, true);
        paramsView.setFloat32(4, state.gravity, true);
        paramsView.setFloat32(8, state.viscosityScale, true);
        paramsView.setFloat32(12, state.flipRatio, true);
        paramsView.setFloat32(16, state.velocityDamping, true);
        paramsView.setFloat32(20, state.gridDamping, true);
        paramsView.setFloat32(24, state.fluidity / 100, true);
        paramsView.setFloat32(28, SIM.SOLID_DAMPING, true);
        paramsView.setFloat32(32, state.separationStrength, true);
        paramsView.setFloat32(36, state.vorticityStrength, true);
        paramsView.setUint32(40, state.frame, true);
        paramsView.setUint32(44, 0, true);
        paramsView.setUint32(48, 0, true);
        paramsView.setUint32(52, 0, true);
        paramsView.setUint32(56, 0, true);
        paramsView.setUint32(60, 0, true);
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    }

    function updateBallBuffer() {
        ballView.setFloat32(0, ballState.pos.x, true);
        ballView.setFloat32(4, ballState.pos.y, true);
        ballView.setFloat32(8, state.ballRadius, true);
        ballView.setUint32(12, state.ballEnabled ? 1 : 0, true);
        ballView.setFloat32(16, ballState.vel.x, true);
        ballView.setFloat32(20, ballState.vel.y, true);
        ballView.setFloat32(24, 0, true);
        ballView.setFloat32(28, 0, true);
        device.queue.writeBuffer(ballBuffer, 0, ballData);
    }

    function updateDisplayMode() {
        const value = new Uint32Array([state.displayMode]);
        device.queue.writeBuffer(displayModeBuffer, 0, value);
    }

    function updateSimSettings() {
        const value = new Uint32Array([state.airMixing ? 1 : 0, 0, 0, 0]);
        device.queue.writeBuffer(simSettingsBuffer, 0, value);
    }

    updateDisplayMode();
    updateSimSettings();

    function updateStatsUnavailable() {
        document.getElementById('sandCount').textContent = '-';
        document.getElementById('waterCount').textContent = '-';
        document.getElementById('lavaCount').textContent = '-';
        document.getElementById('steamCount').textContent = '-';
    }

    updateStatsUnavailable();

    function resetBallInteraction() {
        ballState.dragging = false;
        ballState.vel = { x: 0, y: 0 };
        ballState.lastTime = 0;
    }

    function updateMaterialButtons() {
        document.querySelectorAll('.material-btn').forEach(btn => {
            if (btn.dataset.material) {
                const selected = !state.ballEnabled && btn.dataset.material === state.selectedMaterialName;
                btn.classList.toggle('selected', selected);
                return;
            }
            if (btn.dataset.tool === 'ball') {
                btn.classList.toggle('selected', state.ballEnabled);
            }
        });
    }

    function setSelectedMaterial(name) {
        const value = MATERIAL_BY_NAME[name] ?? MATERIAL.WATER;
        state.selectedMaterial = value;
        state.selectedMaterialName = name;
        if (state.ballEnabled) {
            state.ballEnabled = false;
            resetBallInteraction();
            updateBallBuffer();
        }
        updateMaterialButtons();
    }

    function toggleBallTool() {
        state.ballEnabled = !state.ballEnabled;
        if (!state.ballEnabled) {
            resetBallInteraction();
        }
        updateBallBuffer();
        updateMaterialButtons();
    }

    const DISPLAY_MODES = {
        none: 0,
        pressure: 1,
        velocity: 2,
        marching: 3,
    };

    function updateDisplayButtons() {
        document.querySelectorAll('.option-btn').forEach(btn => {
            const mode = DISPLAY_MODES[btn.dataset.display] ?? 0;
            btn.classList.toggle('selected', mode === state.displayMode);
        });
    }

    function setDisplayModeFromName(name) {
        state.displayMode = DISPLAY_MODES[name] ?? 0;
        updateDisplayMode();
        updateDisplayButtons();
    }

    function applyBrush(x, y) {
        const radius = Math.max(1, state.brushSize);
        const r2 = radius * radius;
        const mat = state.selectedMaterial;

        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                if (dx * dx + dy * dy > r2) continue;
                const px = x + dx;
                const py = y + dy;
                if (px <= 0 || py <= 0 || px >= WIDTH - 1 || py >= HEIGHT - 1) continue;
                spawnData[py * WIDTH + px] = mat;
            }
        }
    }

    function getCellFromEvent(e) {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor(((e.clientX - rect.left) / rect.width) * WIDTH);
        const y = Math.floor(((e.clientY - rect.top) / rect.height) * HEIGHT);
        return { x, y };
    }

    function getPosFromEvent(e) {
        const rect = canvas.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * WIDTH;
        const y = ((e.clientY - rect.top) / rect.height) * HEIGHT;
        return { x, y };
    }

    function updateBallFromEvent(e) {
        const pos = getPosFromEvent(e);
        const now = performance.now();
        if (ballState.lastTime > 0) {
            const dt = (now - ballState.lastTime) / 1000;
            if (dt > 0) {
                ballState.vel = {
                    x: (pos.x - ballState.lastPos.x) / dt,
                    y: (pos.y - ballState.lastPos.y) / dt,
                };
            }
        } else {
            ballState.vel = { x: 0, y: 0 };
        }
        ballState.pos = pos;
        ballState.lastPos = pos;
        ballState.lastTime = now;
        updateBallBuffer();
    }

    document.querySelectorAll('.material-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.dataset.material) {
                setSelectedMaterial(btn.dataset.material);
            } else if (btn.dataset.tool === 'ball') {
                toggleBallTool();
            }
        });
    });

    document.querySelectorAll('.option-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setDisplayModeFromName(btn.dataset.display);
        });
    });

    setSelectedMaterial('water');

    document.getElementById('gravity').value = state.gravity;
    document.getElementById('gravityValue').textContent = state.gravity.toFixed(1);
    document.getElementById('viscosity').value = state.viscosityScale;
    document.getElementById('viscosityValue').textContent = state.viscosityScale.toFixed(1);
    document.getElementById('damping').value = state.velocityDamping;
    document.getElementById('dampingValue').textContent = state.velocityDamping.toFixed(2);
    document.getElementById('gridDamping').value = state.gridDamping;
    document.getElementById('gridDampingValue').textContent = state.gridDamping.toFixed(2);
    document.getElementById('flipRatio').value = state.flipRatio;
    document.getElementById('flipValue').textContent = state.flipRatio.toFixed(2);
    document.getElementById('vorticity').value = state.vorticityStrength;
    document.getElementById('vorticityValue').textContent = state.vorticityStrength.toFixed(2);
    document.getElementById('separationStrength').value = state.separationStrength;
    document.getElementById('separationStrengthValue').textContent = state.separationStrength.toFixed(2);
    document.getElementById('separationIters').value = state.separationIters;
    document.getElementById('separationItersValue').textContent = state.separationIters;
    document.getElementById('ballRadius').value = state.ballRadius;
    document.getElementById('ballRadiusValue').textContent = state.ballRadius;
    document.getElementById('airMixing').checked = state.airMixing;
    updateMaterialButtons();
    updateDisplayButtons();

    document.getElementById('brushSize').addEventListener('input', (e) => {
        state.brushSize = parseInt(e.target.value, 10);
        document.getElementById('brushValue').textContent = state.brushSize;
    });

    document.getElementById('simSpeed').value = state.simSpeed;
    document.getElementById('speedValue').textContent = state.simSpeed;

    document.getElementById('simSpeed').addEventListener('input', (e) => {
        state.simSpeed = parseInt(e.target.value, 10);
        document.getElementById('speedValue').textContent = state.simSpeed;
    });

    document.getElementById('gravity').addEventListener('input', (e) => {
        state.gravity = parseFloat(e.target.value);
        document.getElementById('gravityValue').textContent = state.gravity.toFixed(1);
    });

    document.getElementById('viscosity').addEventListener('input', (e) => {
        state.viscosityScale = parseFloat(e.target.value);
        document.getElementById('viscosityValue').textContent = state.viscosityScale.toFixed(1);
    });

    document.getElementById('damping').addEventListener('input', (e) => {
        state.velocityDamping = parseFloat(e.target.value);
        document.getElementById('dampingValue').textContent = state.velocityDamping.toFixed(2);
    });

    document.getElementById('gridDamping').addEventListener('input', (e) => {
        state.gridDamping = parseFloat(e.target.value);
        document.getElementById('gridDampingValue').textContent = state.gridDamping.toFixed(2);
    });

    document.getElementById('flipRatio').addEventListener('input', (e) => {
        state.flipRatio = parseFloat(e.target.value);
        document.getElementById('flipValue').textContent = state.flipRatio.toFixed(2);
    });

    document.getElementById('vorticity').addEventListener('input', (e) => {
        state.vorticityStrength = parseFloat(e.target.value);
        document.getElementById('vorticityValue').textContent = state.vorticityStrength.toFixed(2);
    });

    document.getElementById('separationStrength').addEventListener('input', (e) => {
        state.separationStrength = parseFloat(e.target.value);
        document.getElementById('separationStrengthValue').textContent = state.separationStrength.toFixed(2);
    });

    document.getElementById('separationIters').addEventListener('input', (e) => {
        state.separationIters = parseInt(e.target.value, 10);
        document.getElementById('separationItersValue').textContent = state.separationIters;
    });

    document.getElementById('ballRadius').addEventListener('input', (e) => {
        state.ballRadius = parseFloat(e.target.value);
        document.getElementById('ballRadiusValue').textContent = state.ballRadius.toFixed(0);
    });

    document.getElementById('fluidity').addEventListener('input', (e) => {
        state.fluidity = parseInt(e.target.value, 10);
        document.getElementById('fluidityValue').textContent = state.fluidity;
    });

    setDisplayModeFromName('none');

    document.getElementById('airMixing').addEventListener('change', (e) => {
        state.airMixing = e.target.checked;
        updateSimSettings();
    });

    document.getElementById('pauseBtn').addEventListener('click', () => {
        state.paused = !state.paused;
        document.getElementById('pauseBtn').textContent = state.paused ? 'Resume' : 'Pause';
    });

    document.getElementById('clearBtn').addEventListener('click', () => {
        clearSimulation(false);
    });

    function endPointerInteraction() {
        state.isDrawing = false;
        resetBallInteraction();
        updateBallBuffer();
    }

    canvas.addEventListener('pointerdown', (e) => {
        canvas.setPointerCapture(e.pointerId);
        if (state.ballEnabled) {
            ballState.dragging = true;
            updateBallFromEvent(e);
            return;
        }
        state.isDrawing = true;
        const pos = getCellFromEvent(e);
        applyBrush(pos.x, pos.y);
    });

    canvas.addEventListener('pointermove', (e) => {
        if (state.ballEnabled) {
            if (ballState.dragging) {
                updateBallFromEvent(e);
            }
            return;
        }
        if (!state.isDrawing) return;
        const pos = getCellFromEvent(e);
        applyBrush(pos.x, pos.y);
    });

    canvas.addEventListener('pointerup', (e) => {
        if (canvas.hasPointerCapture(e.pointerId)) {
            canvas.releasePointerCapture(e.pointerId);
        }
        endPointerInteraction();
    });
    canvas.addEventListener('pointercancel', endPointerInteraction);

    document.addEventListener('keydown', (e) => {
        if (e.key === ' ') {
            e.preventDefault();
            document.getElementById('pauseBtn').click();
        }
        if (e.key === 'c' || e.key === 'C') {
            clearSimulation(false);
        }
        if (e.key === '[') {
            state.brushSize = Math.max(1, state.brushSize - 1);
            document.getElementById('brushSize').value = state.brushSize;
            document.getElementById('brushValue').textContent = state.brushSize;
        }
        if (e.key === ']') {
            state.brushSize = Math.min(20, state.brushSize + 1);
            document.getElementById('brushSize').value = state.brushSize;
            document.getElementById('brushValue').textContent = state.brushSize;
        }
        if (e.key >= '1' && e.key <= '6') {
            const names = ['sand', 'water', 'lava', 'steam', 'wall', 'eraser'];
            setSelectedMaterial(names[parseInt(e.key, 10) - 1]);
        }
    });

    let matIndex = 0;
    let velIndex = 0;
    let pressureIndex = 0;

    const cellWorkgroupsX = Math.ceil(WIDTH / WORKGROUP_SIZE);
    const cellWorkgroupsY = Math.ceil(HEIGHT / WORKGROUP_SIZE);
    const velWorkgroupsX = Math.ceil((WIDTH + 1) / WORKGROUP_SIZE);
    const velWorkgroupsY = Math.ceil((HEIGHT + 1) / WORKGROUP_SIZE);
    const particleWorkgroups = Math.ceil(MAX_PARTICLES / 256);
    const clearCount = Math.max(WIDTH * HEIGHT, velUWidth * HEIGHT, WIDTH * velVHeight);
    const clearWorkgroups = Math.ceil(clearCount / 256);
    const cellCountWorkgroups = Math.ceil((WIDTH * HEIGHT) / 256);
    const extrapolationIters = Math.max(0, SIM.EXTRAPOLATION_ITERS | 0);
    const reseedIters = Math.max(0, SIM.RESEED_ITERS | 0);

    function frame() {
        const now = performance.now();
        state.fpsFrameCount++;
        if (now - state.lastFpsTime >= 1000) {
            document.getElementById('fps').textContent = state.fpsFrameCount;
            state.fpsFrameCount = 0;
            state.lastFpsTime = now;
        }

        if (ballState.dragging && now - ballState.lastTime > 100) {
            ballState.vel = { x: 0, y: 0 };
        }

        device.queue.writeTexture(
            { texture: spawnTex },
            spawnData,
            { bytesPerRow: WIDTH * 4 },
            { width: WIDTH, height: HEIGHT }
        );
        spawnData.fill(0);
        updateBallBuffer();

        const encoder = device.createCommandEncoder();
        const steps = Math.max(1, state.simSpeed | 0);

        {
            const pass = encoder.beginComputePass();

            if (!state.paused) {
                pass.setPipeline(sandStepPipeline);
                pass.setBindGroup(0, sandGroups[matIndex][velIndex]);
                pass.dispatchWorkgroups(cellWorkgroupsX, cellWorkgroupsY);
                matIndex = 1 - matIndex;

                pass.setPipeline(evictParticlesPipeline);
                pass.setBindGroup(0, evictGroups[matIndex]);
                pass.dispatchWorkgroups(particleWorkgroups);
            }

            pass.setPipeline(clearParticleGridPipeline);
            pass.setBindGroup(0, clearParticleGridGroup);
            pass.dispatchWorkgroups(cellCountWorkgroups);

            pass.setPipeline(binParticlesPipeline);
            pass.setBindGroup(0, binParticlesGroup);
            pass.dispatchWorkgroups(particleWorkgroups);

            pass.setPipeline(clearFreeListPipeline);
            pass.setBindGroup(0, clearFreeListGroup);
            pass.dispatchWorkgroups(1);

            pass.setPipeline(buildFreeListPipeline);
            pass.setBindGroup(0, buildFreeListGroup);
            pass.dispatchWorkgroups(particleWorkgroups);

            pass.setPipeline(spawnPipeline);
            pass.setBindGroup(0, spawnGroups[matIndex]);
            pass.dispatchWorkgroups(cellWorkgroupsX, cellWorkgroupsY);
            matIndex = 1 - matIndex;

            if (!state.paused) {
                for (let step = 0; step < steps; step++) {
                    state.frame++;
                    updateParams();
                    const separationIters = Math.max(0, state.separationIters | 0);

                    pass.setPipeline(integratePipeline);
                    pass.setBindGroup(0, integrateGroups[matIndex]);
                    pass.dispatchWorkgroups(particleWorkgroups);

                    for (let iter = 0; iter < separationIters; iter++) {
                        pass.setPipeline(clearParticleGridPipeline);
                        pass.setBindGroup(0, clearParticleGridGroup);
                        pass.dispatchWorkgroups(cellCountWorkgroups);

                        pass.setPipeline(binParticlesPipeline);
                        pass.setBindGroup(0, binParticlesGroup);
                        pass.dispatchWorkgroups(particleWorkgroups);

                        pass.setPipeline(separateParticlesPipeline);
                        pass.setBindGroup(0, separateGroups[matIndex]);
                        pass.dispatchWorkgroups(particleWorkgroups);
                    }

                    pass.setPipeline(clearParticleGridPipeline);
                    pass.setBindGroup(0, clearParticleGridGroup);
                    pass.dispatchWorkgroups(cellCountWorkgroups);

                    pass.setPipeline(binParticlesPipeline);
                    pass.setBindGroup(0, binParticlesGroup);
                    pass.dispatchWorkgroups(particleWorkgroups);

                    pass.setPipeline(clearGridPipeline);
                    pass.setBindGroup(0, clearGridGroup);
                    pass.dispatchWorkgroups(clearWorkgroups);

                    pass.setPipeline(depositPipeline);
                    pass.setBindGroup(0, depositGroup);
                    pass.dispatchWorkgroups(particleWorkgroups);

                    pass.setPipeline(normalizePipeline);
                    pass.setBindGroup(0, normalizeGroup);
                    pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);

                    pass.setPipeline(buildMarkerPipeline);
                    pass.setBindGroup(0, markerGroups[matIndex]);
                    pass.dispatchWorkgroups(cellWorkgroupsX, cellWorkgroupsY);

                    for (let iter = 0; iter < reseedIters; iter++) {
                        pass.setPipeline(reseedPipeline);
                        pass.setBindGroup(0, reseedGroup);
                        pass.dispatchWorkgroups(cellWorkgroupsX, cellWorkgroupsY);
                    }

                    if (reseedIters > 0) {
                        pass.setPipeline(clearGridPipeline);
                        pass.setBindGroup(0, clearGridGroup);
                        pass.dispatchWorkgroups(clearWorkgroups);

                        pass.setPipeline(depositPipeline);
                        pass.setBindGroup(0, depositGroup);
                        pass.dispatchWorkgroups(particleWorkgroups);

                        pass.setPipeline(normalizePipeline);
                        pass.setBindGroup(0, normalizeGroup);
                        pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);

                        pass.setPipeline(buildMarkerPipeline);
                        pass.setBindGroup(0, markerGroups[matIndex]);
                        pass.dispatchWorkgroups(cellWorkgroupsX, cellWorkgroupsY);
                    }

                    velIndex = 0;
                    pass.setPipeline(maskSolidVelPipeline);
                    pass.setBindGroup(0, maskVelGroups[matIndex][velIndex]);
                    pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);

                    pressureIndex = 0;
                    for (let iter = 0; iter < SIM.PRESSURE_ITERATIONS; iter++) {
                        pass.setPipeline(pressurePipeline);
                        pass.setBindGroup(0, pressureGroups[matIndex][velIndex][pressureIndex]);
                        pass.dispatchWorkgroups(cellWorkgroupsX, cellWorkgroupsY);
                        pressureIndex = 1 - pressureIndex;
                    }

                    pass.setPipeline(applyPressurePipeline);
                    pass.setBindGroup(0, applyGroups[matIndex][velIndex][pressureIndex]);
                    pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);
                    velIndex = 1;

                    pass.setPipeline(maskSolidVelPipeline);
                    pass.setBindGroup(0, maskVelGroups[matIndex][velIndex]);
                    pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);

                    if (extrapolationIters > 0) {
                        let readIndex = velIndex;
                        let writeIndex = 1 - readIndex;
                        for (let iter = 0; iter < extrapolationIters; iter++) {
                            pass.setPipeline(extrapolatePipeline);
                            pass.setBindGroup(0, extrapolateGroups[matIndex][readIndex][writeIndex]);
                            pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);
                            const tmp = readIndex;
                            readIndex = writeIndex;
                            writeIndex = tmp;
                        }
                        velIndex = readIndex;

                        pass.setPipeline(maskSolidVelPipeline);
                        pass.setBindGroup(0, maskVelGroups[matIndex][velIndex]);
                        pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);
                    }

                    pass.setPipeline(dampVelocityPipeline);
                    pass.setBindGroup(0, dampVelGroups[velIndex]);
                    pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);

                    if (state.vorticityStrength > 0.0) {
                        pass.setPipeline(vorticityPipeline);
                        pass.setBindGroup(0, vorticityGroups[velIndex]);
                        pass.dispatchWorkgroups(cellWorkgroupsX, cellWorkgroupsY);

                        pass.setPipeline(applyVorticityPipeline);
                        pass.setBindGroup(0, applyVorticityGroups[velIndex]);
                        pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);

                        pass.setPipeline(maskSolidVelPipeline);
                        pass.setBindGroup(0, maskVelGroups[matIndex][velIndex]);
                        pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);
                    }

                    if (state.ballEnabled) {
                        pass.setPipeline(ballForcePipeline);
                        pass.setBindGroup(0, ballForceGroups[velIndex]);
                        pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);

                        pass.setPipeline(maskSolidVelPipeline);
                        pass.setBindGroup(0, maskVelGroups[matIndex][velIndex]);
                        pass.dispatchWorkgroups(velWorkgroupsX, velWorkgroupsY);
                    }

                    pass.setPipeline(transferPipeline);
                    pass.setBindGroup(0, transferGroups[matIndex][velIndex]);
                    pass.dispatchWorkgroups(particleWorkgroups);
                }
            }

            pass.end();
        }

        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(renderPipeline);
            pass.setBindGroup(0, renderGroups[matIndex][pressureIndex][velIndex]);
            pass.dispatchWorkgroups(cellWorkgroupsX, cellWorkgroupsY);
            pass.end();
        }

        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.04, g: 0.04, b: 0.08, a: 1 },
            }],
        });

        renderPass.setPipeline(screenPipeline);
        renderPass.setBindGroup(0, screenBindGroup);
        renderPass.draw(4);
        renderPass.end();

        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

init().catch(err => {
    console.error(err);
    showError(`Error: ${err.message}`);
});
