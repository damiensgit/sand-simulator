// Grid dimensions
const WIDTH = 256;
const HEIGHT = 256;
const WORKGROUP_SIZE = 8;

// Cell types (stored in lower 8 bits)
const EMPTY = 0;
const SAND = 1;
const WALL = 2;
const WATER = 3;
const STEAM = 4;
const ICE = 5;
const LAVA = 6;
const ROCK = 7;

// Simulation state
let simSpeed = 4;
let brushSize = 3;
let selectedMaterial = SAND;
let isPaused = false;
let sandColor = [0.92, 0.79, 0.41]; // Default sand color (RGB 0-1)
let colorByAge = false;
let fluidity = 50; // 0 = sticky, 100 = very fluid

// World parameters
let gravity = 1.0;      // 0.1 = moon, 1.0 = earth, 3.0 = heavy
let temperature = 20;   // Celsius: -50 to 1500

async function init() {
    if (!navigator.gpu) {
        document.getElementById('error').textContent =
            'WebGPU not supported! Make sure you\'re using Chrome/Edge with WebGPU enabled.';
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        document.getElementById('error').textContent = 'Failed to get GPU adapter';
        return;
    }

    const device = await adapter.requestDevice();

    const canvas = document.getElementById('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = `${WIDTH * 2}px`;
    canvas.style.height = `${HEIGHT * 2}px`;

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    // Texture stores: bits 0-7 = type, bits 8-23 = age (max 65535)
    const textureDesc = {
        size: { width: WIDTH, height: HEIGHT },
        format: 'r32uint',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    };

    const textures = [
        device.createTexture(textureDesc),
        device.createTexture(textureDesc),
    ];

    // Spawn texture - holds pending spawn requests (0 = no request, 1-255 = material to spawn)
    // 255 = eraser request (special case)
    const spawnTexture = device.createTexture({
        size: { width: WIDTH, height: HEIGHT },
        format: 'r32uint',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Buffer to clear spawn texture each frame
    const emptySpawnData = new Uint32Array(WIDTH * HEIGHT);

    const readbackBuffer = device.createBuffer({
        size: WIDTH * HEIGHT * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    function createInitialData() {
        const data = new Uint32Array(WIDTH * HEIGHT);
        for (let x = 0; x < WIDTH; x++) {
            data[(HEIGHT - 1) * WIDTH + x] = WALL;
        }
        return data;
    }

    function clearGrid() {
        const initialData = createInitialData();
        device.queue.writeTexture(
            { texture: textures[0] },
            initialData,
            { bytesPerRow: WIDTH * 4 },
            { width: WIDTH, height: HEIGHT }
        );
        device.queue.writeTexture(
            { texture: textures[1] },
            initialData,
            { bytesPerRow: WIDTH * 4 },
            { width: WIDTH, height: HEIGHT }
        );
    }

    clearGrid();

    // Compute shader with improved physics + age tracking
    const computeShaderCode = `
            @group(0) @binding(0) var inputTex: texture_storage_2d<r32uint, read>;
            @group(0) @binding(1) var outputTex: texture_storage_2d<r32uint, write>;
            @group(0) @binding(2) var<uniform> params: vec4u; // frame, width, height, fluidity (0-100)
            @group(0) @binding(3) var spawnTex: texture_storage_2d<r32uint, read>; // spawn requests
            @group(0) @binding(4) var<uniform> worldParams: vec4f; // gravity, temperature, 0, 0

            // Material types
            const EMPTY: u32 = 0u;
            const SAND: u32 = 1u;
            const WALL: u32 = 2u;
            const WATER: u32 = 3u;
            const STEAM: u32 = 4u;
            const ICE: u32 = 5u;
            const LAVA: u32 = 6u;
            const ROCK: u32 = 7u;

            // Bit packing constants
            const TYPE_MASK: u32 = 0xFFu;
            const AGE_MASK: u32 = 0xFFFFu;
            const AGE_SHIFT: u32 = 8u;
            const COLOR_SHIFT: u32 = 24u;
            const MAX_AGE: u32 = 65535u;

            // Material property functions
            fn getMaterialDensity(mat: u32) -> f32 {
                // Higher = sinks, Lower = floats, 0 = empty air
                switch(mat) {
                    case 1u: { return 2.0; }   // SAND - sinks in water/lava
                    case 2u: { return 999.0; } // WALL - immovable
                    case 3u: { return 1.0; }   // WATER - baseline liquid
                    case 4u: { return 0.1; }   // STEAM - rises
                    case 5u: { return 0.9; }   // ICE - floats on water
                    case 6u: { return 3.0; }   // LAVA - very heavy liquid
                    case 7u: { return 999.0; } // ROCK - immovable solid
                    default: { return 0.0; }   // EMPTY - air
                }
            }

            fn getMaterialFluidity(mat: u32) -> f32 {
                // How easily it flows (0 = solid, 1 = very liquid)
                switch(mat) {
                    case 1u: { return 0.4; }   // SAND - granular
                    case 3u: { return 1.0; }   // WATER - max fluid
                    case 4u: { return 1.0; }   // STEAM - max fluid (but rises)
                    case 6u: { return 0.6; }   // LAVA - viscous
                    default: { return 0.0; }   // Solids don't flow
                }
            }

            fn isLiquid(mat: u32) -> bool {
                return mat == WATER || mat == LAVA;
            }

            fn isGas(mat: u32) -> bool {
                return mat == STEAM;
            }

            fn isSolid(mat: u32) -> bool {
                // Only WALL is truly immovable - rock and ice can fall
                return mat == WALL;
            }

            fn isMovable(mat: u32) -> bool {
                // Rock and ice fall like sand, so they're movable
                return mat == SAND || mat == WATER || mat == STEAM || mat == LAVA || mat == ROCK || mat == ICE;
            }

            fn isFalling(mat: u32) -> bool {
                // Materials that fall down (not steam which rises)
                return mat == SAND || mat == WATER || mat == LAVA || mat == ROCK || mat == ICE;
            }

            fn canDisplace(mover: u32, dest: u32) -> bool {
                // Can mover displace dest? (based on density)
                if (dest == EMPTY) { return true; }
                if (dest == WALL || dest == ROCK) { return false; }
                return getMaterialDensity(mover) > getMaterialDensity(dest);
            }

            // Simple hash for randomness
            fn hash(p: vec2u, seed: u32) -> u32 {
                var h = p.x * 374761393u + p.y * 668265263u + seed * 1013904223u;
                h = (h ^ (h >> 13u)) * 1274126177u;
                return h ^ (h >> 16u);
            }

            fn getType(value: u32) -> u32 {
                return value & TYPE_MASK;
            }

            fn getAge(value: u32) -> u32 {
                return (value >> AGE_SHIFT) & AGE_MASK;
            }

            fn getColorVar(value: u32) -> u32 {
                return value >> COLOR_SHIFT;
            }

            fn packCell(cellType: u32, age: u32, colorVar: u32) -> u32 {
                return cellType | (min(age, MAX_AGE) << AGE_SHIFT) | (colorVar << COLOR_SHIFT);
            }

            fn getRawCell(pos: vec2i) -> u32 {
                let width = i32(params.y);
                let height = i32(params.z);
                if (pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height) {
                    return WALL;
                }
                return textureLoad(inputTex, pos).r;
            }

            fn getCell(pos: vec2i) -> u32 {
                return getType(getRawCell(pos));
            }

            fn getSandMovement(pos: vec2i, rng: u32) -> vec2i {
                let below = getCell(pos + vec2i(0, 1));
                // Sand falls into EMPTY space only (simplified - no density displacement)
                if (below == EMPTY) {
                    return vec2i(0, 1);
                }

                let belowLeft = getCell(pos + vec2i(-1, 1));
                let belowRight = getCell(pos + vec2i(1, 1));
                let left = getCell(pos + vec2i(-1, 0));
                let right = getCell(pos + vec2i(1, 0));

                // Diagonal fall - only into EMPTY cells
                var canFallLeft = belowLeft == EMPTY && left == EMPTY;
                var canFallRight = belowRight == EMPTY && right == EMPTY;

                // Check for priority conflicts - sand directly beside us falling straight down
                // Block diagonal if neighbor CAN fall straight (they have priority)
                if (canFallLeft && left == SAND) {
                    let leftBelow = getCell(pos + vec2i(-1, 1));
                    if (leftBelow == EMPTY) {
                        canFallLeft = false;
                    }
                }
                if (canFallRight && right == SAND) {
                    let rightBelow = getCell(pos + vec2i(1, 1));
                    if (rightBelow == EMPTY) {
                        canFallRight = false;
                    }
                }

                // Position-based direction to prevent diagonal collisions
                let preferLeft = (pos.x % 2) == 0;

                if (canFallLeft && canFallRight) {
                    if (preferLeft) {
                        return vec2i(-1, 1);
                    } else {
                        return vec2i(1, 1);
                    }
                } else if (canFallLeft) {
                    return vec2i(-1, 1);
                } else if (canFallRight) {
                    return vec2i(1, 1);
                }

                // SETTLING: Sand slides sideways to find lower ground
                // But settling urge decreases with age - old sand stays put
                // Fluidity affects how eager sand is to settle (0=sticky, 100=fluid)
                let currentAge = getAge(getRawCell(pos));
                let settleRoll = rng % 100u;
                let fluidityFactor = f32(params.w) / 50.0; // 0-2 range, 1.0 at fluidity=50

                // Base settle chances, scaled by fluidity
                // Age 0-10: base 60% settle chance (fresh, active)
                // Age 10-30: base 30% settle chance (slowing down)
                // Age 30-60: base 12% settle chance (mostly settled)
                // Age 60+: base 2% settle chance (rare adjustments)
                var baseChance = 60.0;
                if (currentAge > 10u) { baseChance = 30.0; }
                if (currentAge > 30u) { baseChance = 12.0; }
                if (currentAge > 60u) { baseChance = 2.0; }

                let maxSettleChance = u32(min(baseChance * fluidityFactor, 95.0));

                // Only consider settling if we pass the age-based probability check
                if (settleRoll >= maxSettleChance) {
                    return vec2i(0, 0);
                }

                // Look further for paths to lower ground
                let left2 = getCell(pos + vec2i(-2, 0));
                let right2 = getCell(pos + vec2i(2, 0));
                let belowLeft2 = getCell(pos + vec2i(-2, 1));
                let belowRight2 = getCell(pos + vec2i(2, 1));

                // Can slide if there's a path to eventually fall:
                // - Immediate: diagonal below is empty (will fall next frame)
                // - Extended: 2 cells out has empty diagonal (will fall after sliding twice)
                var canSlideLeft = left == EMPTY && (belowLeft == EMPTY || (left2 == EMPTY && belowLeft2 == EMPTY));
                var canSlideRight = right == EMPTY && (belowRight == EMPTY || (right2 == EMPTY && belowRight2 == EMPTY));

                // Prevent conflicts: check if sand 2 cells away might also slide to same destination
                if (canSlideLeft && left2 == SAND) {
                    let left2BelowRight = getCell(pos + vec2i(-1, 1));
                    if (left2BelowRight == EMPTY) {
                        canSlideLeft = false;
                    }
                }
                if (canSlideRight && right2 == SAND) {
                    let right2BelowLeft = getCell(pos + vec2i(1, 1));
                    if (right2BelowLeft == EMPTY) {
                        canSlideRight = false;
                    }
                }

                if (canSlideLeft && canSlideRight) {
                    if (preferLeft) {
                        return vec2i(-1, 0);
                    } else {
                        return vec2i(1, 0);
                    }
                } else if (canSlideLeft) {
                    return vec2i(-1, 0);
                } else if (canSlideRight) {
                    return vec2i(1, 0);
                }

                return vec2i(0, 0);
            }

            // Liquid movement - fall, then horizontal leveling, then diagonal
            fn getLiquidMovement(pos: vec2i, rng: u32, liquidType: u32, frame: u32) -> vec2i {
                let gravity = worldParams.x;
                let gravityRoll = f32(rng % 100u) / 100.0;
                if (gravityRoll > gravity) {
                    return vec2i(0, 0);
                }

                let below = getCell(pos + vec2i(0, 1));
                let left = getCell(pos + vec2i(-1, 0));
                let right = getCell(pos + vec2i(1, 0));
                let belowLeft = getCell(pos + vec2i(-1, 1));
                let belowRight = getCell(pos + vec2i(1, 1));

                // 1. FALL STRAIGHT DOWN
                if (below == EMPTY) {
                    return vec2i(0, 1);
                }

                let preferLeft = ((u32(pos.x) + frame) & 1u) == 0u;

                // 2. HORIZONTAL SPREAD
                let canSpreadLeft = left == EMPTY;
                let canSpreadRight = right == EMPTY;

                if (canSpreadLeft && canSpreadRight) {
                    if (preferLeft) {
                        return vec2i(-1, 0);
                    }
                    return vec2i(1, 0);
                } else if (canSpreadLeft) {
                    return vec2i(-1, 0);
                } else if (canSpreadRight) {
                    return vec2i(1, 0);
                }

                // 3. DIAGONAL FALL
                let canFallLeft = belowLeft == EMPTY && left == EMPTY;
                let canFallRight = belowRight == EMPTY && right == EMPTY;

                if (canFallLeft && canFallRight) {
                    if (preferLeft) {
                        return vec2i(-1, 1);
                    }
                    return vec2i(1, 1);
                } else if (canFallLeft) {
                    return vec2i(-1, 1);
                } else if (canFallRight) {
                    return vec2i(1, 1);
                }

                return vec2i(0, 0);
            }

            // Gas movement (steam) - rises and drifts
            fn getGasMovement(pos: vec2i, rng: u32) -> vec2i {
                let gravity = worldParams.x;

                // Gases always try to rise (inverted gravity effect)
                let above = getCell(pos + vec2i(0, -1));

                // Rise into empty space
                if (above == EMPTY) {
                    // In high gravity, steam rises slower
                    if (f32(rng % 100u) / 100.0 < (2.0 - gravity)) {
                        return vec2i(0, -1);
                    }
                }

                // Diagonal rise
                let aboveLeft = getCell(pos + vec2i(-1, -1));
                let aboveRight = getCell(pos + vec2i(1, -1));
                let left = getCell(pos + vec2i(-1, 0));
                let right = getCell(pos + vec2i(1, 0));

                let canRiseLeft = aboveLeft == EMPTY && left == EMPTY;
                let canRiseRight = aboveRight == EMPTY && right == EMPTY;

                // Position-based direction to prevent collisions
                let preferLeft = (pos.x % 2) == 0;

                if (canRiseLeft && canRiseRight) {
                    if (preferLeft) { return vec2i(-1, -1); }
                    else { return vec2i(1, -1); }
                } else if (canRiseLeft) {
                    return vec2i(-1, -1);
                } else if (canRiseRight) {
                    return vec2i(1, -1);
                }

                // Horizontal drift when can't rise (random chance to drift)
                if ((rng % 3u) == 0u) {
                    if (left == EMPTY && preferLeft) { return vec2i(-1, 0); }
                    if (right == EMPTY && !preferLeft) { return vec2i(1, 0); }
                }

                return vec2i(0, 0);
            }

            // Solid falling movement (rock, ice) - falls but doesn't spread horizontally
            fn getSolidFallMovement(pos: vec2i, rng: u32) -> vec2i {
                let below = getCell(pos + vec2i(0, 1));
                if (below == EMPTY) {
                    return vec2i(0, 1);
                }

                // Diagonal fall only
                let belowLeft = getCell(pos + vec2i(-1, 1));
                let belowRight = getCell(pos + vec2i(1, 1));
                let left = getCell(pos + vec2i(-1, 0));
                let right = getCell(pos + vec2i(1, 0));

                let canFallLeft = belowLeft == EMPTY && left == EMPTY;
                let canFallRight = belowRight == EMPTY && right == EMPTY;

                // Position-based direction to prevent collisions
                let preferLeft = (pos.x % 2) == 0;

                if (canFallLeft && canFallRight) {
                    if (preferLeft) { return vec2i(-1, 1); }
                    else { return vec2i(1, 1); }
                } else if (canFallLeft) {
                    return vec2i(-1, 1);
                } else if (canFallRight) {
                    return vec2i(1, 1);
                }

                return vec2i(0, 0);
            }

            // General movement dispatcher
            fn getMovement(cellType: u32, pos: vec2i, rng: u32, frame: u32) -> vec2i {
                if (cellType == SAND) { return getSandMovement(pos, rng); }
                if (cellType == WATER) { return getLiquidMovement(pos, rng, WATER, frame); }
                if (cellType == LAVA) { return getLiquidMovement(pos, rng, LAVA, frame); }
                if (cellType == STEAM) { return getGasMovement(pos, rng); }
                if (cellType == ROCK || cellType == ICE) { return getSolidFallMovement(pos, rng); }
                return vec2i(0, 0);  // WALL doesn't move
            }

            // Temperature state changes (temp is in Celsius)
            fn applyTemperature(cellType: u32, temp: f32, rng: u32) -> u32 {
                // Water state changes
                if (cellType == WATER) {
                    if (temp < 0.0 && rng % 100u < 3u) { return ICE; }     // Freeze below 0°C
                    if (temp > 100.0 && rng % 100u < 3u) { return STEAM; } // Boil above 100°C
                }
                if (cellType == ICE) {
                    if (temp > 0.0 && rng % 100u < 5u) { return WATER; }   // Melt above 0°C
                }
                if (cellType == STEAM) {
                    if (temp < 100.0 && rng % 100u < 3u) { return WATER; } // Condense below 100°C
                }

                // Lava state changes - only from extreme temperatures
                // Normal solidification happens via water contact (checkMaterialInteraction)
                if (cellType == LAVA) {
                    if (temp < -20.0 && rng % 100u < 2u) { return ROCK; }  // Freeze only at extreme cold
                }
                if (cellType == ROCK) {
                    if (temp > 1200.0 && rng % 100u < 1u) { return LAVA; } // Melt above 1200°C
                }

                return cellType;  // No change
            }

            // Material interactions (lava + water)
            fn checkMaterialInteraction(pos: vec2i, current: u32, rng: u32) -> u32 {
                // Check adjacent cells for interactions
                let neighborBelow = getCell(pos + vec2i(0, 1));
                let neighborAbove = getCell(pos + vec2i(0, -1));
                let neighborLeft = getCell(pos + vec2i(-1, 0));
                let neighborRight = getCell(pos + vec2i(1, 0));

                // Water + Lava = Steam (water becomes steam)
                if (current == WATER) {
                    if (neighborBelow == LAVA || neighborAbove == LAVA ||
                        neighborLeft == LAVA || neighborRight == LAVA) {
                        if (rng % 100u < 50u) { return STEAM; }
                    }
                }

                // Lava + Water = Rock (lava becomes rock)
                if (current == LAVA) {
                    if (neighborBelow == WATER || neighborAbove == WATER ||
                        neighborLeft == WATER || neighborRight == WATER) {
                        if (rng % 100u < 50u) { return ROCK; }
                    }
                }

                return current;
            }

            // Check if material from a neighbor is moving into this position
            fn checkIncoming(pos: vec2i, fromOffset: vec2i, frame: u32) -> u32 {
                let fromPos = pos + fromOffset;
                let fromRaw = getRawCell(fromPos);
                let fromType = getType(fromRaw);

                if (!isMovable(fromType)) { return 0u; }

                let fromRng = hash(vec2u(u32(fromPos.x), u32(fromPos.y)), frame);
                let movement = getMovement(fromType, fromPos, fromRng, frame);

                if (fromPos.x + movement.x == pos.x && fromPos.y + movement.y == pos.y) {
                    return fromRaw;
                }
                return 0u;
            }

            // Check if any higher-priority neighbor would move into destPos
            fn hasHigherPriorityIncoming(destPos: vec2i, myOffset: vec2i, frame: u32) -> bool {
                let flipIncoming = ((u32(destPos.x + destPos.y) + frame) & 1u) == 1u;

                if (myOffset.x == 0 && myOffset.y == -1) { return false; }
                if (checkIncoming(destPos, vec2i(0, -1), frame) != 0u) { return true; }

                if (!flipIncoming) {
                    if (myOffset.x == -1 && myOffset.y == 0) { return false; }
                    if (checkIncoming(destPos, vec2i(-1, 0), frame) != 0u) { return true; }

                    if (myOffset.x == 1 && myOffset.y == 0) { return false; }
                    if (checkIncoming(destPos, vec2i(1, 0), frame) != 0u) { return true; }
                } else {
                    if (myOffset.x == 1 && myOffset.y == 0) { return false; }
                    if (checkIncoming(destPos, vec2i(1, 0), frame) != 0u) { return true; }

                    if (myOffset.x == -1 && myOffset.y == 0) { return false; }
                    if (checkIncoming(destPos, vec2i(-1, 0), frame) != 0u) { return true; }
                }

                if (!flipIncoming) {
                    if (myOffset.x == -1 && myOffset.y == -1) { return false; }
                    if (checkIncoming(destPos, vec2i(-1, -1), frame) != 0u) { return true; }

                    if (myOffset.x == 1 && myOffset.y == -1) { return false; }
                    if (checkIncoming(destPos, vec2i(1, -1), frame) != 0u) { return true; }
                } else {
                    if (myOffset.x == 1 && myOffset.y == -1) { return false; }
                    if (checkIncoming(destPos, vec2i(1, -1), frame) != 0u) { return true; }

                    if (myOffset.x == -1 && myOffset.y == -1) { return false; }
                    if (checkIncoming(destPos, vec2i(-1, -1), frame) != 0u) { return true; }
                }

                if (myOffset.x == 0 && myOffset.y == 1) { return false; }
                if (checkIncoming(destPos, vec2i(0, 1), frame) != 0u) { return true; }

                if (!flipIncoming) {
                    if (myOffset.x == -1 && myOffset.y == 1) { return false; }
                    if (checkIncoming(destPos, vec2i(-1, 1), frame) != 0u) { return true; }

                    if (myOffset.x == 1 && myOffset.y == 1) { return false; }
                    if (checkIncoming(destPos, vec2i(1, 1), frame) != 0u) { return true; }
                } else {
                    if (myOffset.x == 1 && myOffset.y == 1) { return false; }
                    if (checkIncoming(destPos, vec2i(1, 1), frame) != 0u) { return true; }

                    if (myOffset.x == -1 && myOffset.y == 1) { return false; }
                    if (checkIncoming(destPos, vec2i(-1, 1), frame) != 0u) { return true; }
                }

                return false;
            }


            @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let pos = vec2i(id.xy);
                let width = i32(params.y);
                let height = i32(params.z);
                let frame = params.x;
                let temp = worldParams.y;

                if (pos.x >= width || pos.y >= height) {
                    return;
                }

                let rawCell = getRawCell(pos);
                var current = getType(rawCell);
                var currentAge = getAge(rawCell);
                var currentColorVar = getColorVar(rawCell);

                // Generate randomness for this cell
                let rng = hash(vec2u(id.xy), frame);

                // Read spawn request
                let spawnRequest = textureLoad(spawnTex, pos).r;

                var resultType = current;
                var resultAge = currentAge;
                var resultColorVar = currentColorVar;

                // Handle eraser for any material
                if (spawnRequest == 255u && current != EMPTY) {
                    resultType = EMPTY;
                    resultAge = 0u;
                    resultColorVar = 0u;
                    textureStore(outputTex, pos, vec4u(packCell(resultType, resultAge, resultColorVar), 0u, 0u, 0u));
                    return;
                }

                // Immovable solids (WALL only)
                if (isSolid(current)) {
                    textureStore(outputTex, pos, vec4u(packCell(resultType, resultAge, resultColorVar), 0u, 0u, 0u));
                    return;
                }

                // Movable materials (SAND, WATER, LAVA, STEAM, ROCK, ICE)
                if (isMovable(current)) {
                    // Check for material interactions first
                    let interacted = checkMaterialInteraction(pos, current, rng);
                    if (interacted != current) {
                        resultType = interacted;
                        resultAge = 0u;
                        textureStore(outputTex, pos, vec4u(packCell(resultType, resultAge, resultColorVar), 0u, 0u, 0u));
                        return;
                    }

                    // Apply temperature effects
                    let tempChanged = applyTemperature(current, temp, rng);
                    if (tempChanged != current) {
                        resultType = tempChanged;
                        resultAge = 0u;
                        textureStore(outputTex, pos, vec4u(packCell(resultType, resultAge, resultColorVar), 0u, 0u, 0u));
                        return;
                    }

                    // Check movement
                    var movement = getMovement(current, pos, rng, frame);
                    if (movement.x != 0 || movement.y != 0) {
                        let destPos = pos + movement;
                        let myOffset = vec2i(-movement.x, -movement.y);
                        if (hasHigherPriorityIncoming(destPos, myOffset, frame)) {
                            movement = vec2i(0, 0);
                        }
                    }

                    if (movement.x != 0 || movement.y != 0) {
                        // Moving - become empty (higher-priority checks prevent conflicts)
                        resultType = EMPTY;
                        resultAge = 0u;
                        resultColorVar = 0u;
                    } else {
                        // Staying put - age increases
                        resultAge = currentAge + 1u;
                    }
                }

                // EMPTY cell - check for incoming materials
                if (current == EMPTY || resultType == EMPTY) {
                    var incomingRaw = 0u;
                    let flipIncoming = ((u32(pos.x + pos.y) + frame) & 1u) == 1u;

                    // Check incoming in priority order (above has highest priority)
                    // Vertical falling - highest priority
                    if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(0, -1), frame); }
                    // Horizontal spreading - second priority
                    if (!flipIncoming) {
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(-1, 0), frame); }
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(1, 0), frame); }
                    } else {
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(1, 0), frame); }
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(-1, 0), frame); }
                    }
                    // Diagonal falling - third priority
                    if (!flipIncoming) {
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(-1, -1), frame); }
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(1, -1), frame); }
                    } else {
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(1, -1), frame); }
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(-1, -1), frame); }
                    }
                    // Rising (for gases) - lowest priority
                    if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(0, 1), frame); }
                    if (!flipIncoming) {
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(-1, 1), frame); }
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(1, 1), frame); }
                    } else {
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(1, 1), frame); }
                        if (incomingRaw == 0u) { incomingRaw = checkIncoming(pos, vec2i(-1, 1), frame); }
                    }

                    if (incomingRaw != 0u) {
                        // Something moved in
                        resultType = getType(incomingRaw);
                        resultAge = 0u;
                        resultColorVar = getColorVar(incomingRaw);
                    } else if (current == EMPTY && spawnRequest > 0u && spawnRequest < 255u) {
                        // Spawn new material
                        resultType = spawnRequest;
                        resultAge = 0u;
                        resultColorVar = rng % 256u;
                    }
                }

                textureStore(outputTex, pos, vec4u(packCell(resultType, resultAge, resultColorVar), 0u, 0u, 0u));
            }
        `;

    const computeShader = device.createShaderModule({
        label: 'Sand Compute Shader',
        code: computeShaderCode
    });

    // Check for shader compilation errors
    computeShader.getCompilationInfo().then(info => {
        if (info.messages.length > 0) {
            console.error('Compute shader compilation messages:');
            info.messages.forEach(msg => {
                console.error(`  ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
            });
        }
    });

    // Render shader with color options
    const renderShaderCode = `
            @group(0) @binding(0) var inputTex: texture_2d<u32>;
            @group(0) @binding(1) var<uniform> colorParams: vec4f; // r, g, b, colorByAge

            const TYPE_MASK: u32 = 0xFFu;
            const AGE_MASK: u32 = 0xFFFFu;
            const AGE_SHIFT: u32 = 8u;
            const COLOR_SHIFT: u32 = 24u;

            struct VertexOutput {
                @builtin(position) pos: vec4f,
                @location(0) uv: vec2f,
            }

            @vertex
            fn vs(@builtin(vertex_index) i: u32) -> VertexOutput {
                var pos = array<vec2f, 3>(
                    vec2f(-1.0, -1.0),
                    vec2f(3.0, -1.0),
                    vec2f(-1.0, 3.0)
                );
                var uv = array<vec2f, 3>(
                    vec2f(0.0, 1.0),
                    vec2f(2.0, 1.0),
                    vec2f(0.0, -1.0)
                );
                var out: VertexOutput;
                out.pos = vec4f(pos[i], 0.0, 1.0);
                out.uv = uv[i];
                return out;
            }

            // HSV to RGB conversion
            fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3f {
                let c = v * s;
                let hp = h * 6.0;
                let x = c * (1.0 - abs(hp % 2.0 - 1.0));
                var rgb: vec3f;
                if (hp < 1.0) { rgb = vec3f(c, x, 0.0); }
                else if (hp < 2.0) { rgb = vec3f(x, c, 0.0); }
                else if (hp < 3.0) { rgb = vec3f(0.0, c, x); }
                else if (hp < 4.0) { rgb = vec3f(0.0, x, c); }
                else if (hp < 5.0) { rgb = vec3f(x, 0.0, c); }
                else { rgb = vec3f(c, 0.0, x); }
                return rgb + vec3f(v - c);
            }

            // Material type constants for rendering
            const EMPTY_R: u32 = 0u;
            const SAND_R: u32 = 1u;
            const WALL_R: u32 = 2u;
            const WATER_R: u32 = 3u;
            const STEAM_R: u32 = 4u;
            const ICE_R: u32 = 5u;
            const LAVA_R: u32 = 6u;
            const ROCK_R: u32 = 7u;

            @fragment
            fn fs(in: VertexOutput) -> @location(0) vec4f {
                let texSize = vec2f(textureDimensions(inputTex));
                let texCoord = vec2i(in.uv * texSize);
                let rawCell = textureLoad(inputTex, texCoord, 0).r;

                let cellType = rawCell & TYPE_MASK;
                let age = (rawCell >> AGE_SHIFT) & AGE_MASK;
                let colorVar = rawCell >> COLOR_SHIFT; // 0-255 per-particle variation

                // Per-particle brightness variation
                let varNorm = f32(colorVar) / 255.0;
                let brightMult = 0.95 + varNorm * 0.10;

                if (cellType == SAND_R) {
                    // Sand - user-selected color
                    let baseColor = vec3f(colorParams.r, colorParams.g, colorParams.b);

                    if (colorParams.a > 0.5) {
                        // Color by age mode
                        let normalizedAge = min(f32(age) / 500.0, 1.0);
                        let hue = 0.08 - normalizedAge * 0.06;
                        let sat = 0.9 - normalizedAge * 0.3;
                        let val = (1.0 - normalizedAge * 0.5) * brightMult;
                        return vec4f(hsv2rgb(hue, sat, clamp(val, 0.2, 1.0)), 1.0);
                    } else {
                        // Standard color with subtle per-particle brightness variation
                        return vec4f(clamp(baseColor * brightMult, vec3f(0.0), vec3f(1.0)), 1.0);
                    }
                } else if (cellType == WALL_R) {
                    // Wall - gray stone
                    return vec4f(0.45, 0.45, 0.5, 1.0);
                } else if (cellType == WATER_R) {
                    // Water - blue, semi-transparent look
                    let waterBlue = vec3f(0.2, 0.5, 0.9) * brightMult;
                    // Add slight wave effect based on position and age
                    let wave = sin(f32(age) * 0.1 + varNorm * 6.28) * 0.05;
                    return vec4f(clamp(waterBlue + wave, vec3f(0.0), vec3f(1.0)), 1.0);
                } else if (cellType == STEAM_R) {
                    // Steam - white/gray, translucent effect
                    let steamGray = 0.7 + varNorm * 0.2;
                    // Fade with age (dissipation visual)
                    let fade = 1.0 - min(f32(age) / 200.0, 0.5);
                    return vec4f(steamGray * fade, steamGray * fade, steamGray * fade * 1.05, 1.0);
                } else if (cellType == ICE_R) {
                    // Ice - light blue, crystalline
                    let iceColor = vec3f(0.7, 0.85, 0.95) * brightMult;
                    return vec4f(iceColor, 1.0);
                } else if (cellType == LAVA_R) {
                    // Lava - glowing orange/red with pulsing effect
                    let pulse = sin(f32(age) * 0.15 + varNorm * 3.14) * 0.15 + 0.85;
                    let lavaColor = vec3f(1.0, 0.3 + varNorm * 0.2, 0.1) * pulse;
                    return vec4f(clamp(lavaColor, vec3f(0.0), vec3f(1.0)), 1.0);
                } else if (cellType == ROCK_R) {
                    // Rock - dark gray, solid
                    let rockColor = vec3f(0.3, 0.28, 0.28) * brightMult;
                    return vec4f(rockColor, 1.0);
                }
                // Empty - dark background
                return vec4f(0.08, 0.08, 0.12, 1.0);
            }
        `;

    const renderShader = device.createShaderModule({
        label: 'Render Shader',
        code: renderShaderCode
    });

    // Check for render shader compilation errors
    renderShader.getCompilationInfo().then(info => {
        if (info.messages.length > 0) {
            console.error('Render shader compilation messages:');
            info.messages.forEach(msg => {
                console.error(`  ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
            });
        }
    });

    // Pipelines
    const computePipeline = device.createComputePipeline({
        label: 'Sand Compute Pipeline',
        layout: 'auto',
        compute: { module: computeShader, entryPoint: 'main' },
    });

    const renderPipeline = device.createRenderPipeline({
        label: 'Render Pipeline',
        layout: 'auto',
        vertex: { module: renderShader, entryPoint: 'vs' },
        fragment: { module: renderShader, entryPoint: 'fs', targets: [{ format }] },
    });

    // Buffers
    const paramsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const colorParamsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const worldParamsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compute bind groups
    const spawnTextureView = spawnTexture.createView();
    const computeBindGroups = [
        device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: textures[0].createView() },
                { binding: 1, resource: textures[1].createView() },
                { binding: 2, resource: { buffer: paramsBuffer } },
                { binding: 3, resource: spawnTextureView },
                { binding: 4, resource: { buffer: worldParamsBuffer } },
            ],
        }),
        device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: textures[1].createView() },
                { binding: 1, resource: textures[0].createView() },
                { binding: 2, resource: { buffer: paramsBuffer } },
                { binding: 3, resource: spawnTextureView },
                { binding: 4, resource: { buffer: worldParamsBuffer } },
            ],
        }),
    ];

    // Render bind groups (need both texture and color params)
    const renderBindGroups = [
        device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: textures[1].createView() },
                { binding: 1, resource: { buffer: colorParamsBuffer } },
            ],
        }),
        device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: textures[0].createView() },
                { binding: 1, resource: { buffer: colorParamsBuffer } },
            ],
        }),
    ];

    // =========================================================================
    // UI Controls
    // =========================================================================
    const fpsElement = document.getElementById('fps');
    const particleCountElement = document.getElementById('particleCount');
    const brushSizeSlider = document.getElementById('brushSize');
    const brushValueDisplay = document.getElementById('brushValue');
    const simSpeedSlider = document.getElementById('simSpeed');
    const speedValueDisplay = document.getElementById('speedValue');
    const pauseBtn = document.getElementById('pauseBtn');
    const clearBtn = document.getElementById('clearBtn');
    const materialBtns = document.querySelectorAll('.material-btn');
    const colorPicker = document.getElementById('sandColor');
    const colorByAgeCheckbox = document.getElementById('colorByAge');
    const fluiditySlider = document.getElementById('fluidity');
    const fluidityValueDisplay = document.getElementById('fluidityValue');

    // Fluidity control
    fluiditySlider.addEventListener('input', (e) => {
        fluidity = parseInt(e.target.value);
        fluidityValueDisplay.textContent = fluidity;
    });

    // Color picker
    colorPicker.addEventListener('input', (e) => {
        const hex = e.target.value;
        sandColor[0] = parseInt(hex.slice(1, 3), 16) / 255;
        sandColor[1] = parseInt(hex.slice(3, 5), 16) / 255;
        sandColor[2] = parseInt(hex.slice(5, 7), 16) / 255;
    });

    // Color by age toggle
    colorByAgeCheckbox.addEventListener('change', (e) => {
        colorByAge = e.target.checked;
    });

    // Brush size control
    brushSizeSlider.addEventListener('input', (e) => {
        brushSize = parseInt(e.target.value);
        brushValueDisplay.textContent = brushSize;
    });

    // Sim speed control
    simSpeedSlider.addEventListener('input', (e) => {
        simSpeed = parseInt(e.target.value);
        speedValueDisplay.textContent = simSpeed;
    });

    // World parameter controls
    const gravitySlider = document.getElementById('gravity');
    const gravityValueDisplay = document.getElementById('gravityValue');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValueDisplay = document.getElementById('temperatureValue');

    if (gravitySlider) {
        gravitySlider.addEventListener('input', (e) => {
            gravity = parseFloat(e.target.value);
            gravityValueDisplay.textContent = gravity.toFixed(1);
        });
    }

    if (temperatureSlider) {
        temperatureSlider.addEventListener('input', (e) => {
            temperature = parseInt(e.target.value);
            temperatureValueDisplay.textContent = temperature + '°C';
        });
    }

    // Material selection
    materialBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            materialBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const mat = btn.dataset.material;
            if (mat === 'sand') selectedMaterial = SAND;
            else if (mat === 'water') selectedMaterial = WATER;
            else if (mat === 'lava') selectedMaterial = LAVA;
            else if (mat === 'wall') selectedMaterial = WALL;
            else if (mat === 'eraser') selectedMaterial = EMPTY;
        });
    });

    // Pause button
    pauseBtn.addEventListener('click', () => {
        isPaused = !isPaused;
        pauseBtn.textContent = isPaused ? 'Play' : 'Pause';
        pauseBtn.classList.toggle('paused', isPaused);
    });

    // Clear button
    clearBtn.addEventListener('click', clearGrid);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;

        switch(e.key) {
            case '1':
                selectMaterial('sand');
                break;
            case '2':
                selectMaterial('water');
                break;
            case '3':
                selectMaterial('lava');
                break;
            case '4':
                selectMaterial('wall');
                break;
            case '5':
                selectMaterial('eraser');
                break;
            case ' ':
                e.preventDefault();
                pauseBtn.click();
                break;
            case 'c':
            case 'C':
                clearGrid();
                break;
            case '[':
                brushSize = Math.max(1, brushSize - 1);
                brushSizeSlider.value = brushSize;
                brushValueDisplay.textContent = brushSize;
                break;
            case ']':
                brushSize = Math.min(15, brushSize + 1);
                brushSizeSlider.value = brushSize;
                brushValueDisplay.textContent = brushSize;
                break;
            case 'a':
            case 'A':
                colorByAgeCheckbox.checked = !colorByAgeCheckbox.checked;
                colorByAge = colorByAgeCheckbox.checked;
                break;
            case '-':
            case '_':
                fluidity = Math.max(0, fluidity - 10);
                fluiditySlider.value = fluidity;
                fluidityValueDisplay.textContent = fluidity;
                break;
            case '=':
            case '+':
                fluidity = Math.min(100, fluidity + 10);
                fluiditySlider.value = fluidity;
                fluidityValueDisplay.textContent = fluidity;
                break;
            case 'g':
                gravity = Math.max(0.1, gravity - 0.2);
                if (gravitySlider) {
                    gravitySlider.value = gravity;
                    gravityValueDisplay.textContent = gravity.toFixed(1);
                }
                break;
            case 'G':
                gravity = Math.min(3.0, gravity + 0.2);
                if (gravitySlider) {
                    gravitySlider.value = gravity;
                    gravityValueDisplay.textContent = gravity.toFixed(1);
                }
                break;
            case 't':
                temperature = Math.max(-50, temperature - 50);
                if (temperatureSlider) {
                    temperatureSlider.value = temperature;
                    temperatureValueDisplay.textContent = temperature + '°C';
                }
                break;
            case 'T':
                temperature = Math.min(1500, temperature + 50);
                if (temperatureSlider) {
                    temperatureSlider.value = temperature;
                    temperatureValueDisplay.textContent = temperature + '°C';
                }
                break;
        }
    });

    function selectMaterial(mat) {
        materialBtns.forEach(b => {
            b.classList.toggle('active', b.dataset.material === mat);
        });
        if (mat === 'sand') selectedMaterial = SAND;
        else if (mat === 'water') selectedMaterial = WATER;
        else if (mat === 'lava') selectedMaterial = LAVA;
        else if (mat === 'wall') selectedMaterial = WALL;
        else if (mat === 'eraser') selectedMaterial = EMPTY;
    }

    // =========================================================================
    // Mouse interaction
    // =========================================================================
    let mouseDown = false;
    let mousePos = { x: 0, y: 0 };

    canvas.addEventListener('mousedown', (e) => {
        mouseDown = true;
        updateMousePos(e);
    });

    canvas.addEventListener('mouseup', () => {
        mouseDown = false;
    });

    canvas.addEventListener('mouseleave', () => {
        mouseDown = false;
    });

    canvas.addEventListener('mousemove', updateMousePos);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    function updateMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        mousePos.x = Math.floor((e.clientX - rect.left) / (rect.width / WIDTH));
        mousePos.y = Math.floor((e.clientY - rect.top) / (rect.height / HEIGHT));
    }

    function spawnParticles(type) {
        const data = new Uint32Array(1);
        // Use 255 for eraser (EMPTY), otherwise use the material type
        data[0] = type === EMPTY ? 255 : type;

        for (let dy = -brushSize; dy <= brushSize; dy++) {
            for (let dx = -brushSize; dx <= brushSize; dx++) {
                if (dx * dx + dy * dy <= brushSize * brushSize) {
                    const x = mousePos.x + dx;
                    const y = mousePos.y + dy;
                    if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT - 1) {
                        // Write to spawn texture - shader will only apply to empty cells
                        device.queue.writeTexture(
                            { texture: spawnTexture, origin: { x, y } },
                            data,
                            { bytesPerRow: 4 },
                            { width: 1, height: 1 }
                        );
                    }
                }
            }
        }
    }

    // =========================================================================
    // Particle counting
    // =========================================================================
    let countFrame = 0;

    async function updateParticleCount(texture) {
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer(
            { texture },
            { buffer: readbackBuffer, bytesPerRow: WIDTH * 4 },
            { width: WIDTH, height: HEIGHT }
        );
        device.queue.submit([encoder.finish()]);

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(readbackBuffer.getMappedRange());
        let count = 0;
        for (let i = 0; i < data.length; i++) {
            const type = data[i] & 0xFF;
            // Count all non-empty, non-wall materials
            if (type !== EMPTY && type !== WALL && type !== ROCK) count++;
        }
        readbackBuffer.unmap();
        particleCountElement.textContent = count.toLocaleString();
    }

    // =========================================================================
    // Main render loop
    // =========================================================================
    let frame = 0;
    let lastTime = performance.now();
    let frameCount = 0;

    function render() {
        frameCount++;
        const now = performance.now();
        if (now - lastTime >= 1000) {
            fpsElement.textContent = `${frameCount} FPS`;
            frameCount = 0;
            lastTime = now;
        }

        const readIndex = frame % 2;

        if (mouseDown) {
            spawnParticles(selectedMaterial);
        }

        // Update color params
        device.queue.writeBuffer(colorParamsBuffer, 0, new Float32Array([
            sandColor[0], sandColor[1], sandColor[2], colorByAge ? 1.0 : 0.0
        ]));

        // Update world params
        device.queue.writeBuffer(worldParamsBuffer, 0, new Float32Array([
            gravity, temperature, 0.0, 0.0
        ]));

        const encoder = device.createCommandEncoder();

        if (!isPaused) {
            for (let step = 0; step < simSpeed; step++) {
                const currentReadIndex = (frame + step) % 2;
                device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([frame + step, WIDTH, HEIGHT, fluidity]));

                const computePass = encoder.beginComputePass();
                computePass.setPipeline(computePipeline);
                computePass.setBindGroup(0, computeBindGroups[currentReadIndex]);
                computePass.dispatchWorkgroups(
                    Math.ceil(WIDTH / WORKGROUP_SIZE),
                    Math.ceil(HEIGHT / WORKGROUP_SIZE)
                );
                computePass.end();
            }
        }

        const renderIndex = isPaused ? readIndex : (frame + simSpeed) % 2;

        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
            }],
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroups[(renderIndex + 1) % 2]);
        renderPass.draw(3);
        renderPass.end();

        device.queue.submit([encoder.finish()]);

        // Clear spawn texture for next frame
        device.queue.writeTexture(
            { texture: spawnTexture },
            emptySpawnData,
            { bytesPerRow: WIDTH * 4 },
            { width: WIDTH, height: HEIGHT }
        );

        if (!isPaused) {
            frame += simSpeed;
        }

        countFrame++;
        if (countFrame % 30 === 0) {
            const countTexture = textures[isPaused ? readIndex : (renderIndex + 1) % 2];
            updateParticleCount(countTexture);
        }

        requestAnimationFrame(render);
    }

    render();
    console.log('Sand simulator initialized!');
}

init().catch(console.error);
