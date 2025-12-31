// Grid dimensions
const WIDTH = 256;
const HEIGHT = 256;
const WORKGROUP_SIZE = 8;

// Cell types (stored in lower 8 bits)
const EMPTY = 0;
const SAND = 1;
const WALL = 2;

// Simulation state
let simSpeed = 4;
let brushSize = 3;
let selectedMaterial = SAND;
let isPaused = false;
let sandColor = [0.92, 0.79, 0.41]; // Default sand color (RGB 0-1)
let colorByAge = false;

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
    const computeShader = device.createShaderModule({
        label: 'Sand Compute Shader',
        code: `
            @group(0) @binding(0) var inputTex: texture_storage_2d<r32uint, read>;
            @group(0) @binding(1) var outputTex: texture_storage_2d<r32uint, write>;
            @group(0) @binding(2) var<uniform> params: vec4u; // frame, width, height, unused
            @group(0) @binding(3) var spawnTex: texture_storage_2d<r32uint, read>; // spawn requests

            const EMPTY: u32 = 0u;
            const SAND: u32 = 1u;
            const WALL: u32 = 2u;
            const TYPE_MASK: u32 = 0xFFu;
            const AGE_SHIFT: u32 = 8u;
            const MAX_AGE: u32 = 65535u;

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
                return value >> AGE_SHIFT;
            }

            fn packCell(cellType: u32, age: u32) -> u32 {
                return cellType | (min(age, MAX_AGE) << AGE_SHIFT);
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
                if (below == EMPTY) {
                    return vec2i(0, 1);
                }

                let belowLeft = getCell(pos + vec2i(-1, 1));
                let belowRight = getCell(pos + vec2i(1, 1));
                let left = getCell(pos + vec2i(-1, 0));
                let right = getCell(pos + vec2i(1, 0));

                var canFallLeft = belowLeft == EMPTY;
                var canFallRight = belowRight == EMPTY;

                // Check for priority conflicts - sand directly beside us falling straight down
                if (canFallLeft && left == SAND) {
                    canFallLeft = false;
                }
                if (canFallRight && right == SAND) {
                    canFallRight = false;
                }

                // Randomize direction choice for natural spreading
                let preferLeft = (rng % 2u) == 0u;

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
                let currentAge = getAge(getRawCell(pos));
                let settleRoll = rng % 100u;

                // Settling probability drops off with age:
                // Age 0-10: 80% settle chance (fresh, active)
                // Age 10-30: 40% settle chance (slowing down)
                // Age 30-60: 15% settle chance (mostly settled)
                // Age 60+: 3% settle chance (rare adjustments)
                var maxSettleChance = 80u;
                if (currentAge > 10u) { maxSettleChance = 40u; }
                if (currentAge > 30u) { maxSettleChance = 15u; }
                if (currentAge > 60u) { maxSettleChance = 3u; }

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

            @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let pos = vec2i(id.xy);
                let width = i32(params.y);
                let height = i32(params.z);
                let frame = params.x;

                if (pos.x >= width || pos.y >= height) {
                    return;
                }

                let rawCell = getRawCell(pos);
                var current = getType(rawCell);
                var currentAge = getAge(rawCell);

                // Check for spawn requests - only apply to EMPTY cells (or eraser always works)
                var wasJustSpawned = false;
                let spawnRequest = textureLoad(spawnTex, pos).r;
                if (spawnRequest == 255u) {
                    // Eraser - always clear
                    current = EMPTY;
                    currentAge = 0u;
                } else if (spawnRequest > 0u && current == EMPTY) {
                    // Spawn sand/wall only in empty cells
                    current = spawnRequest;
                    currentAge = 0u;
                    wasJustSpawned = true;
                }

                // Generate randomness for this cell
                let rng = hash(vec2u(id.xy), frame);

                var resultType = current;
                var resultAge = currentAge;

                if (current == WALL) {
                    resultType = WALL;
                    resultAge = 0u;
                }
                else if (current == SAND) {
                    // Freshly spawned sand doesn't move this frame
                    // (neighbors don't know about it yet since they read from inputTex)
                    if (wasJustSpawned) {
                        resultType = SAND;
                        resultAge = 0u;
                    } else {
                        let movement = getSandMovement(pos, rng);
                        if (movement.x != 0 || movement.y != 0) {
                            resultType = EMPTY;
                            resultAge = 0u;
                        } else {
                            resultType = SAND;
                            resultAge = currentAge + 1u; // Age increases when stationary
                        }
                    }
                }
                else if (current == EMPTY) {
                    // Check for incoming sand from various directions
                    var incomingAge = 0u;

                    // From directly above
                    let above = pos + vec2i(0, -1);
                    let aboveRaw = getRawCell(above);
                    if (getType(aboveRaw) == SAND) {
                        let aboveRng = hash(vec2u(u32(above.x), u32(above.y)), frame);
                        let movement = getSandMovement(above, aboveRng);
                        if (above.x + movement.x == pos.x && above.y + movement.y == pos.y) {
                            resultType = SAND;
                            incomingAge = getAge(aboveRaw);
                        }
                    }

                    // From above-left (diagonal)
                    if (resultType == EMPTY) {
                        let aboveLeft = pos + vec2i(-1, -1);
                        let aboveLeftRaw = getRawCell(aboveLeft);
                        if (getType(aboveLeftRaw) == SAND) {
                            let alRng = hash(vec2u(u32(aboveLeft.x), u32(aboveLeft.y)), frame);
                            let movement = getSandMovement(aboveLeft, alRng);
                            if (aboveLeft.x + movement.x == pos.x && aboveLeft.y + movement.y == pos.y) {
                                resultType = SAND;
                                incomingAge = getAge(aboveLeftRaw);
                            }
                        }
                    }

                    // From above-right (diagonal)
                    if (resultType == EMPTY) {
                        let aboveRight = pos + vec2i(1, -1);
                        let aboveRightRaw = getRawCell(aboveRight);
                        if (getType(aboveRightRaw) == SAND) {
                            let arRng = hash(vec2u(u32(aboveRight.x), u32(aboveRight.y)), frame);
                            let movement = getSandMovement(aboveRight, arRng);
                            if (aboveRight.x + movement.x == pos.x && aboveRight.y + movement.y == pos.y) {
                                resultType = SAND;
                                incomingAge = getAge(aboveRightRaw);
                            }
                        }
                    }

                    // From direct left (horizontal settling)
                    if (resultType == EMPTY) {
                        let left = pos + vec2i(-1, 0);
                        let leftRaw = getRawCell(left);
                        if (getType(leftRaw) == SAND) {
                            let lRng = hash(vec2u(u32(left.x), u32(left.y)), frame);
                            let movement = getSandMovement(left, lRng);
                            if (left.x + movement.x == pos.x && left.y + movement.y == pos.y) {
                                resultType = SAND;
                                incomingAge = getAge(leftRaw);
                            }
                        }
                    }

                    // From direct right (horizontal settling)
                    if (resultType == EMPTY) {
                        let right = pos + vec2i(1, 0);
                        let rightRaw = getRawCell(right);
                        if (getType(rightRaw) == SAND) {
                            let rRng = hash(vec2u(u32(right.x), u32(right.y)), frame);
                            let movement = getSandMovement(right, rRng);
                            if (right.x + movement.x == pos.x && right.y + movement.y == pos.y) {
                                resultType = SAND;
                                incomingAge = getAge(rightRaw);
                            }
                        }
                    }

                    // Moving sand resets age to 0 (it's "fresh")
                    if (resultType == SAND) {
                        resultAge = 0u;
                    }
                }

                textureStore(outputTex, pos, vec4u(packCell(resultType, resultAge), 0u, 0u, 0u));
            }
        `
    });

    // Render shader with color options
    const renderShader = device.createShaderModule({
        label: 'Render Shader',
        code: `
            @group(0) @binding(0) var inputTex: texture_2d<u32>;
            @group(0) @binding(1) var<uniform> colorParams: vec4f; // r, g, b, colorByAge

            const TYPE_MASK: u32 = 0xFFu;
            const AGE_SHIFT: u32 = 8u;

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

            @fragment
            fn fs(in: VertexOutput) -> @location(0) vec4f {
                let texSize = vec2f(textureDimensions(inputTex));
                let texCoord = vec2i(in.uv * texSize);
                let rawCell = textureLoad(inputTex, texCoord, 0).r;

                let cellType = rawCell & TYPE_MASK;
                let age = rawCell >> AGE_SHIFT;

                if (cellType == 1u) {
                    // Sand
                    let baseColor = vec3f(colorParams.r, colorParams.g, colorParams.b);

                    if (colorParams.a > 0.5) {
                        // Color by age mode
                        let normalizedAge = min(f32(age) / 500.0, 1.0);

                        // Young sand: bright/saturated, Old sand: darker/more muted
                        // Use a nice gradient from yellow-orange (new) to brown-red (old)
                        let hue = 0.08 - normalizedAge * 0.06; // Orange to red-brown
                        let sat = 0.9 - normalizedAge * 0.3;
                        let val = 1.0 - normalizedAge * 0.5;

                        return vec4f(hsv2rgb(hue, sat, val), 1.0);
                    } else {
                        // Standard color with slight variation based on age for texture
                        let ageVar = f32(age % 20u) / 100.0;
                        return vec4f(baseColor * (1.0 - ageVar * 0.1), 1.0);
                    }
                } else if (cellType == 2u) {
                    // Wall
                    return vec4f(0.45, 0.45, 0.5, 1.0);
                }
                // Empty
                return vec4f(0.08, 0.08, 0.12, 1.0);
            }
        `
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
            ],
        }),
        device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: textures[1].createView() },
                { binding: 1, resource: textures[0].createView() },
                { binding: 2, resource: { buffer: paramsBuffer } },
                { binding: 3, resource: spawnTextureView },
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

    // Material selection
    materialBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            materialBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const mat = btn.dataset.material;
            if (mat === 'sand') selectedMaterial = SAND;
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
                selectMaterial('wall');
                break;
            case '3':
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
        }
    });

    function selectMaterial(mat) {
        materialBtns.forEach(b => {
            b.classList.toggle('active', b.dataset.material === mat);
        });
        if (mat === 'sand') selectedMaterial = SAND;
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
            if ((data[i] & 0xFF) === SAND) count++;
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

        const encoder = device.createCommandEncoder();

        if (!isPaused) {
            for (let step = 0; step < simSpeed; step++) {
                const currentReadIndex = (frame + step) % 2;
                device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([frame + step, WIDTH, HEIGHT, 0]));

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
