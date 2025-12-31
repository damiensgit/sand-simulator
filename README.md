# Sand Simulator

A fun falling sand game built I build with my kids (using claude) with WebGPU compute shaders to demonstrate **cellular automata** - where simple rules create complex, natural-looking behavior.

## What is a Cellular Automaton?

Imagine a grid of tiny squares (cells). Each cell follows simple rules based on its neighbors:

- **Sand falls down** if there's empty space below
- **Sand slides diagonally** if it can't fall straight down
- **Sand settles sideways** to form natural-looking piles

Even though each cell only knows about itself and its neighbors, together they create realistic sand physics!

## How to Run

1. Open `index.html` in Chrome or Edge (requires WebGPU support)
2. Click and drag to draw sand
3. Watch it fall and pile up!

## Controls

| Key | Action |
|-----|--------|
| **1** | Select Sand |
| **2** | Select Wall |
| **3** | Select Eraser |
| **Space** | Pause/Resume |
| **C** | Clear everything |
| **[ ]** | Decrease/Increase brush size |
| **- +** | Decrease/Increase fluidity |
| **A** | Toggle age-based coloring |

## The Cool Tech Stuff

This runs on your **GPU** (graphics card) using compute shaders. That means thousands of sand particles are simulated in parallel - each cell calculates its next state at the same time as every other cell!

### The "Gather" Approach

Each cell asks: "Should sand move INTO me?" rather than "Should I move somewhere else?"

This avoids conflicts when multiple sand particles might want to move to the same spot.

## Try These Experiments

1. **Build a funnel** with walls and pour sand through it
2. **Set fluidity to 0** - sand becomes sticky and forms tall columns
3. **Set fluidity to 100** - sand flows like water
4. **Turn on "Color by age"** - watch how fresh sand (bright) settles into old sand (dark)

## Files

- `index.html` - The webpage with controls
- `main.js` - WebGPU setup and shader code

---

*A fun project to explore cellular automata and GPU computing!*
