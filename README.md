# Sand Simulator

A fun falling sand game I built with my kids (using Claude) with WebGPU compute shaders to demonstrate **cellular automata** - where simple rules create complex, natural-looking behavior.

**Try it live:** https://damiensgit.github.io/sand-simulator/

## What is a Cellular Automaton?

Imagine a grid of tiny squares (cells). Each cell follows simple rules based on its neighbors:

- **Sand falls down** if there's empty space below
- **Sand slides diagonally** if it can't fall straight down
- **Water spreads out** and flows to find the lowest point
- **Lava glows** and turns to rock when it cools
- **Steam rises** and eventually condenses back to water

Even though each cell only knows about itself and its neighbors, together they create realistic physics!

## How to Run

1. Open `index.html` in Chrome or Edge (requires WebGPU support)
2. Click and drag to draw sand
3. Watch it fall and pile up!

## Controls

| Key | Action |
|-----|--------|
| **1** | Select Sand |
| **2** | Select Water |
| **3** | Select Lava |
| **4** | Select Wall |
| **5** | Select Eraser |
| **Space** | Pause/Resume |
| **C** | Clear everything |
| **[ ]** | Decrease/Increase brush size |
| **- +** | Decrease/Increase fluidity |
| **g/G** | Decrease/Increase gravity |
| **t/T** | Decrease/Increase temperature (±50°C) |
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
4. **Pour water on sand** - watch the sand sink through the water!
5. **Mix lava and water** - creates steam and rock
6. **Lower gravity** - watch materials float and fall slowly
7. **Raise temperature** - water evaporates into steam, rock melts into lava
8. **Lower temperature** - water freezes into ice, lava solidifies into rock
9. **Turn on "Color by age"** - watch how fresh particles (bright) settle into old ones (dark)

## Materials

| Material | Behavior |
|----------|----------|
| **Sand** | Falls, piles up, sinks in water |
| **Water** | Flows, spreads out, evaporates when hot |
| **Lava** | Flows slowly, glows, turns to rock when cool |
| **Wall** | Solid, doesn't move |
| **Steam** | Rises up, condenses back to water |
| **Ice** | Frozen water, melts when warm |
| **Rock** | Cooled lava, melts at extreme heat |

## Files

- `index.html` - The webpage with controls
- `main.js` - WebGPU setup and shader code

---

*A fun project to explore cellular automata and GPU computing!*
