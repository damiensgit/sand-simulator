# Hybrid Grid + Water FLIP Specification

This project uses a grid-authoritative world model with a specialized water solver.
All materials interact through shared grid fields. Water uses particles internally
but writes back to the grid every step so other materials can stay grid-based.

## Grid Fields (Authoritative)

- material (u32): EMPTY, WALL, SAND, WATER, LAVA, STEAM, etc.
- solidMask (bool): WALL, ROCK, SAND (blocks fluid flow).
- density (f32): water density per cell (0.0..1.0+).
- velocity u/v (MAC grid): u on vertical faces, v on horizontal faces.
- pressure (f32): fluid pressure for incompressibility.
- temperature (f32): heat for phase changes and buoyancy.
- optional: fuel, moisture, burning, blast (if TNT/fire is added).

## Water (Particles + Grid)

Particles exist only for water. They are internal and never replace the grid.

Particle fields:
- position (vec2)
- velocity (vec2)
- mass (f32)

Step order:
1) Integrate particles with gravity and external forces.
2) Deposit particle mass and velocity onto the grid (density + u/v).
3) Solve pressure on the grid (fluid cells only, air p=0, solids block flow).
4) Transfer grid velocity back to particles using PIC/FLIP blending.
5) Resolve particle collisions with solidMask and boundaries.

PIC/FLIP blend:
- v_pic = sampled grid velocity
- v_flip = v_old + (grid_new - grid_old)
- v = mix(v_pic, v_flip, flipRatio), typically 0.9..0.98

## Other Materials (Grid Automata)

Sand, lava, steam, fire, etc. remain cell-based and use grid fields:
- Water sees sand as solid via solidMask.
- Sand sees water via density threshold, not render threshold.
- Shockwaves and explosions inject velocity/pressure into grid fields.

## Thresholds

Two thresholds control water visibility vs. interaction:
- renderThreshold: e.g. 0.5, only render if density >= this.
- interactThreshold: e.g. 0.05, treat as "wet" if density >= this.

## Coupling Rules

- Water uses solidMask for boundaries (WALL/ROCK/SAND).
- Sand can be influenced by fluid via grid velocity/pressure impulses.
- Heat and phase changes are grid-driven (temperature affects material changes).

## Update Order (Per Frame)

1) Apply brush edits to material.
2) Water FLIP step (particles <-> grid).
3) Automata step for sand/lava/fire using grid fields.
4) Coupling passes (phase changes, heat exchange, shock impulses).
5) Render (water from density, solids from material).

## Notes

- Water density is continuous internally; rendering can stay discrete via threshold.
- Grid remains the source of truth for all non-water materials.
