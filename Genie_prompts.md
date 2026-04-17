# Genie Prompts — for Project Genie (Google AI Ultra)

Prompts to prototype the PAT-Scan instrumentation idea: handheld force/displacement sensor pressed against a deformable hemisphere.

## How Genie prompts work

Two parts:
1. **Environment** — describe the world (weather, objects, atmosphere, feel)
2. **Character** — what you are, and how you move

You can also upload an image. Preview uses Nano Banana Pro. Choose first-person or third-person view.

---

## Prompt 1 — Baseline hemisphere + handheld sensor (v2, tuned after Prompt 5 test)

> **Notes from Prompt 5 test (2026-04-17):**
> - Genie over-softens deformable objects — a "soft" ball behaved like a Swiss ball, not a rubber ball. To get tissue-like stiffness, say "firm", "barely deforms", "subtle local indentation only".
> - First ~10–15 seconds are unstable: objects can ghost through each other and duplicate. Wait for stabilization before judging.
> - Readout dials won't update with real numbers — Genie generates visuals, not physics. Treat any readout as aesthetic decoration, not live data.

**Environment:**
A calm, warm, sunlit kitchen-counter scene. On a smooth wooden countertop rests a LARGE FIRM silicone hemisphere — roughly the size of a volleyball, about 30 centimeters across, half-embedded in the wood so only the rounded dome rises up. Slightly translucent, pale flesh-coloured. The hemisphere is stiff — it barely deforms under light pressure, showing only a subtle, shallow local dimple right at the contact point, no more than a few millimeters deep. Everywhere else it holds its shape. The atmosphere is calm, domestic, warm — like a home kitchen mid-morning, not a laboratory, not a game arcade, not a showroom. Absolutely NO game tables, NO arenas, NO hockey setups, NO medical equipment, NO clinical instruments, NO display plinths, NO mall showrooms.

**Character:**
The character is literally a smartphone — a modern iPhone-like device, matte white, about 15 centimeters long. Just an ordinary smartphone. Held gently in mid-air at human-hand height as if by invisible fingers pinching it by the sides. The phone's FLAT BACK FACE points downward toward the hemisphere. There are NO protruding tips, NO antennas, NO probes, NO rounded bulges — absolutely just a smartphone. To "sample" the hemisphere, the phone is lowered slowly and deliberately and its FLAT BACK FACE is pressed lightly against the top of the dome. Motion is SLOW AND DELIBERATE — like a chef plating delicate food, or a jeweler inspecting a gem — not like a hockey puck, not like a video game, not bouncing, not sliding, not jumping. Every movement is gentle and controlled. When the phone's back face touches the hemispherical dome, the dome surface indents only slightly right where the phone sits — a subtle flattened patch a few millimeters deep — and springs back immediately when the phone lifts. No sliding, no bouncing, no arena dynamics. This is a CAREFUL SAMPLING gesture, not a GAME.

---

## Prompt 2 — Full displacement field visualisation

**Environment:**
A dark research lab with ambient blue lighting. On a black optical breadboard sits a soft silicone hemisphere, 10 cm across, covered in a fine white speckle pattern. Two cameras mounted on stands flank the hemisphere, pointed at it from slightly above. A floating heatmap overlay shows surface displacement — warm colours where the surface has moved inward, cool colours where it's unchanged.

**Character:**
A handheld force-torque sensor block with a polished spherical indenter tip. When it presses into the hemisphere, the heatmap updates in real time: bright red at the contact point, fading through orange and yellow into cool blue at the edges. A small text readout above the sensor shows the three components of force: normal (N), tangential X (N), tangential Y (N).

---

## Prompt 3 — Comparing sensor types side-by-side

**Environment:**
A bright tactile-sensing comparison lab. Three identical soft silicone hemispheres sit in a row on a long white table, each labelled. From left to right: "Bota F/T", "GelSight", "Tekscan FlexiForce". Behind each hemisphere is a readout panel styled appropriately — the Bota panel shows a 6-DOF wrench, the GelSight panel shows a pixel field of normal + shear pressure over the contact patch, the FlexiForce panel shows a single scalar pressure gauge. Calm, educational atmosphere — like a physics museum exhibit.

**Character:**
An invisible hand that holds each sensor in turn and presses them against their respective hemispheres. The relevant panel lights up when its sensor is in contact. The motion is slow and deliberate — press, hold for 2 seconds so the readings settle, release, move to the next sensor.

---

## Prompt 4 — Inverse-problem framing (PAT-Scan flavoured)

**Environment:**
A computational imaging lab with a soft silicone hemisphere on a glass platform. The hemisphere has a hidden stiffer inclusion inside it (like a tumour phantom). A translucent "X-ray style" overlay shows the inclusion in faint red, but only as a hint — the user is trying to find it by palpation, not by looking. The walls of the room display a slowly-rotating 3D reconstruction of the inferred elasticity map, updated as new measurements arrive.

**Character:**
A handheld palpation sensor that returns force + local displacement when pressed. Each press adds a data point to the 3D reconstruction on the wall, and the inclusion's hidden shape slowly sharpens. Movement is point-by-point — pick a spot, press, read, move, press, read.

---

## Prompt 5 — Minimal sanity-check scene (if Genie gets confused)

**Environment:**
A plain white room. A pink rubber ball sits on the floor. The lighting is soft and even.

**Character:**
A small glowing cube, the size of an apple, that can be pushed in any direction. When it touches the ball, the ball deforms briefly and springs back.

Use this if the more complex prompts fail — it's the simplest possible "soft object + movable presser" scenario.

---

## Tips

- Start with Prompt 5 to verify Genie is working, then move to Prompt 1.
- Keep environments specific about **materials** ("silicone", "glass", "rubber") — Genie's physics seems more accurate with material cues.
- Keep characters specific about **how they move** ("floats at human-hand height", "pressed into surface", "slow and deliberate") — vague motion = weird physics.
- If the readout panels don't appear, drop them from the prompt; Genie is better at physical objects than floating UI.
