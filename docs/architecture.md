# End-to-End System Architecture: FlyLab & PERCEPT-Fly
**A Deep Neuro-Symbolic and Connectome-Constrained Embodied AI Framework**

---

## Abstract

Modern foundation models operate on ungrounded symbolic tokens via programmatic tool-calling or DOM-tree manipulation. **FlyLab** and its complementary sensory module **PERCEPT-Fly (EgoFlyNet)** present an alternative paradigm: **embodied, contact-gated neuro-symbolic interaction**. 

Rather than querying internal data structures, the agent is constrained by a biological sensory-motor bottleneck:
1. It perceives its environment exclusively through rendered optical projections (1200×600 RGB screenshots or compound eye hexagonal ommatidia lattices).
2. It processes visual stimuli through a recurrent *Drosophila* connectome rate-dynamics model (Janelia MaleCNS v1.0).
3. It emits actions via continuous kinematic foreleg motion that must physically collide with rendered keycaps in 3D Euclidean space to authorize native OS-level keyboard events.

```mermaid
flowchart TD
    subgraph Sensory_Environment ["Physical & Virtual Environment"]
        CHR[Isolated Chromium Browser Workspace] -->|Render 1200x600 PNG| PXL[Retinal Pixel Grid]
        POSE[Multi-Fly Pose Tracking Keypoints] -->|World Coordinates| EGO[Egocentric Body Coordinate Frame]
    end

    subgraph Sensory_Transduction ["Sensory Transduction Layer"]
        PXL --> RET[80 Retinal Dwell Windows & Hash Memory]
        EGO --> HEX[Hexagonal Ommatidia Lattice Optics]
        HEX -->|Ommatidial Flux| FLYVIS["Optional Connectome Vision Bridge (flyvis T4/T5)"]
    end

    subgraph Neural_Substrate ["Connectome-Constrained Neural Substrate"]
        RET --> STIM[Sensory Injection Nodes]
        STIM --> RES["MaleCNS Continuous Rate Dynamics (677 / 176,422 Neurons)"]
        RES --> RAST[Poisson/Threshold Spike Raster]
        RES --> READ[Readout Sites / Reservoir State]
    end

    subgraph Policy_Decisions ["Learned Neuro-Symbolic Controller"]
        READ --> MLP[Glyph Readout MLP]
        MLP --> TRANS[Heading Transducer RNN]
        TRANS --> GATE{Edit Decision Gate}
        GATE -->|FOCUS| ACT_F[Aim at Editor Window]
        GATE -->|TYPE| ACT_T[Plan Target Keystroke]
        GATE -->|BACKSPACE| ACT_B[Plan Error Correction]
        GATE -->|RUN| ACT_R[Plan F8 Compile Execution]
    end

    subgraph Kinematic_Embodiment ["Kinematic Body & Contact Boundary"]
        ACT_T & ACT_B & ACT_F & ACT_R --> KIN[Kinematic Foreleg Trajectory Generation]
        KIN --> COL{3D Physical Collision Contact?}
        COL -->|Yes: Travel >= Threshold| TRUST[Trusted Chromium Key Event Dispatch]
        COL -->|No: Below Threshold| BLOCK[Input Withheld: Zero Browser Mutation]
        TRUST --> CHR
    end

    subgraph Provenance ["Clocked Evidence & Telemetry"]
        RES & KIN & CHR --> AUDIT[Cryptographic Audit Log & Replay Archive]
        AUDIT --> THREE[React 19 / Three.js 3D Virtual Workroom]
    end
```

---

## 1. The Closed-Loop Physical Sensory-Motor Cycle

The simulation operates on a discrete canonical clock ($dt = 2\text{ ms}$, $500\text{ Hz}$). Sensory input, neural rate integration, motor displacement, and browser DOM evaluation are lock-stepped:

```mermaid
sequenceDiagram
    autonumber
    participant UI as React / Three.js Client
    participant Engine as Simulation Server (FastAPI)
    participant Chrome as Chromium Browser Core
    participant Brain as MaleCNS Connectome Dynamics
    participant Motor as Kinematic Foreleg Actuator

    Engine->>Chrome: Render workspace frame (1200x600 PNG)
    Chrome-->>Engine: Raw screenshot pixels
    Engine->>Brain: Inject sensory patches into input neurons
    loop Neural Integration (16-128 ticks, dt = 2ms)
        Brain->>Brain: Sparse CSR rate update W*r + stimulus
        Brain->>Brain: Evaluate threshold spike raster
    end
    Brain-->>Engine: Sample readout reservoir state r[t]
    Engine->>Engine: Policy evaluates glyph readout & edit gate
    alt Decision: TYPE or BACKSPACE
        Engine->>Motor: Interpolate 3D trajectory towards target keycap
        Motor->>UI: Stream joint poses for rendering
        UI-->>Engine: Frame rendered acknowledgment (handshake)
        Motor->>Motor: Evaluate physical collision threshold (travel >= 0.018 mm)
        Motor->>Chrome: Dispatch trusted native input event (CDP)
        Chrome->>Chrome: Update textarea DOM & recalculate syntax
    else Decision: RUN
        Motor->>Chrome: Dispatch F8 keydown event
        Chrome->>Chrome: Execute HTML/JS in isolated iframe preview
    end
    Engine->>UI: Broadcast synchronized neural, motor & screen frame
```

### The Hard Trust Boundary
The policy network is subjected to a strict sensory-motor isolation boundary:
* **DOM Blindness:** The policy never queries DOM elements, accessibility trees, ASTs, or tokenized strings.
* **Reward Isolation:** No loss signal, reward metric, or error indicator is passed to the agent during runtime.
* **Input Integrity:** JavaScript `element.value = ...` and synthetic `dispatchEvent` calls are prohibited. Every character requires a physical contact collision event.

---

## 2. Sensory Subsystem: Retinal Transduction & Memory

The visual cortex interface translates the $1200 \times 600$ viewport into localized neural stimulations:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1200 × 600 Browser Viewport                     │
├──────────────────────┬───────────────────────┬─────────────────────────┤
│    Section 1: TARGET │   Section 2: EDITOR   │   Section 3: PREVIEW    │
│    (400 × 540 px)    │   (400 × 540 px)      │   (400 × 540 px)        │
│                      │                       │                         │
│  [24 Retinal Slots]  │  [24 Retinal Slots]   │  [24 Retinal Slots]     │
│   (24 × 24 px each)  │   (24 × 24 px each)   │   (24 × 24 px each)     │
└──────────────────────┴───────────────────────┴─────────────────────────┘
```

### 2.1 Fixed Retinal Sampling Slots
1. **Target Substrate:** 24 horizontal retinal slots positioned over the target specification heading.
2. **Editor Substrate:** 24 retinal slots capturing typed code inside the editor textarea.
3. **Preview Substrate:** 24 retinal slots capturing the live browser execution output.
4. **State Indicators:** 8 dedicated slots monitoring cursor caret focus, execution run state, and compile status.

### 2.2 Saccadic Visual Memory & Differential Refresh
To prevent redundant computation across static frames, each retinal patch $\mathbf{P}_i \in \mathbb{R}^{24 \times 24}$ is hashed:
$$h_i = \text{SHA256}(\mathbf{P}_i)$$
* If $h_i(t) == h_i(t - 1)$, the visual reading is drawn from short-term retinal cache.
* If $h_i(t) \neq h_i(t - 1)$, the new patch is injected into the sensory nodes, advancing simulation by $\Delta t_{\text{saccade}} = 16\text{ ms}$ to allow transient dynamics to settle before decoding.

---

## 3. Neural Substrate: Connectome-Constrained Rate Dynamics

FlyLab embeds the visual policy inside a connectome reconstruction derived from the Janelia **MaleCNS v1.0** dataset:

### 3.1 Anatomical Topologies
* **Full-CNS Network:** 176,422 identified neuron records and 25,862,574 directed synaptic connections.
* **Subgraph Model:** 677 structurally conserved neurons spanning the optical lobes, central complex (protocerebral bridge, ellipsoid body, fan-shaped body), and descending motor pathways.

### 3.2 Continuous Rate Formulation
The recurrent state $\mathbf{r}(t) \in \mathbb{R}^N_{\ge 0}$ evolves according to continuous leaky rate dynamics:
$$\tau \frac{d\mathbf{r}}{dt} = -\mathbf{r}(t) + \operatorname{ReLU}\left(\mathbf{W}_{\text{norm}} \mathbf{r}(t) + \mathbf{W}_{\text{in}} \mathbf{s}(t) + \mathbf{b}\right)$$

Where:
* $\tau = 20\text{ ms}$ is the biological membrane integration time constant.
* $\mathbf{W}_{\text{norm}} = \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}$ is the normalized connectome adjacency matrix with strictly positive synapse counts.
* $\mathbf{s}(t) \in \mathbb{R}^{101}$ is the sensory injection vector.
* $\mathbf{b} \in \mathbb{R}^N$ is the baseline resting potential vector.

### 3.3 Threshold Spike Generation
While integration is continuous, binary event streams are extracted via a dynamic threshold model for neuromorphic visualization and instrumentation:
$$S_j(t) = \mathbb{I}\left(r_j(t) - \theta_j(t) > 0\right)$$
$$\theta_j(t + dt) = \theta_j(t) + \frac{dt}{\tau_\theta} (\theta_{\text{rest}} - \theta_j(t)) + \Delta_\theta S_j(t)$$

### 3.4 Numerical Kernels
* **Native C Kernel (`csr_parallel.c`):** Utilizes OpenMP multi-threading with 8-way SIMD FMA (Fused Multiply-Add) vectorization over sparse CSR storage formats.
* **SciPy Fallback:** Vectorized sparse matrix-vector multiplication (`scipy.sparse.csr_matrix`).

---

## 4. Learned Controller & Self-Correction

The high-level policy consists of trained feed-forward readouts from the connectome reservoir:

```mermaid
graph LR
    subgraph Reservoir
        R["Connectome State r(t)"]
    end
    subgraph Readouts
        R -->|Sample 576 Nodes| GLYPH["Glyph Readout MLP"]
        GLYPH --> DEC_C["Character Decoders (A-Z, <, >, /, etc.)"]
        DEC_C --> TRANSDUCER["Heading Transducer RNN"]
        TRANSDUCER --> GATE["Edit Decision Gate"]
    end
    subgraph Actions
        GATE --> A1["FOCUS"]
        GATE --> A2["TYPE(c)"]
        GATE --> A3["BACKSPACE"]
        GATE --> A4["RUN"]
        GATE --> A5["FINISH"]
    end
```

### Closed-Loop Visual Self-Repair
FlyLab tests visual feedback by intentionally injecting an unannounced physical key substitution fault at step $i = 10$ (e.g., typing `X` instead of `W`). 
Because the policy relies strictly on rendered pixels:
1. The erroneous glyph appears in the Chromium screenshot at step $i = 11$.
2. The retinal slots over the editor observe a mismatch with the target heading specification.
3. The edit gate emits `BACKSPACE` decisions until the erroneous characters are erased, followed by the correct stroke.

---

## 5. PERCEPT-Fly (EgoFlyNet) Visual Reconstruction Subsystem

While FlyLab models interaction with digital interfaces, **PERCEPT-Fly (EgoFlyNet)** provides an egocentric sensory model for behavioral tracking and visual neuroscience experiments.

### 5.1 Kinematic Body Reference Frame
Given tracked 3D coordinates for head $\mathbf{p}_{\text{head}}$ and thorax $\mathbf{p}_{\text{thorax}}$:

1. **Forward Axis:**
   $$\mathbf{u}_{\text{fwd}} = \frac{\mathbf{p}_{\text{head}} - \mathbf{p}_{\text{thorax}}}{\|\mathbf{p}_{\text{head}} - \mathbf{p}_{\text{thorax}}\|}$$
2. **Right Axis:**
   $$\mathbf{u}_{\text{right}} = \frac{\mathbf{u}_{\text{world\_up}} \times \mathbf{u}_{\text{fwd}}}{\|\mathbf{u}_{\text{world\_up}} \times \mathbf{u}_{\text{fwd}}\|}$$
3. **Dorsal Axis:**
   $$\mathbf{u}_{\text{up}} = \mathbf{u}_{\text{fwd}} \times \mathbf{u}_{\text{right}}$$

The rotation matrix $\mathbf{R} = [\mathbf{u}_{\text{fwd}} \;\; \mathbf{u}_{\text{right}} \;\; \mathbf{u}_{\text{up}}]$ maps any external point $\mathbf{p}_w$ into egocentric coordinates:
$$\mathbf{p}_{\text{ego}} = \mathbf{R}^T (\mathbf{p}_w - \mathbf{p}_{\text{head}})$$

### 5.2 Hexagonal Compound Eye Lattice
The compound eye is parameterized into $N \approx 720$ discrete ommatidia arranged on alternating hexagonal rows:

$$\phi_{jk} = \phi_{\text{start}} + k \cdot \Delta\phi + (j \bmod 2) \frac{\Delta\phi}{2}$$

Each ommatidial optical axis $\mathbf{a}_m$ is defined on the unit sphere:
$$\mathbf{a}_m = \begin{bmatrix} \cos\theta_m \cos\phi_m \\ \cos\theta_m \sin\phi_m \\ \sin\theta_m \end{bmatrix}$$

```
                Left Eye (Azimuth < 0)         Right Eye (Azimuth > 0)
Elevation (+)        ·   ·   ·   ·   ·                 ·   ·   ·   ·   ·
                      ·   ·   ·   ·   ·                 ·   ·   ·   ·   ·
Elevation (0)   ·   ·   ·   ·   ·   · [Overlap] ·   ·   ·   ·   ·   ·   ·
                      ·   ·   ·   ·   ·                 ·   ·   ·   ·   ·
Elevation (-)        ·   ·   ·   ·   ·                 ·   ·   ·   ·   ·
               -160°            -20°   0°   +20°            +160°
```

### 5.3 Geometric Optics & Receptive Fields
For every partner limb element $i$ at position $\mathbf{p}_{\text{ego}, i}$, distance $d_i = \|\mathbf{p}_{\text{ego}, i}\|$ and unit line-of-sight $\mathbf{u}_i = \mathbf{p}_{\text{ego}, i} / d_i$:
1. **Angular Divergence:** $\alpha_{im} = \arccos(\operatorname{clip}(\mathbf{a}_m \cdot \mathbf{u}_i, -1, 1))$
2. **Gaussian Acceptance Profile:** $W_{im} = \exp\left(-\frac{\alpha_{im}^2}{2\sigma^2}\right)$, with $\sigma \approx 5.0^\circ$.
3. **Solid-Angle Inverse-Square Falloff:**
   $$I_m(t) = \sum_{i} \frac{W_{im}(t)}{\max(d_i(t), d_{\min})^2}$$

### 5.4 Connectome Vision Bridge (`flyvis`)
The resulting spatiotemporal array $\mathbf{I} \in \mathbb{R}^{T \times N}$ can optionally pass to the pretrained `flyvis` connectome network (Lappalainen et al., Nature 2024), yielding population activity traces for motion-sensitive neural subtypes:
* **T4 Cells (Medulla $\to$ Lobula Plate):** Four directional subtypes (T4a, T4b, T4c, T4d) sensitive to front-to-back, back-to-front, upward, and downward ON-motion.
* **T5 Cells (Lobula $\to$ Lobula Plate):** Equivalent directional selectivity for OFF-contrast edges.

---

## 6. End-to-End Scientific Architecture Comparison

| Dimension | FlyLab Physical Coding | PERCEPT-Fly (EgoFlyNet) |
|---|---|---|
| **Primary Domain** | Contact-gated physical coding & tool-use | Multi-fly behavioral tracking & egocentric vision |
| **Input Modality** | Rendered 1200×600 Chromium browser PNG | Tracked 3D pose keypoints (SLEAP / DeepLabCut) |
| **Visual Geometry** | 80 Planar Retinal Sampling Windows | 720 Hexagonal Compound Eye Ommatidia Lattice |
| **Neural Substrate** | Full MaleCNS (176k) / Subgraph (677) rate model | Optional `flyvis` Lobula Plate Motion Model (T4/T5) |
| **Actuation** | Kinematic foreleg with contact collision detection | Kinematic focal-fly body frame (orthonormal basis) |
| **Downstream Mutation**| Native Chromium OS-level keyboard events | Visual excitation tracks & neural motion traces |
| **Trust Model** | DOM-blind, AST-blind, reward-blind | Calibration-blind, point-mass geometric projection |

---

## 7. Limitations and Epistemic Honesty Disclosures

1. **Dynamics Are Modeled, Not Reconstructed Electrophysiology:**
   The neural substrate uses normalized, non-negative continuous rate equations. It does not model individual ion channels, dendritic conductance compartmentalization, or biophysical neurotransmitter dynamics.
2. **Simplified Biomechanics:**
   Fly foreleg motion is governed by continuous forward-kinematic point-attractors rather than a full multi-segment musculo-skeletal physics engine.
3. **Curriculum Scope:**
   The neural controller is trained specifically on single-line HTML heading synthesis (`<h1>...</h1>`). It is an experimental demonstration of grounded physical keyboard contact, **not** an artificial general intelligence or a general coding agent.
