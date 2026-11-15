# Methods and Scientific References

## 1. Mathematical Formulation

### 1.1 Focal-Fly Body Reference Frame
Let $\mathbf{p}_{\text{head}}(t), \mathbf{p}_{\text{thorax}}(t) \in \mathbb{R}^3$ denote the tracked 3D coordinates of the focal fly's head and thorax at time step $t$.

1. **Forward Axis (Anteroposterior):**
   $$\mathbf{u}_{\text{fwd}}(t) = \frac{\mathbf{p}_{\text{head}}(t) - \mathbf{p}_{\text{thorax}}(t)}{\|\mathbf{p}_{\text{head}}(t) - \mathbf{p}_{\text{thorax}}(t)\|}$$

2. **Right Axis (Mediolateral):**
   With the global dorsal vertical reference $\mathbf{u}_{\text{up,ref}} = [0, 0, 1]^T$:
   $$\mathbf{u}_{\text{right}}(t) = \frac{\mathbf{u}_{\text{up,ref}} \times \mathbf{u}_{\text{fwd}}(t)}{\|\mathbf{u}_{\text{up,ref}} \times \mathbf{u}_{\text{fwd}}(t)\|}$$

3. **Dorsal Axis (Dorsoventral):**
   $$\mathbf{u}_{\text{up}}(t) = \mathbf{u}_{\text{fwd}}(t) \times \mathbf{u}_{\text{right}}(t)$$

The rotation matrix $R(t) = [\mathbf{u}_{\text{fwd}}(t) \;\; \mathbf{u}_{\text{right}}(t) \;\; \mathbf{u}_{\text{up}}(t)]$ forms an orthonormal basis. Any world point $\mathbf{p}_w$ is transformed into egocentric head coordinates via:
$$\mathbf{p}_{\text{ego}}(t) = R(t)^T \left(\mathbf{p}_w - \mathbf{p}_{\text{head}}(t)\right)$$

---

### 1.2 Hexagonal Compound Eye Lattice
The compound eye is modeled as a set of discrete optical units (ommatidia). Each ommatidium $j$ has an optical axis unit vector $\mathbf{a}_j = [\cos\theta_j \cos\phi_j, \; \cos\theta_j \sin\phi_j, \; \sin\theta_j]^T$, where $\phi_j$ is azimuth and $\theta_j$ is elevation in head coordinates.

Hexagonal close-packing is achieved by staggering alternating elevation rows by half the interommatidial azimuthal spacing:
$$\phi_{j, \text{row } r} = \phi_{\text{start}} + k \cdot \Delta\phi + (r \bmod 2) \cdot \frac{\Delta\phi}{2}$$

---

### 1.3 Receptive Field and Egocentric Projection
For each sample point $i$ on the partner fly with egocentric vector $\mathbf{p}_{\text{ego}, i}$, distance $d_i = \|\mathbf{p}_{\text{ego}, i}\|$ and unit line-of-sight $\mathbf{u}_i = \mathbf{p}_{\text{ego}, i} / d_i$:

1. The angular separation with ommatidium $j$ is:
   $$\alpha_{ij} = \arccos(\operatorname{clip}(\mathbf{a}_j \cdot \mathbf{u}_i, -1, 1))$$

2. The angular acceptance profile is governed by a Gaussian function:
   $$W_{ij} = \exp\left(-\frac{\alpha_{ij}^2}{2\sigma^2}\right)$$
   where $\sigma \approx 5.0^\circ$ corresponds to the biological half-width of the Drosophila acceptance angle.

3. The distance attenuation scales with the solid angle subtended by the target element:
   $$I_j(t) = \sum_{i} \frac{W_{ij}(t)}{\max(d_i(t), d_{\min})^2}$$

---

## 2. References & Software Licenses

1. **flyvis**
   - **Citation:** Lappalainen, J. K., Tschopp, F., Grover, D., et al. (2024). *Connectome-constrained networks predict neural activity across the Drosophila visual system*. Nature, 634, 1132–1140.
   - **Repository:** [https://github.com/turagalab/flyvis](https://github.com/turagalab/flyvis)
   - **License:** MIT License.

2. **FlyBody**
   - **Citation:** Vaxenburg, R., et al. (2024). *FlyBody: Realistic biomimetic models of Drosophila melanogaster*.
   - **License:** Apache License 2.0.

3. **SLEAP**
   - **Citation:** Pereira, T. D., et al. (2022). *SLEAP: A deep learning system for multi-animal pose tracking*. Nature Methods, 19, 486–495.
   - **Repository:** [https://github.com/talmo/sleap](https://github.com/talmo/sleap)
   - **License:** BSD-3-Clause License.

---

## 3. Explicit Modeling Assumptions & Limitations
- **Geometric Simplification:** The eye curvature is modeled as spherical sectors rather than the true asymmetrical prolate eye shape of adult Drosophila.
- **Acceptance Uniformity:** Biological ommatidia exhibit non-uniform spatial resolution (fovea-like high acuity in the anterior-dorsal region); here, acceptance angles and spacing are approximated as piecewise uniform.
- **Occlusion:** Complex self-occlusion (e.g. wings occluding the abdomen from certain vantage points) is approximated via volumetric point sampling without full ray-traced mesh shadow testing.
- **Scientific Claim:** This software reconstructs modeled visual excitation for computational analysis and connectome simulation. It is **not** a direct empirical electrophysiological measurement or proof of conscious perception.
