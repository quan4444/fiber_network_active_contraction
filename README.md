# Fiber Network Active Contraction in FEniCSx

The goal of this repository is to model the contraction phase during the wound healing process for our microtissue wound healing experiments. In this current version, we have implemented and validated the morphoelastic rod on a single beam. The next steps for this repository include: implementing the Robin boundary condition, validating the model on multiple intersecting beams, and modeling the contraction of a fiber network representing the microtissue contraction phase.


## Table of Contents

* [Installation Instructions](#install)
* [Morphoelastic Rod Theory](#morph)
* [Validating a Single Beam Active Contraction](#single)
* [Validating a Constrained Single Beam Active Contraction](#constrained)
* [Skills used from class](#skills)


## Installation Instructions <a name="install"></a>

To use the codes in this tutorial, please begin by setting up FEniCSx:
```bash
mamba create --name fenicsx-env
mamba activate fenicsx-env
mamba install -c conda-forge fenics-dolfinx openmpi pyvista
```

When running the code for the first time, VSCode will prompt you to install ipykernel, necessary to run Jupyter Notebook.


## Morphoelastic Rod Theory <a name="morph"></a>

### Short introduction to shear deformable geometrically exact beams
This shortened version was written by Jeffrey Prachaseree, who is also the author of a more in-depth [geometrically exact rod theory.](https://fenics-arclength.readthedocs.io/en/latest/examples/force_control/beam/README.html) 

The code here uses shear deformable geometrically exact beams as the beam constitutive model. For simplicity, the implemented code and equations below uses the 2D formulation so rotation parameterization is not needed.

#### Kinematics
In brief, the kinematics of geometrically exact beams can be described through a beam centerline $s$ and material triads $g_{0i}$. At time $t \neq 0$ the position of the beam centerline can be described as:

```math
\mathbf r(s) = \mathbf r_0(s) + \mathbf u(s)
```
where $\mathbf{r}(s)$ is the current position, $\mathbf{r}_0$ is the initial position and $\mathbf{u}(s)$ is the displacement vector.

and the beam triads can be described as:

```math
\mathbf g_i(s) = \mathbf \Lambda \mathbf g_{0i} = \mathbf{\Lambda \Lambda}_0 \mathbf e_i
```
where $\Lambda$ and $\Lambda_0$ are the current and initial rotation tensors respectively, and $e_i$ are the initial global basis. More information on constructing the rotation tensors can be found in the detailed derivation of geometrically exact beams. 

The strain measures are:

```math
\mathbf{\epsilon} = \mathbf{\Lambda}_0^\top \mathbf{\Lambda}^\top \frac{d\mathbf{r}(s)}{d s} - \mathbf{e}_1
```

where $\epsilon$ is the translational strain and $\frac{d }{d s}$ is the directional derivative with respect to the beam centerline.

The rotational strains can be expressed as:

```math
\mathbf{\chi} = \text{axial}(\mathbf{\Lambda}^\top \mathbf{\Lambda}_{,s} - \mathbf{\Lambda}^\top \mathbf{\Lambda}_{0,s})
```

#### Constitutive Model
The constitutive model used in this code is a linear elastic shear deformable beam. As such, the internal hyperelastic strain energy density of the beam is:

```math
W = \frac{1}{2}( \mathbf{\epsilon} \cdot \mathbf{C}_N \mathbf{\epsilon} + EI \chi^2)
```
where:

```math
\mathbf{C}_N = \begin{bmatrix}
EA & 0 \\
0 & \mu A^*
\end{bmatrix}
```
where $E$ is the Young's modulus, $\mu$ is the shear modulus and $A^*$ is the effective shear area.

The force can be computed by:
```math
\mathbf{n} = \frac{\partial W}{\partial \mathbf{\epsilon}}
```
```math
\mathbf{m} = \frac{\partial W}{\partial \chi}
```

### Morphoelastic Beams
The main theory for morphoelastic rods can be found in this paper: [Morphoelastic rods. Part I: A single growing elastic rod](http://goriely.com/wp-content/uploads/2012-JMPSmorphorods-1.pdf). 

### Strong Form
In brief, this theory is the extension of the the multiplicative decomposition of the deformation gradient for 3D growth (i.e. $\mathbf{F}= \mathbf{F}^e\mathbf{F}^g$). Following growth theory through multiplicative decompostion, three configurations are defined: the unstressed initial configuration $\mathcal{B}_0$, the virtual reference configuration (i.e. unstressed but with growth) $\mathcal{B}_v$, and the current configuration $\mathcal{B}$. Different from the three-dimentional growth case, the virtual reference configuration has no local compatibility issues, so the material law can be expressed in the virtual configuration in a well-defined manner. However, the virtual configuration might be able to be represented in Euclidean space (i.e. it can intersect itself). Following Moulton et. al's description, the central arclength of the beam is defined as $S_0$, $S$ and $s$ in the inital, reference, and current configuration respectively. The growth stretch is defined as $\gamma = \frac{\partial S}{\partial S_0}$. Additionally, the mechanical stretch is defined as $\alpha = \frac{\partial s}{\partial S}$, and the total stretch is defined as $\lambda = \frac{\partial s}{\partial S_0}$. As such, to map the initial configuration to the current configuration we can use the total stretch $\lambda = \alpha \gamma \leftrightarrow \frac{\partial s}{\partial S_0} = \frac{\partial s}{\partial S} \frac{\partial S}{\partial S_0}$. Therefore, the balance equations in the virtual reference configuration are:

```math
\frac{\partial \mathbf{n}}{ \partial S} + \mathbf{f} = 0,
```

```math 
\frac{\partial m}{\partial S} + (\frac{\partial \mathbf{r}}{\partial S} \times \mathbf{n}) \cdot e_3 + l = 0,
```
where $f$ and $l$ are the body force and moment couple per length in the virtual reference configuration.

Using change of variables, the pushforward operation on the balance equations to the current configuration yields:

```math
\frac{\partial \mathbf{n}}{ \partial s} + \alpha^{-1}\mathbf{f} = 0,
```

```math 
\frac{\partial m}{\partial s} + (\frac{\partial \mathbf{r}}{\partial s} \times \mathbf{n}) \cdot e_3 + \alpha^{-1} l = 0.
```

Similarly, using change of variables, the pullback operation to the initial configuration is expressed by:

```math
\frac{\partial \mathbf{n}}{ \partial S_0} + \gamma\mathbf{f} = 0,
```

```math 
\frac{\partial m}{\partial S_0} + (\frac{\partial \mathbf{r}}{\partial S_0} \times \mathbf{n}) \cdot e_3 + \gamma l = 0.
```

### Weak form
Again, we emphasize that the material law is formulated in the *virtual reference configuration* and only depends on the mechanical stretch  $W = W(\alpha, \chi) = W(\lambda \gamma^{-1}, \chi)$. 

As such, strain energy in the virtual reference configuration is:

```math
\Pi_{int} = \int_{B_v} W(\alpha,\chi) \; dS.
```

However, the function spaces defined in FEniCS are in the initial configuration. We use the change of variables (i.e. $\gamma = \frac{d S}{d S_0} \rightarrow dS = \gamma dS_0$). Therefore, the strain energy in the initial configuration is:


```math
\Pi_{int} = \int_{B_0} W(\alpha,\chi) \; \gamma \; dS_0.
```

Now that the material law and the function spaces are both defined in the initial configuration, it is possible to take the Gateaux derivative to get the variational form. 

From the theory of geometrically exact beams, the external loads in the initial configuration is:
```math
\delta \Pi_{ext} = \int_L (\mathbf{F} \cdot \delta \mathbf{u} + \mathbf{M} \cdot \mathbf{H}\delta\mathbf{\theta}) \; ds + \sum \mathbf{f}\delta \mathbf{u} + \sum \mathbf{m}\delta \mathbf{\theta}
```

Finally, the equilibrium solution is obtained by finding the stationary point in the total potential energy:

```math
\delta \Pi_{int} - \delta \Pi_{ext} = 0.
```



### Analytical Solution of constrained rod

The validation problem we are comparing the FEA solutions to is a shrinking rod confined on both ends. This means that the total stretch $\lambda = 1$. As such, the mechanical stretch $\alpha = \lambda \gamma^{-1} = \gamma^{-1}$ From the contitutive model, the axial force is:

```math
n_3= EA(\alpha-1) = EA(\gamma^{-1} - 1).
```


## Validating a Single Beam Active Contraction <a name="single"></a>

The implementation and validation of a single beam, fixed on the left end, and undergoing active contraction can be found in ``tutorials/single_line_active_contraction.ipynb``.

## Validating a Constrained Single Beam Active Contraction <a name="constrained"></a>

The implementation and validation of a single beam, constrained on both ends, and undergoing active contraction can be found in ``tutorials/constrained_single_line_active_contraction.ipynb``.


## Skills used from class <a name="install"></a>
For this final project, the skills I used from class are: implementation of mechanic problems in FEniCSx, validating FEM solution against analytical solution, setting up GitHub repositories, organize and test files based on Test-Driven Development, meshing.