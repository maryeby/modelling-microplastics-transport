`figure9_reproduction.py`  
This program reproduces Figure 9 from Sevilla 2021, and is the only program that does not employ the object-oriented structure. In other words, the program is self-contained and does not import the `OceanWave` class.  

## `ocean_wave` directory
`ocean_wave.py`  
This program contains the `OceanWave` class, which defines an object representing a 
wave in the ocean.

- `__init__(amplitude=0.1, wavelength=10, depth=8, density=2/3, stokes_num=1e-5, bet
a=1)`  
   Initializes an OceanWave object according to the default or provided parameters.
- `get_amplitude()`  
   Returns the amplitude $A$.  
- `get_wavelength()`  
   Returns the wavelength $\lambda$.
- `get_depth()`  
   Returns the depth $h$.
- `get_density()`  
   Returns the density $R$.
- `get_stokes_num()`  
   Returns the Stokes number $St$.
- `get_beta()`  
   Returns the value of beta.
- `get_wave_num`  
   Returns the wave number $k$.
- `get_angular_freq()`  
   Returns the angular frequency $\omega$.
- `get_max_velocity()`  
   Returns the maximum velocity $U$.
- `get_response_time()`  
   Returns the response time $\tau$.
- `get_period()`  
   Returns the period.
- `get_particle_history()`  
   Returns the history of v dot.
- `get_fluid_history()`  
   Returns the history of u dot.
- `get_timesteps()`  
   Returns the time steps at which the model is evaluated.
- `fluid_velocity(x, z, t)`  
   Returns the fluid velocity vector u for water of arbitrary depth.
- `fluid_derivative(x, z, t)`  
   Returns the derivative along the trajectory of the fluid element, $\frac{\mathrm{D}\textit{\textbf{u}}}{\mathrm{D}t}$.
- `particle_trajectory(fun, x_0, z_0, u_0, w_0)`  
   Computes the particle trajectory for specified initial conditions.
- `particle_velocity(fun, x_0, z_0, u_0, w_0)`  
   Computes the particle velocities for specified initial conditions.
- `compare_drift_velocities(fun, initial_depths, x_0)`  
   Compares and plots the drift velocities for various initial depths.
- `mr_no_history(t, y)`  
   Returns the evaluation of the M-R equation without the history term.
- `mr_with_history(t, y)`  
   Returns the evaluation of the M-R equation with the history term.
- `santamaria(t, y)`  
   Returns the evaluation of the M-R equation using the model from Santamaria et al. 2013.
   
`single_trajectory.py`  
This program plots the trajectory of a single particle.  
`drift_vel_comparisons.py`  
This program plots the analytical and numerical solutions for the Stokes drift velocity for various depth cases: shallow, intermediate, and deep water.  
- `run_comparisons(depth)`  
   Runs drift velocity comparisons for a specified depth.
`orbits_with_drift_vel.py`    
This program plots the analytical and numerical solutions for the Stokes drift velocity for the shallow, intermediate, and deep water cases, as well as plotting the corresponding particle trajectories.  
