
### Coordinate Frames
#### [WGS-84](https://gnssdecoded.com/wgs-84-world-geodetic-system/#how-to-convert-wgs-84-coordinates)
World Geodetic System 1984 is a realization of Geodetic Coordinate System. It uses an Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinate system where
 - the X-axis points to the Greenwich Meridian, 
 - the Z-axis aligns with the Conventional Terrestrial Pole (CTP), and 
 - the Y-axis completes the right-handed system.

The ECEF realization of WGS-84 serves as the global parent frame, while ENU (East-North-Up) and NED (North-East-Down) are local tangent plane frames derived from it for regional navigation and control.


#### Geodetic Coordinates: Latitude, Longitude, Altitude
A curvilinear representation of the ECEF frame. The coordinates are converted from $(X, Y, Z)$ meters to $(\phi, \lambda, h)$ degrees/meters for intuitive terrestrial positioning.

- Latitude ( $\phi$ ): The angle between the equatorial plane and the normal to the ellipsoid at the point of interest. 
- Longitude ( $\lambda$ ): The angle in the equatorial plane from the prime meridian to the point's meridian plane. 
- Height ( $h$ ): The distance along the ellipsoidal normal from the ellipsoid surface to the point. 


#### ENU and NED
Designed to express relative motion, velocity, and orientation, not absolute position.

| Feature | **ENU (East-North-Up)** | **NED (North-East-Down)** |
| :--- | :--- | :--- |
| **X-Axis** | Points **East** | Points **North** |
| **Y-Axis** | Points **North** | Points **East** |
| **Z-Axis** | Points **Up** (away from Earth) | Points **Down** (toward Earth) |
| **Handedness** | Right-handed | Right-handed |
| **Gravity Vector** | Negative Z ($[0, 0, -g]$) | Positive Z ($[0, 0, g]$) |
| **Primary Use** | Robotics, GIS, Ground Vehicles | Aerospace, Aviation, Marine |

In ENU, the "Up" axis aligns with the normal of the WGS-84 ellipsoid pointing outward, making it intuitive for ground-based mapping. In NED, the "Down" axis points inward toward the Earth's center, which simplifies aviation dynamics where altitude is often treated as a positive distance below the vehicle and gravity acts in the positive Z direction.

Since airplanes and ships move above ground they wanna know "how hight am I", so the NED convention, where positive direction for Z-axis is down is more natural. The ENU is more natural for ground vehicles and robots. 

##### Transformation from WGS-84
The relationship is defined by a translation and rotation process centered on a specific *reference point* (latitude $\phi$, longitude $\lambda$, altitude $h$ ):

- Translation: The origin is shifted from the Earth's center of mass (ECEF) to a specific point on the Earth's surface (the local observer or vehicle position). 
- Rotation: The global axes are rotated using a rotation matrix based on the reference latitude and longitude to align with the local cardinal directions. 

Mathematically, a position vector in ECEF ($r_{ecef}$) is converted to a local frame ($r_{local}$) by subtracting the reference point's ECEF position ($r_{ref}$) and applying a rotation matrix ($R$): $r_{local} = R⋅(r_{ecef} − r_{ref})$

Even though it may seem that whenever the reference point is chosen as your current position, the position in that ENU/NED is always zero, the frame is useful for expressing velocity, acceleration, attitude (RPY relative to local horizon). 

For example, in NED, velocity components represent Northward speed, Eastward speed, and Downward sink/climb rate. This is critical for flight control systems. Expressing velocity in ECEF is unintuitive (mixing X, Y, Z changes), whereas ENU/NED directly gives "forward," "sideways," and "up/down" speeds relative to the ground.


### Rotation differentials

Rotation matrix differential equation:
$$
    \dot{R} = R\omega^{\wedge}
$$


### Estimation Theory Refresher
Given a model $p(z \mid x)$ describing how measurements $z$ are generated from the state $x$, we wanna find the best estimator mapping observations to state estimates $\hat{x}(z)$.

The mean square error (MSE) of the estimator is given by
$$
    MSE(\hat{x}) = E[(x - \hat{x})^2] = E[(\hat{x} - E[\hat{x}])^2] + (x - E[\hat{x}])^2
$$
where bias and variance are identified
$$
\begin{align*}
    Var(\hat{x})    &= E[(\hat{x} - E[\hat{x}])^2] \\
    Bias(\hat{x})   &= (x - E[\hat{x}])
\end{align*}
$$
_Note: the expectation $E[\hat{x}]$ is over $z$, because $\hat{x}(z)$ is random variable due to noise in measurements._

Estimator $\hat{x}(z)$ is said to be:
- _unbiased_: if $\mathrm{Bias}(\hat{x}) = 0$
- _consistent_: if $\mathrm{Bias}(\hat{x}) \to 0$ and $\mathrm{Var}(\hat{x}) \to 0$ as $N \to \infty$
- _efficient_: if $\mathrm{Bias}(\hat{x}) = 0$ and $\mathrm{Var}(\hat{x}) \to \mathrm{CRLB}$ as $N \to \infty$
- _minimum variance unbiased (MVU)_: unbiased + variance minimizing
- _best linear unbiased (BLUE)_: $\hat{x}(z) = Lz$ (linear) + unbiased + MSE minimizing



## Kalman Filter Tunning

_Tricks to monitor, tune, set and fit covariances in KFs._

See: _Gustafsson, Statistical Sensor Fusion_

### Intuition
In object tracking scenarios, for kinematic states (position, velocity, acceleration) the process noise standard deviations can be set by intuition about the type of the tracked moving object.

For example, if I know the tunned process noise covariance is used to track cars, I can set the standard devition of acceleration $\sigma_a = 9 \tfrac{m^2}{s^2}$, by which I'm saying: "I expect the true acceleration to somewhere in $[-\sigma_a, \sigma_a]$ interval with $95\%$ probability (assuming that acceleration noise follows standard normal distribution).

In the measurement model, the sensor noise variances can be read off the sensor datasheet.


### NIS: Normalized Innovation Squared
Useful for checking if measurement noise covariance $R$ is well-calibrated.

Assume measurement model
$$
    z = h(x) + r,\qquad r \sim N(0, R)
$$
the NIS is given by
$$
    \nu = (z - \hat{z})^T S^{-1} (z - \hat{z})
$$

Any deviation from [$\chi^2$ distributedness](https://en.wikipedia.org/wiki/Chi-squared_distribution) of $\nu$ indicates something wrong with either:
- measurement covariance $R$ is not set properly
- measurement model $h(x)$: could be inappropriate, especially relevant in 
- the moment transform for computing the predicted measurement could have high error

### NEES: Normalized Estimation Error Squared
Measures estimator consistency by testing if the filtered error covariance $P$ accurately reflects the actual estimation error $(x - \hat{x})$, i.e. whether it is well-calibrated.

Needs access to ground truth (true system trajectories of the estimated state $x$), hence limited practical usability

$$
    \epsilon = (x - \hat{x})^T P^{-1} (x - \hat{x})
$$
where $\hat{x}$ is the state estimate and the error covariance is
$$
    P = E[(x - \hat{x})(x - \hat{x})^T]
$$
$\epsilon$ is $\chi^2$-distributed with $n_x = dim(x)$ degrees of freedom,  shortly then $\epsilon\sim\chi^2(n_x)$, if estimation error $(x - \hat{x})\sim N(0, I)$.

The hypothesis

> $H_0$: The estimation error $(x - \hat{x})$ is consistent with the filters error covariance $P$

is accepted if NEES $\epsilon$ falls within bounds of confidence (acceptance) interval
$$
    \epsilon \in [r_0, r_1]
$$

which is chosen such that the probability that $H_0$ is accepted is $(1 - \alpha)$, thus
$$
    P(\epsilon \in [r_0, r_1] \mid H_0) = 1 - \alpha.
$$
The confidence interval is calculated from inverse CDF $F^{-1}$ of $\chi^2$ PDF
$$
    [r_0,\ r_1] = [F(\frac{\alpha}{2}; n_x),\ F(1 - \frac{\alpha}{2}; n_x)].
$$

During analysis, first we choose desired confidence level $\alpha$, then get the confidence interval $[r_0, r_1]$, plot the NEES over time together with the interval. The number of times $\epsilon$ falss within the confidence interval should be below $(1 - \alpha)$.

- $\epsilon > r_1$: _overconfidence_: filter estimates the error to be lower than it is in reality, process noise covariance too low (provided measurement covariance correct)
- $\epsilon < r_0$: _inefficiency_: filter estimates the error to be larger than it is in reality, process noise covariance too large (provided measurement covariance correct)

In both cases, we can expect that an inconsistent state estimator will not reach the smallest possible estimation error (CRLB).

### Error-State EKF (ESKF)
Key trick in ESKF is expression of the state as composition of nominal state and error term
$$
    x = \hat{x} \,\oplus\, \delta x \quad\Longrightarrow\quad \delta x = x \,\ominus\, \hat{x} \qquad \delta x \sim N(0, P)
$$
where the operators $\,\oplus\,$ and $\,\ominus\,$ are classical addition/subtraction for vectors and the right plus/minus for Lie group elements (e.g. quaternions, rotation mats).

Prediction
$$
\begin{align*}
    \hat{x}_{k∣k−1} &= f(\hat{x}_{k−1|k-1}​, u_{k-1}) \\
    % \delta x_k &​= F_k ​\delta x_{k−1}​ + G_k​w_k​ \\
    P^{\delta x}_{k∣k−1}​ &= F_{k-1} ​P^{\delta x}_{k−1∣k−1}​ F_{k-1}^T ​+ Q
\end{align*}
$$

Update
$$
\begin{align*}
    \hat{z}_{k|k-1} &= h(\hat{x}_{k|k-1}) \\
    y_k &= z_k - \hat{z}_{k|k-1} \\
    K_k &= P^{\delta x}_{k|k-1}H_k^T(H_kP^{\delta x}_{k|k-1}H_k^T + R)^{-1} \\
    \delta \hat{x}_{k|k} &= K_ky_k \\
    P^{\delta x}_{k|k} &= (I - K_kH_k)P^{\delta x}_{k|k-1}
\end{align*}
$$
State injection and reset
$$
\begin{align*}
    \hat{x}_{k|k} &= \hat{x}_{k|k-1} \,\oplus\, \delta\hat{x}_{k|k} \\
    P_{k|k} &= G_k P_{k|k} G_k^T \\
    \delta\hat{x}_{k|k} &\leftarrow 0
\end{align*}
$$
If $\oplus$ is standard addition, the covariance adjustment effectively doesn't occur because then $G_k = I$. 
For manifold components (rotations), Lie group composition is non-commutative. $\mathbf{G}_k$ acts as a parallel transport / frame transformation operator that maps the error covariance to the updated tangent space centered at the new nominal state $\hat{x}_{k|k}$.


### Inertial Navigation System (INS)
INS is a great motivation for sensor fusion and integrating complementary information from multiple sensors.

An inertial navigation system (INS) is a self-contained navigation device that uses motion sensors, specifically accelerometers and gyroscopes, to continuously calculate an object's position, orientation, and velocity through dead reckoning without needing external references. 

Once initialized with a known starting point, the system integrates sensor data to track movement relative to that origin, making it immune to jamming and effective in environments where GPS or GNSS signals are unavailable or unreliable, such as underwater, in tunnels, or in contested military zones. 

Modern INS solutions often integrate with GNSS receivers to correct the inevitable drift errors that accumulate over time, combining the absolute accuracy of satellite navigation with the continuous, high-refresh-rate tracking of inertial sensors.  These systems are critical for aviation, defense, submarines, and autonomous vehicles, providing precise, real-time positioning data even when external infrastructure is disrupted or denied. 



### IMU Preintegration

See also:
- [OpenVINS Docs: IMU Propagation Derivations](https://docs.openvins.com/propagation.html)
- [[PDF] Trawny, Roumeliotis, Indirect Kalman Filter for 3D Attitude Estimation](https://mars.cs.umn.edu/tr/reports/Trawny05b.pdf): derivations of attitude propagation equations involving quaternions, with all the necessary definitions of quaternion operations and calculus (derivative and integral).
- [OpenIMU docs](https://openimu.readthedocs.io/en/latest/algorithms/STM_Quaternion.html): based on Trawny but more condensed.

IMU delivers measurements at a orders of magnitude higher rate (100-1000 Hz) than the camera images (10-30 Hz).

IMU does not observe the pose directly, but rather the angular velocity $ \boldsymbol{\omega} $ and linear acceleration $\mathbf{a}$ which need to be integrated over time to obtain the pose $\mathbf{p}$.

<!-- TODO: needs fleshing out: formulate model, describe everything, then continue with the estimation algorithm on that model -->

The continuous-time dynamcs are:
$$
   \begin{align*}
      \dot{\mathbf{q}} &= \frac{1}{2} \mathbf{q} \otimes \boldsymbol{\omega} \\
      \dot{\mathbf{v}} &= \mathbf{a} \\
      \dot{\mathbf{p}} &= \mathbf{v} 
   \end{align*}
$$
<!-- Analytically intractable due to noise and nonlinearity, and computationally expensive if done naively at such high rate. -->


IMU preintegration is a technique to compute the IMU measurements in the local frame of the IMU. 
It is used to compute the relative pose between two keyframes $i$ and $j$


$$
\begin{align*}
  \dot{\bar{\mathbf{q}}}(t) &= \frac{1}{2} \bar{\mathbf{q}}(t) \otimes \begin{bmatrix} \mathbf{0} \\ \boldsymbol{\omega}(t) \end{bmatrix} \\
  \dot{\mathbf{p}}(t) &= \mathbf{v}(t) \\
  \dot{\mathbf{v}}(t) &= \mathbf{R}(\bar{\mathbf{q}}(t))^\top \mathbf{a}(t) + \mathbf{g} \\
  \dot{\mathbf{b}}_g(t) &= \mathbf{n}_{wg}(t) \\
  \dot{\mathbf{b}}_a(t) &= \mathbf{n}_{wa}(t) 
\end{align*}
$$

> TODO: Quat vs Rot mat formulation: Which should I use?

Between keyframes the IMU biases are assumed constant, and the IMU measurements are integrated to compute the relative pose, velocity, and biases between the two keyframes.

Solution to the IMU kinematics is given by the following equations:
$$
\begin{align*}
  \Delta \mathbf{R}_{i+1} &= \Delta \mathbf{R}_i \cdot \exp([\tilde{\boldsymbol{\omega}}_i \Delta t]_\times) \\
  \Delta \mathbf{v}_{i+1} &= \Delta \mathbf{v}_i + \Delta \mathbf{R}_i \cdot \tilde{\mathbf{a}}_i \Delta t \\
  \Delta \mathbf{p}_{i+1} &= \Delta \mathbf{p}_i + \Delta \mathbf{v}_i \Delta t + \frac12 \Delta \mathbf{R}_i \cdot \tilde{\mathbf{a}}_i \Delta t^2 
\end{align*}
$$

The first equation is right $\oplus$ on $SO(3)$ defined as $R \oplus \delta\theta = R(\theta + \delta\theta) = R(\theta)\exp(\delta\theta^\wedge)$, where the hat operator $^\wedge$ for $SO(3)$ is the skew-symmetric operator $[\;\cdot\;]_{\times}$. We could then write $\Delta R_{i+1} = \Delta R_i \oplus \delta\theta$, where $\delta\theta = \omega_i\Delta t$ is the angle increment.

These need to be fed with estimates of the true angular velocity and linear acceleration, which are obtained from bias estimates:
$$
\begin{align*}
  \tilde{\boldsymbol{\omega}}_i &= \boldsymbol{\omega}_{m,i} - \hat{\mathbf{b}}_g \\
  \tilde{\mathbf{a}}_i &= \mathbf{a}_{m,i} - \hat{\mathbf{b}}_a
\end{align*}
$$
The IMU bias estimates are held constant for the duration of the preintegration between keyframes.
However, the IMU bias estimates are actively updated using EKF (or other filter) running independently.


### 🚧 MSCKF: Multi-State Constraint Kalman Filter

Mainly for fusing camera and IMU (VIO), but one can replace camera with other exteroceptive sensors instead.

Basically an error-state EKF whose state vector contains the current pose and a sliding window of past poses

- 🚧 : Null-space projection trick: stack measurements and pre-multiply with suitable matrix that projects the "image features" into its null-space, effectively marginalizing them away
- 🚧 Q?: Is the image feature information lost? What good is it then to project them away?


### 🚧 IMU + Wheel Encoder Fusion
Assuming wheeled robot (e.g.: car, tank)

#### IMU Measurement Model
Accelerometer measures $a \in R^3$ linear acceleration and gyroscope measures $\omega \in R^3$ angular velocity (angular rate)
$$
\begin{align*}
    a_m &= a + b_a + n_a \\
    \omega_m &= \omega + b_g + n_g \\
\end{align*}
$$
where $b_a, b_g$ are accelerometer and gyro biases that slowly drift over time due to sensor temperature changes.

#### 🚧 Wheel Encoder Measurement Model
Wheel encoders measure $v \in R^3$ linear velocity and $\omega \in R^3$ angular velocity (collectively termed _twist_)

$$
\begin{align*}
    v_m &= \frac{v_R + v_L}{2} \\
    \omega_m &= \frac{v_R - v_L}{L}
\end{align*}
$$
_CHECK: Suspicous units on $\omega_m$ k_


IMU rate $\geq$ wheel encoder rate

#### State evolution

Since biases drift slowly over time, it's therefore difficult to fully eliminate through static calibration alone, we estimate them as part of the state.

You'd think that every measurement gets fused via the update step, but not with IMU.

IMU is "fused" by feeding the bias-corrected measurements into the dynamics (motion model) as a control input; in other words, it drives the prediction dynamics. The predictive state estimate advances to the next time step on every IMU measurement.

Update is performed when the wheel encoder measurement arrives.

#### Fusion

Algorithm sketch

```
Initialise x̂₀, P₀
for each new sensor message:
    if message is IMU (high-rate):
        // PREDICTION – IMU information flows in
        u ← (ω_m, a_m)
        x̂ ← f(x̂, u)               // mean propagation
        F ← Jacobian of f
        P ← F P Fᵀ + Q             // covariance propagation

    if message is wheel odometry:
        // UPDATE – wheel information flows in
        z ← (v_m, ω_m)
        ẑ ← h(x̂)                  // predicted measurement
        y ← z – ẑ                 // innovation
        H ← Jacobian of h
        S ← H P Hᵀ + R
        K ← P Hᵀ S⁻¹              // Kalman gain
        x̂ ← x̂ + K y              // correct mean
        P ← (I – K H) P           // correct covariance
```



🤖 Grok: simplified EKF for demonstration: state $[x, y, \theta, \omega, b_g]$

### Zero Velocity Update (ZUPT)

Zero Velocity Update (ZUPT) is applied in Kalman filters by using the known stationary state of a platform (velocity = 0) as a measurement update to correct accumulated inertial navigation errors.  When a zero-velocity detector identifies that the sensor platform is stopped, the filter treats the measured velocity as zero, allowing it to estimate and compensate for sensor biases, attitude errors, and position drift that have occurred since the last stationary period. 

This process involves two main stages:

- **Detection**: Algorithms (such as neural networks, SVMs, or threshold-based detectors) analyze accelerometer and gyroscope data to identify specific intervals where the platform is stationary. 
- **Correction**: During these detected intervals, the Kalman filter performs a measurement update. You construct a synthetic measurement vector where the observed velocity is explicitly set to zero ($\mathbf{z}_k = \mathbf{0}$ and define a small measurement noise covariance ($\mathbf{R}$) representing your confidence that the platform is truly still. 

By forcing the velocity error to zero, the filter not only resets the velocity estimate but also retroactively corrects position and attitude errors, effectively bounding the error growth of the inertial navigation system.

The IMU data continues to drive the prediction, while the "doctored" zero-velocity measurement drives the correction. The raw accelerometer and gyroscope readings are always integrated to propagate the state estimate (position, velocity, attitude) and the error covariance matrix forward in time. This is the "dead reckoning" phase.

ZUPT is particularly **effective in GNSS-denied environments** or for pedestrian and vehicle navigation, where it limits the double-integration error of low-cost MEMS sensors. The filter uses this zero-velocity constraint to calibrate IMU biases and improve overall localization accuracy, often outperforming methods that rely solely on curve fitting or maximum likelihood estimation.



TODO: 
- wheel slip detection: 
  - check NIS
  - compare v, omega from both sources, if diff too big => wheel slip
- time sync of updates: interpolation, retrodiction / prediction to same time stamp
- delayed measurements: OOSM update



Once slip is detected you typically:

inflate the wheel measurement covariance $  R  $ (so the filter trusts the IMU more),
temporarily ignore the wheel update,
or estimate an explicit slip velocity state.