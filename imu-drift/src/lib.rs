// SPDX-License-Identifier: AGPL-3.0-or-later
//
// imu-drift: closed-form drift propagation and Allan variance estimation
// for IMU error models.
//
// Mathematical references (cited per item in rustdoc):
//   [W07]    Woodman, "An Introduction to Inertial Navigation,"
//            Univ. of Cambridge Computer Lab, Tech. Rep. UCAM-CL-TR-696, 2007.
//   [IEEE952] IEEE Std 952-2020, "IEEE Standard for Specifying and Testing
//            Single-Axis Interferometric Fiber Optic Gyros."
//   [ESN08]  El-Sheimy, Hou, Niu, "Analysis and Modeling of Inertial Sensors
//            Using Allan Variance," IEEE Trans. Instrum. Meas., 57(1), 2008.
//   [TW04]   Titterton & Weston, "Strapdown Inertial Navigation Technology,"
//            2nd ed., IEE/AIAA, 2004.
//   [G13]    Groves, "Principles of GNSS, Inertial, and Multisensor
//            Integrated Navigation Systems," 2nd ed., Artech House, 2013.
//   [F08]    Farrell, "Aided Navigation: GPS with High Rate Sensors," 2008.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

pub mod allan;
pub mod process;
pub mod processes;

#[cfg(feature = "sample")]
pub mod sampling;

pub use process::{DriftProcess, Sum, Vec3};
pub use processes::{
    AngleRandomWalk, BiasInstabilityFlicker, ConstantAccelBiasOnPosition,
    ConstantAccelBiasOnVelocity, ConstantGyroBias, GaussMarkov,
    GyroBiasGravityCoupling, IntegratedVelocityRandomWalk, VelocityRandomWalk,
};
