import type { Mat2 } from "../types";

export interface EllipseParams {
  semiMajor: number;
  semiMinor: number;
  rotationRad: number;
  sigmaX: number;
  sigmaY: number;
  rho: number;
}

// Eigendecomposition of a symmetric 2x2 covariance matrix.
// Scaled by k (1.0 = 1σ, 2.0 = 2σ, sqrt(5.991) ≈ 95% confidence for 2 dof).
export function ellipseFromCov(cov: Mat2, k: number): EllipseParams {
  const a = cov[0][0];
  const b = cov[0][1];
  const d = cov[1][1];

  const trace = a + d;
  const det = a * d - b * b;
  const disc = Math.max(0, (trace * trace) / 4 - det);
  const sqrtDisc = Math.sqrt(disc);

  const lambda1 = trace / 2 + sqrtDisc;
  const lambda2 = trace / 2 - sqrtDisc;

  const semiMajor = k * Math.sqrt(Math.max(0, lambda1));
  const semiMinor = k * Math.sqrt(Math.max(0, lambda2));

  // Eigenvector of lambda1: aligns major axis.
  let rotationRad: number;
  if (Math.abs(b) < 1e-12) {
    rotationRad = a >= d ? 0 : Math.PI / 2;
  } else {
    rotationRad = Math.atan2(lambda1 - a, b);
  }

  const sigmaX = Math.sqrt(Math.max(0, a));
  const sigmaY = Math.sqrt(Math.max(0, d));
  const rho = sigmaX > 0 && sigmaY > 0 ? b / (sigmaX * sigmaY) : 0;

  return { semiMajor, semiMinor, rotationRad, sigmaX, sigmaY, rho };
}
