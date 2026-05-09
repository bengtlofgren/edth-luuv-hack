export type Vec2 = [number, number];
export type Mat2 = [[number, number], [number, number]];

export type SimStatus = "idle" | "running" | "paused" | "done";

export type ServerMsg =
  | { type: "tick"; t: number; mean: Vec2; cov: Mat2 }
  | { type: "status"; state: SimStatus };

export type ClientMsg =
  | { type: "set_path"; waypoints: Vec2[] }
  | { type: "play" }
  | { type: "pause" }
  | { type: "reset" };

export type ConfidenceLevel = "1sigma" | "2sigma" | "95";

export const CONFIDENCE_K: Record<ConfidenceLevel, number> = {
  "1sigma": 1.0,
  "2sigma": 2.0,
  "95": Math.sqrt(5.991),
};
