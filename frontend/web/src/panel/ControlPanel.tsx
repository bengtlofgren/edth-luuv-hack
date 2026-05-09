import type { ConfidenceLevel, SimStatus } from "../types";

const SIGMA_MAX = 10;

interface Props {
  connected: boolean;
  status: SimStatus;
  waypointCount: number;
  plannerSync: "idle" | "syncing" | "linked" | "paused" | "error";
  confidence: ConfidenceLevel;
  onConfidenceChange: (c: ConfidenceLevel) => void;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  t: number | null;
  mean: [number, number] | null;
  sigmaX: number | null;
  sigmaY: number | null;
  rho: number | null;
}

export function ControlPanel({
  connected,
  status,
  waypointCount,
  plannerSync,
  confidence,
  onConfidenceChange,
  onPlay,
  onPause,
  onReset,
  t,
  mean,
  sigmaX,
  sigmaY,
  rho,
}: Props) {
  const canPlay =
    connected &&
    waypointCount >= 2 &&
    (status === "idle" || status === "paused" || status === "done");
  const canPause = connected && status === "running";

  return (
    <aside className="panel">
      <section>
        <h2>Connection</h2>
        <div>
          <span
            className={`status-dot ${connected ? "connected" : "disconnected"}`}
          />
          {connected ? "ws://127.0.0.1:8080/ws" : "disconnected"}
        </div>
        <div style={{ fontSize: 12, color: "#7a90a8" }}>state: {status}</div>
      </section>

      <section>
        <h2>Command Center</h2>
        <div className={`sync-pill ${plannerSync}`}>
          {plannerSync === "idle"
            ? "route idle"
            : plannerSync === "syncing"
              ? "syncing route"
              : plannerSync === "linked"
                ? "route linked"
                : plannerSync === "paused"
                  ? "route paused"
                  : "sync failed"}
        </div>
      </section>

      <section>
        <h2>Path</h2>
        <div style={{ fontSize: 13 }}>
            {waypointCount === 0
              ? "Click on the canvas to add waypoints."
              : `${waypointCount} waypoint${waypointCount === 1 ? "" : "s"}${
                status === "idle" || status === "done" ? " (click to add more)" : ""
              }`}
        </div>
      </section>

      <section>
        <h2>Playback</h2>
        <div className="btn-row">
          <button
            className="btn primary"
            disabled={!canPlay}
            onClick={onPlay}
          >
            Play
          </button>
          <button className="btn" disabled={!canPause} onClick={onPause}>
            Pause
          </button>
          <button
            className="btn"
            disabled={!connected}
            onClick={onReset}
          >
            Reset
          </button>
        </div>
      </section>

      <section>
        <h2>Confidence</h2>
        <div className="radio-row">
          {(
            [
              ["1sigma", "1σ"],
              ["2sigma", "2σ"],
              ["95", "95%"],
            ] as [ConfidenceLevel, string][]
          ).map(([key, label]) => (
            <label key={key}>
              <input
                type="radio"
                name="conf"
                checked={confidence === key}
                onChange={() => onConfidenceChange(key)}
              />{" "}
              {label}
            </label>
          ))}
        </div>
      </section>

      <section>
        <h2>Uncertainty</h2>
        <div>
          <div className="bar-label">
            <span>σx</span>
            <span>{sigmaX === null ? "—" : `${sigmaX.toFixed(2)} m`}</span>
          </div>
          <div className="bar">
            <div
              className="bar-fill"
              style={{
                width: `${Math.min(100, ((sigmaX ?? 0) / SIGMA_MAX) * 100)}%`,
              }}
            />
          </div>
        </div>
        <div>
          <div className="bar-label">
            <span>σy</span>
            <span>{sigmaY === null ? "—" : `${sigmaY.toFixed(2)} m`}</span>
          </div>
          <div className="bar">
            <div
              className="bar-fill"
              style={{
                width: `${Math.min(100, ((sigmaY ?? 0) / SIGMA_MAX) * 100)}%`,
              }}
            />
          </div>
        </div>
      </section>

      <section>
        <h2>Readout</h2>
        <div className="readout">
          {`t   = ${t === null ? "—" : t.toFixed(2)} s
x   = ${mean === null ? "—" : mean[0].toFixed(2)} m
y   = ${mean === null ? "—" : mean[1].toFixed(2)} m
σx  = ${sigmaX === null ? "—" : sigmaX.toFixed(3)} m
σy  = ${sigmaY === null ? "—" : sigmaY.toFixed(3)} m
ρxy = ${rho === null ? "—" : rho.toFixed(3)}`}
        </div>
      </section>
    </aside>
  );
}
