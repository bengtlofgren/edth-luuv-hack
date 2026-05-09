import { useMemo, useState } from "react";
import { Canvas } from "./canvas/Canvas";
import { ellipseFromCov } from "./canvas/ellipse";
import { ControlPanel } from "./panel/ControlPanel";
import {
  CONFIDENCE_K,
  type ConfidenceLevel,
  type Vec2,
} from "./types";
import { useEstimate } from "./ws";

const COMMAND_CENTER_ORIGIN =
  window.location.port === "5173" ? "http://127.0.0.1:8000" : "";

function sendCommandCenterPlanner(path: string, init: RequestInit = {}) {
  fetch(`${COMMAND_CENTER_ORIGIN}${path}`, {
    method: "POST",
    ...init,
  }).catch((err) => console.warn("[app] command-center planner sync failed", err));
}

export function App() {
  const { connected, status, lastTick, send } = useEstimate();
  const [waypoints, setWaypoints] = useState<Vec2[]>([]);
  const [confidence, setConfidence] = useState<ConfidenceLevel>("95");

  const editable = status === "idle" || status === "done";

  const ellipseK = CONFIDENCE_K[confidence];

  const derived = useMemo(() => {
    if (!lastTick) return null;
    const e = ellipseFromCov(lastTick.cov, 1.0);
    return {
      sigmaX: e.sigmaX,
      sigmaY: e.sigmaY,
      rho: e.rho,
    };
  }, [lastTick]);

  const handleAddWaypoint = (p: Vec2) => {
    console.log("[app] addWaypoint", p);
    setWaypoints((prev) => [...prev, p]);
  };

  const handlePlay = () => {
    console.log("[app] play clicked", { waypoints, status, connected });
    if (waypoints.length < 2) return;
    if (status !== "paused") {
      sendCommandCenterPlanner("/api/planner/path", {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ waypoints }),
      });
      send({ type: "set_path", waypoints });
    } else {
      sendCommandCenterPlanner("/api/planner/resume");
    }
    send({ type: "play" });
  };

  const handlePause = () => {
    console.log("[app] pause clicked");
    sendCommandCenterPlanner("/api/planner/pause");
    send({ type: "pause" });
  };

  const handleReset = () => {
    console.log("[app] reset clicked");
    sendCommandCenterPlanner("/api/planner/clear");
    send({ type: "reset" });
    setWaypoints([]);
  };

  return (
    <div className="app">
      <div className="canvas-pane">
        <Canvas
          waypoints={waypoints}
          mean={lastTick?.mean ?? null}
          cov={lastTick?.cov ?? null}
          ellipseK={ellipseK}
          editable={editable}
          onAddWaypoint={handleAddWaypoint}
        />
      </div>
      <ControlPanel
        connected={connected}
        status={status}
        waypointCount={waypoints.length}
        confidence={confidence}
        onConfidenceChange={setConfidence}
        onPlay={handlePlay}
        onPause={handlePause}
        onReset={handleReset}
        t={lastTick?.t ?? null}
        mean={lastTick?.mean ?? null}
        sigmaX={derived?.sigmaX ?? null}
        sigmaY={derived?.sigmaY ?? null}
        rho={derived?.rho ?? null}
      />
    </div>
  );
}
