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

type PlannerSync = "idle" | "syncing" | "linked" | "paused" | "error";

async function sendCommandCenterPlanner(path: string, init: RequestInit = {}) {
  try {
    const res = await fetch(`${COMMAND_CENTER_ORIGIN}${path}`, {
      method: "POST",
      ...init,
    });
    return res.ok;
  } catch (err) {
    console.warn("[app] command-center planner sync failed", err);
    return false;
  }
}

export function App() {
  const { connected, status, lastTick, send } = useEstimate();
  const [waypoints, setWaypoints] = useState<Vec2[]>([]);
  const [confidence, setConfidence] = useState<ConfidenceLevel>("95");
  const [plannerSync, setPlannerSync] = useState<PlannerSync>("idle");

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
    setPlannerSync("idle");
  };

  const handlePlay = async () => {
    console.log("[app] play clicked", { waypoints, status, connected });
    if (waypoints.length < 2) return;
    setPlannerSync("syncing");
    if (status !== "paused") {
      const ok = await sendCommandCenterPlanner("/api/planner/path", {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ waypoints }),
      });
      setPlannerSync(ok ? "linked" : "error");
      send({ type: "set_path", waypoints });
    } else {
      const ok = await sendCommandCenterPlanner("/api/planner/resume");
      setPlannerSync(ok ? "linked" : "error");
    }
    send({ type: "play" });
  };

  const handlePause = async () => {
    console.log("[app] pause clicked");
    const ok = await sendCommandCenterPlanner("/api/planner/pause");
    setPlannerSync(ok ? "paused" : "error");
    send({ type: "pause" });
  };

  const handleReset = async () => {
    console.log("[app] reset clicked");
    await sendCommandCenterPlanner("/api/planner/clear");
    setPlannerSync("idle");
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
        plannerSync={plannerSync}
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
