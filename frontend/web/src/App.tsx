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

export function App() {
  const { connected, status, lastTick, send } = useEstimate();
  const [waypoints, setWaypoints] = useState<Vec2[]>([]);
  const [confidence, setConfidence] = useState<ConfidenceLevel>("95");

  const editable = status === "idle";

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
    setWaypoints((prev) => [...prev, p]);
  };

  const handlePlay = () => {
    if (waypoints.length < 2) return;
    send({ type: "set_path", waypoints });
    send({ type: "play" });
  };

  const handlePause = () => {
    send({ type: "pause" });
  };

  const handleReset = () => {
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
