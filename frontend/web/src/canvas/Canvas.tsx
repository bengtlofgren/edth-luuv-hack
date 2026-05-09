import { useEffect, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Line,
  Circle,
  Ellipse as KEllipse,
  Text,
} from "react-konva";
import type { KonvaEventObject } from "konva/lib/Node";
import type { Mat2, Vec2 } from "../types";
import { ellipseFromCov } from "./ellipse";

const WORLD_HALF = 50;
const GRID_STEP = 10;

interface Props {
  waypoints: Vec2[];
  mean: Vec2 | null;
  cov: Mat2 | null;
  ellipseK: number;
  editable: boolean;
  onAddWaypoint: (p: Vec2) => void;
}

export function Canvas({
  waypoints,
  mean,
  cov,
  ellipseK,
  editable,
  onAddWaypoint,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 800 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      setSize({ w: Math.max(100, r.width), h: Math.max(100, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const side = Math.min(size.w, size.h);
  const scale = side / (WORLD_HALF * 2);
  const cx = size.w / 2;
  const cy = size.h / 2;

  const w2c = (p: Vec2): [number, number] => [
    cx + p[0] * scale,
    cy - p[1] * scale,
  ];
  const c2w = (x: number, y: number): Vec2 => [
    (x - cx) / scale,
    -(y - cy) / scale,
  ];

  const gridLines: number[][] = [];
  for (let v = -WORLD_HALF; v <= WORLD_HALF; v += GRID_STEP) {
    const [, y0] = w2c([0, v]);
    const [x0] = w2c([v, 0]);
    gridLines.push([0, y0, size.w, y0]);
    gridLines.push([x0, 0, x0, size.h]);
  }

  const polyPoints = waypoints.flatMap((p) => w2c(p));
  const e = cov ? ellipseFromCov(cov, ellipseK) : null;
  const meanPx = mean ? w2c(mean) : null;
  const activeSegment = (() => {
    if (!mean || waypoints.length < 2) return null;
    let best = waypoints[1];
    let bestDist = Number.POSITIVE_INFINITY;
    for (let i = 0; i < waypoints.length - 1; i += 1) {
      const a = waypoints[i];
      const b = waypoints[i + 1];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const lenSq = dx * dx + dy * dy;
      if (lenSq <= 1e-9) continue;
      const t = Math.max(
        0,
        Math.min(1, ((mean[0] - a[0]) * dx + (mean[1] - a[1]) * dy) / lenSq),
      );
      const px = a[0] + t * dx;
      const py = a[1] + t * dy;
      const dist = (mean[0] - px) ** 2 + (mean[1] - py) ** 2;
      if (dist < bestDist) {
        bestDist = dist;
        best = b;
      }
    }
    return best;
  })();
  const shipPoints = (() => {
    if (!mean || !meanPx) return [];
    const target = activeSegment ?? [mean[0], mean[1] + 1];
    const dx = target[0] - mean[0];
    const dy = target[1] - mean[1];
    const mag = Math.hypot(dx, dy) || 1;
    const ux = dx / mag;
    const uy = -dy / mag;
    const px = -uy;
    const py = ux;
    const nose = 12;
    const tail = 8;
    const halfWidth = 7;
    return [
      meanPx[0] + ux * nose,
      meanPx[1] + uy * nose,
      meanPx[0] - ux * tail + px * halfWidth,
      meanPx[1] - uy * tail + py * halfWidth,
      meanPx[0] - ux * tail - px * halfWidth,
      meanPx[1] - uy * tail - py * halfWidth,
    ];
  })();

  const handleStageClick = (evt: KonvaEventObject<MouseEvent>) => {
    if (!editable) return;
    const stage = evt.target.getStage();
    if (!stage) return;
    const pos = stage.getPointerPosition();
    if (!pos) return;
    const [wx, wy] = c2w(pos.x, pos.y);
    if (
      wx < -WORLD_HALF ||
      wx > WORLD_HALF ||
      wy < -WORLD_HALF ||
      wy > WORLD_HALF
    )
      return;
    onAddWaypoint([wx, wy]);
  };

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%" }}>
      <Stage width={size.w} height={size.h} onClick={handleStageClick}>
        <Layer listening={false}>
          {gridLines.map((pts, i) => (
            <Line key={i} points={pts} stroke="#13202f" strokeWidth={1} />
          ))}
          <Line
            points={[0, cy, size.w, cy]}
            stroke="#2a3d57"
            strokeWidth={1.5}
          />
          <Line
            points={[cx, 0, cx, size.h]}
            stroke="#2a3d57"
            strokeWidth={1.5}
          />
          <Text
            x={size.w - 60}
            y={cy + 6}
            text="+x (m)"
            fill="#5a7693"
            fontSize={11}
          />
          <Text x={cx + 6} y={6} text="+y (m)" fill="#5a7693" fontSize={11} />
        </Layer>
        <Layer listening={false}>
          {waypoints.length >= 2 && (
            <Line
              points={polyPoints}
              stroke="#7fb3d5"
              strokeWidth={2}
              dash={[6, 4]}
            />
          )}
          {waypoints.map((p, i) => {
            const [x, y] = w2c(p);
            return (
              <Circle
                key={i}
                x={x}
                y={y}
                radius={4}
                fill="#7fb3d5"
                stroke="#0a1320"
                strokeWidth={1}
              />
            );
          })}
          {meanPx && e && (
            <KEllipse
              x={meanPx[0]}
              y={meanPx[1]}
              radiusX={Math.max(1, e.semiMajor * scale)}
              radiusY={Math.max(1, e.semiMinor * scale)}
              rotation={(-e.rotationRad * 180) / Math.PI}
              stroke="#ffb300"
              strokeWidth={2.5}
              dash={[8, 5]}
              fill="rgba(255, 179, 0, 0.18)"
            />
          )}
          {meanPx && e && (
            <KEllipse
              x={meanPx[0]}
              y={meanPx[1]}
              radiusX={Math.max(1, e.semiMajor * 0.45 * scale)}
              radiusY={Math.max(1, e.semiMinor * 0.45 * scale)}
              rotation={(-e.rotationRad * 180) / Math.PI}
              stroke="rgba(255, 179, 0, 0.75)"
              strokeWidth={1.5}
              fillEnabled={false}
            />
          )}
          {meanPx && (
            <Line
              points={shipPoints}
              closed
              fill="#ff5252"
              stroke="#ffffff"
              strokeWidth={1.5}
              shadowColor="#ff5252"
              shadowBlur={8}
            />
          )}
        </Layer>
      </Stage>
    </div>
  );
}
