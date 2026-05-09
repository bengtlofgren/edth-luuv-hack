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
              strokeWidth={2}
              fill="rgba(255, 179, 0, 0.12)"
            />
          )}
          {meanPx && (
            <Circle
              x={meanPx[0]}
              y={meanPx[1]}
              radius={6}
              fill="#ff5252"
              stroke="#ffffff"
              strokeWidth={1.5}
            />
          )}
        </Layer>
      </Stage>
    </div>
  );
}
