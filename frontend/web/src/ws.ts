import { useEffect, useRef, useState } from "react";
import type { ClientMsg, ServerMsg, SimStatus } from "./types";

const WS_URL = "ws://127.0.0.1:8080/ws";
const RECONNECT_DELAY_MS = 1000;
const LOG_TICKS = false;

const log = (...args: unknown[]) => console.log("[ws]", ...args);

export interface WsApi {
  connected: boolean;
  status: SimStatus;
  lastTick: Extract<ServerMsg, { type: "tick" }> | null;
  send: (msg: ClientMsg) => void;
}

export function useEstimate(): WsApi {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<SimStatus>("idle");
  const [lastTick, setLastTick] = useState<Extract<
    ServerMsg,
    { type: "tick" }
  > | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const tickCountRef = useRef(0);
  const prevStatusRef = useRef<SimStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: number | undefined;

    const connect = () => {
      if (cancelled) return;
      // First status from a fresh connection is informational; never wipe lastTick on it.
      prevStatusRef.current = null;
      log("connecting to", WS_URL);
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        log("open");
        setConnected(true);
      };

      ws.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const msg: ServerMsg = JSON.parse(ev.data);
          if (msg.type === "tick") {
            tickCountRef.current += 1;
            if (LOG_TICKS || tickCountRef.current === 1) {
              log("tick", msg);
            }
            setLastTick(msg);
          } else if (msg.type === "status") {
            log("status ->", msg.state);
            const prev = prevStatusRef.current;
            if (msg.state === "idle" && prev !== null && prev !== "idle") {
              tickCountRef.current = 0;
              setLastTick(null);
            }
            prevStatusRef.current = msg.state;
            setStatus(msg.state);
          }
        } catch (err) {
          console.error("[ws] bad message", err, ev.data);
        }
      };

      const handleClose = (ev: CloseEvent) => {
        if (cancelled) return;
        log("close", { code: ev.code, reason: ev.reason });
        setConnected(false);
        wsRef.current = null;
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS);
      };

      ws.onclose = handleClose;
      ws.onerror = (ev) => {
        console.error("[ws] error", ev);
        ws.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  const send = (msg: ClientMsg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      log("send", msg);
      ws.send(JSON.stringify(msg));
    } else {
      console.warn("[ws] dropped (not open)", msg, "readyState:", ws?.readyState);
    }
  };

  return { connected, status, lastTick, send };
}
