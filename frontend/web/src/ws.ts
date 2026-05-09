import { useEffect, useRef, useState } from "react";
import type { ClientMsg, ServerMsg, SimStatus } from "./types";

const WS_URL = "ws://127.0.0.1:8080/ws";
const RECONNECT_DELAY_MS = 1000;

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

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: number | undefined;

    const connect = () => {
      if (cancelled) return;
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setConnected(true);
      };

      ws.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const msg: ServerMsg = JSON.parse(ev.data);
          if (msg.type === "tick") setLastTick(msg);
          else if (msg.type === "status") setStatus(msg.state);
        } catch (err) {
          console.error("bad ws message", err);
        }
      };

      const handleClose = () => {
        if (cancelled) return;
        setConnected(false);
        wsRef.current = null;
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS);
      };

      ws.onclose = handleClose;
      ws.onerror = () => {
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
      ws.send(JSON.stringify(msg));
    }
  };

  return { connected, status, lastTick, send };
}
