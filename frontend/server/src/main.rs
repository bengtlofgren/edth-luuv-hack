use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::interval;

const TICK_HZ: f64 = 20.0;
const TICK_DT: f64 = 1.0 / TICK_HZ;
const SPEED: f64 = 5.0;
const COV_GROWTH_DIAG: f64 = 0.5;
const COV_GROWTH_OFFDIAG: f64 = 0.05;
const LANDMARK_FIX: f64 = 0.6;
const INIT_COV: [[f64; 2]; 2] = [[0.25, 0.0], [0.0, 0.25]];

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMsg {
    SetPath { waypoints: Vec<[f64; 2]> },
    Play,
    Pause,
    Reset,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMsg<'a> {
    Tick {
        t: f64,
        mean: [f64; 2],
        cov: [[f64; 2]; 2],
    },
    Status {
        state: &'a str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimState {
    Idle,
    Running,
    Paused,
    Done,
}

impl SimState {
    fn as_str(&self) -> &'static str {
        match self {
            SimState::Idle => "idle",
            SimState::Running => "running",
            SimState::Paused => "paused",
            SimState::Done => "done",
        }
    }
}

struct Sim {
    state: SimState,
    path: Vec<[f64; 2]>,
    t: f64,
    s: f64,
    cov: [[f64; 2]; 2],
    seg_starts: Vec<f64>,
    total_length: f64,
    current_segment: usize,
}

impl Sim {
    fn new() -> Self {
        Self {
            state: SimState::Idle,
            path: vec![],
            t: 0.0,
            s: 0.0,
            cov: INIT_COV,
            seg_starts: vec![],
            total_length: 0.0,
            current_segment: 0,
        }
    }

    fn set_path(&mut self, path: Vec<[f64; 2]>) {
        self.path = path;
        self.t = 0.0;
        self.s = 0.0;
        self.cov = INIT_COV;
        self.current_segment = 0;
        self.seg_starts.clear();
        self.total_length = 0.0;
        let mut acc = 0.0;
        for i in 0..self.path.len() {
            self.seg_starts.push(acc);
            if i + 1 < self.path.len() {
                let dx = self.path[i + 1][0] - self.path[i][0];
                let dy = self.path[i + 1][1] - self.path[i][1];
                acc += (dx * dx + dy * dy).sqrt();
            }
        }
        self.total_length = acc;
        self.state = SimState::Idle;
    }

    fn current_mean(&self) -> [f64; 2] {
        if self.path.is_empty() {
            return [0.0, 0.0];
        }
        let seg = self.current_segment;
        if seg + 1 >= self.path.len() {
            return *self.path.last().unwrap();
        }
        let p0 = self.path[seg];
        let p1 = self.path[seg + 1];
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let seg_len = (dx * dx + dy * dy).sqrt();
        let alpha = if seg_len > 0.0 {
            ((self.s - self.seg_starts[seg]) / seg_len).clamp(0.0, 1.0)
        } else {
            0.0
        };
        [p0[0] + alpha * dx, p0[1] + alpha * dy]
    }

    fn tick(&mut self) -> Option<([f64; 2], [[f64; 2]; 2])> {
        if self.state != SimState::Running {
            return None;
        }
        if self.path.len() < 2 {
            self.state = SimState::Done;
            return None;
        }
        self.t += TICK_DT;
        self.s += SPEED * TICK_DT;

        self.cov[0][0] += COV_GROWTH_DIAG * TICK_DT;
        self.cov[1][1] += COV_GROWTH_DIAG * TICK_DT;
        self.cov[0][1] += COV_GROWTH_OFFDIAG * TICK_DT;
        self.cov[1][0] = self.cov[0][1];

        while self.current_segment + 1 < self.path.len() - 1
            && self.s >= self.seg_starts[self.current_segment + 1]
        {
            self.current_segment += 1;
            self.cov[0][0] *= LANDMARK_FIX;
            self.cov[1][1] *= LANDMARK_FIX;
            self.cov[0][1] *= LANDMARK_FIX;
            self.cov[1][0] = self.cov[0][1];
        }

        if self.s >= self.total_length {
            self.s = self.total_length;
            self.current_segment = self.path.len().saturating_sub(2);
            self.state = SimState::Done;
        }

        Some((self.current_mean(), self.cov))
    }
}

async fn send_status(sender: &mut futures_util::stream::SplitSink<WebSocket, Message>, state: SimState) -> bool {
    let msg = ServerMsg::Status { state: state.as_str() };
    let text = serde_json::to_string(&msg).unwrap();
    sender.send(Message::Text(text.into())).await.is_ok()
}

async fn handle_socket(socket: WebSocket) {
    println!("[ws] client connected");
    let (mut sender, mut receiver) = socket.split();
    let mut sim = Sim::new();
    let mut ticker = interval(Duration::from_secs_f64(TICK_DT));
    let mut last_state = sim.state;

    if !send_status(&mut sender, sim.state).await {
        println!("[ws] failed to send initial status; closing");
        return;
    }

    loop {
        tokio::select! {
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        println!("[ws] recv: {text}");
                        let parsed: Result<ClientMsg, _> = serde_json::from_str(&text);
                        match parsed {
                            Ok(ClientMsg::SetPath { waypoints }) => {
                                println!("[ws] set_path: {} waypoints", waypoints.len());
                                sim.set_path(waypoints);
                            }
                            Ok(ClientMsg::Play) => {
                                println!(
                                    "[ws] play (path_len={}, state={:?})",
                                    sim.path.len(),
                                    sim.state
                                );
                                if sim.path.len() >= 2 && sim.state != SimState::Done {
                                    sim.state = SimState::Running;
                                }
                            }
                            Ok(ClientMsg::Pause) => {
                                println!("[ws] pause (state={:?})", sim.state);
                                if sim.state == SimState::Running {
                                    sim.state = SimState::Paused;
                                }
                            }
                            Ok(ClientMsg::Reset) => {
                                println!("[ws] reset");
                                sim.set_path(vec![]);
                            }
                            Err(e) => eprintln!("[ws] bad message: {e} (raw: {text})"),
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        println!("[ws] client closed");
                        break;
                    }
                    Some(Err(e)) => {
                        eprintln!("[ws] recv error: {e}");
                        break;
                    }
                    _ => {}
                }
            }
            _ = ticker.tick() => {
                if let Some((mean, cov)) = sim.tick() {
                    let msg = ServerMsg::Tick { t: sim.t, mean, cov };
                    let text = serde_json::to_string(&msg).unwrap();
                    if sender.send(Message::Text(text.into())).await.is_err() {
                        println!("[ws] send failed; closing");
                        break;
                    }
                }
            }
        }

        if sim.state != last_state {
            println!("[ws] state {:?} -> {:?}", last_state, sim.state);
            if !send_status(&mut sender, sim.state).await {
                break;
            }
            last_state = sim.state;
        }
    }

    println!("[ws] handler exited");
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/ws", get(ws_handler));
    let addr = "127.0.0.1:8080";
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    println!("underwater-mission-planner-server listening on ws://{addr}/ws");
    axum::serve(listener, app).await.unwrap();
}
