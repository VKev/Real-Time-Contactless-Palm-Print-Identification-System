import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, Camera, Cpu, ScanSearch } from "lucide-react";

type AppState = {
  camera: {
    width: number;
    height: number;
    has_frame: boolean;
    latest_frame_id: number;
    latest_frame_ts: number;
    target_upload_fps: number;
  };
  processed: {
    has_roi: boolean;
    last_processed_frame_id: number;
    last_processed_at: number;
    last_processing_ms: number;
  };
  backend: {
    mode: string;
    warmup_ready: boolean;
    warmup_error: string | null;
    process_max_width: number;
    roi_display_size: number;
  };
  streams: {
    roi: string;
    preprocessed: string;
  };
  services: {
    triton: {
      grpc_url: string;
      model_name: string;
      input_name: string;
      output_name: string;
      connected: boolean;
      error: string | null;
    };
    qdrant: {
      host: string;
      port: number;
      grpc_port: number;
      collection: string;
      vector_size: number;
      distance: string;
      connected: boolean;
      error: string | null;
    };
    register: {
      enabled: boolean;
      target_id: number | null;
      insert_count: number;
      pending_count: number;
      last_error: string | null;
      keep_all_samples: boolean;
      queue_maxsize: number;
    };
    verify: {
      enabled: boolean;
      threshold: number;
      top_k: number;
      pending_count: number;
      queue_maxsize: number;
      last_result: string | null;
      last_error: string | null;
      top_matches: VerifyMatch[];
    };
    plot: {
      count: number;
      version: number;
      pending_count: number;
      max_points: number;
    };
  };
};

type PlotPoint = {
  id: string;
  subject_id: string;
  x: number;
  y: number;
  latest: boolean;
};

type ProjectionKind = "pca" | "umap" | "tsne";

type ProjectionSnapshot = {
  kind: ProjectionKind;
  version: number;
  count: number;
  pending_count: number;
  max_points: number;
  status: string;
  error: string | null;
  refreshed_at: number;
  points: PlotPoint[];
};

type VerifyMatch = {
  rank: number;
  point_id: string;
  subject_id: string;
  label: string;
  distance: number;
};

type VerifyConsensus = {
  subjectId: string;
  count: number;
  averageDistance: number;
  bestDistance: number;
  bestRank: number;
};

type StreamMessage = {
  type: "snapshot" | "result" | "register" | "verify" | "plot" | "projection";
  state: AppState;
  images: {
    roi: string;
    preprocessed: string;
  };
  plot?: ProjectionSnapshot;
  umap?: ProjectionSnapshot;
  tsne?: ProjectionSnapshot;
};

const initialState: AppState = {
  camera: {
    width: 1280,
    height: 720,
    has_frame: false,
    latest_frame_id: 0,
    latest_frame_ts: 0,
    target_upload_fps: 16,
  },
  processed: {
    has_roi: false,
    last_processed_frame_id: -1,
    last_processed_at: 0,
    last_processing_ms: 0,
  },
  backend: {
    mode: "roi_plus_preprocessed",
    warmup_ready: false,
    warmup_error: null,
    process_max_width: 0,
    roi_display_size: 224,
  },
  streams: {
    roi: "/api/frame/latest/roi",
    preprocessed: "/api/frame/latest/preprocessed",
  },
  services: {
    triton: {
      grpc_url: "",
      model_name: "",
      input_name: "",
      output_name: "",
      connected: false,
      error: null,
    },
    qdrant: {
      host: "",
      port: 6333,
      grpc_port: 6334,
      collection: "",
      vector_size: 128,
      distance: "EUCLID",
      connected: false,
      error: null,
    },
    register: {
      enabled: false,
      target_id: null,
      insert_count: 0,
      pending_count: 0,
      last_error: null,
      keep_all_samples: true,
      queue_maxsize: 512,
    },
    verify: {
      enabled: false,
      threshold: 35,
      top_k: 10,
      pending_count: 0,
      queue_maxsize: 512,
      last_result: "Verify OFF.",
      last_error: null,
      top_matches: [],
    },
    plot: {
      count: 0,
      version: 0,
      pending_count: 0,
      max_points: 0,
    },
  },
};

const MAX_SOCKET_BUFFERED_BYTES = 2_500_000;
const CAMERA_IDEAL_WIDTH = 4096;
const CAMERA_IDEAL_HEIGHT = 2160;
const CAMERA_TARGET_FRAME_RATE = 24;
const CAMERA_MAX_FRAME_RATE = 30;

function buildPlaceholderDataUrl(label: string) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="224" height="224" viewBox="0 0 224 224">
      <rect width="224" height="224" fill="#000000" />
      <text x="50%" y="50%" fill="#ffffff" font-family="IBM Plex Mono, monospace" font-size="14" text-anchor="middle" dominant-baseline="middle">
        ${label}
      </text>
    </svg>
  `;

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

const DEFAULT_ROI_IMAGE = buildPlaceholderDataUrl("Waiting for ROI");
const DEFAULT_PREPROCESSED_IMAGE = buildPlaceholderDataUrl("Waiting for ROI");

type VideoTrackWithCapabilities = MediaStreamTrack & {
  getCapabilities?: () => MediaTrackCapabilities;
};

function normalizeBase(url: string) {
  return url.replace(/\/+$/, "");
}

async function maximizeVideoTrackResolution(track: MediaStreamTrack) {
  const capableTrack = track as VideoTrackWithCapabilities;
  if (typeof capableTrack.getCapabilities !== "function") {
    return;
  }

  const capabilities = capableTrack.getCapabilities();
  const constraints: MediaTrackConstraints = {};

  if (typeof capabilities.width?.max === "number" && capabilities.width.max > 0) {
    constraints.width = { ideal: capabilities.width.max, max: capabilities.width.max };
  }
  if (typeof capabilities.height?.max === "number" && capabilities.height.max > 0) {
    constraints.height = { ideal: capabilities.height.max, max: capabilities.height.max };
  }
  if (typeof capabilities.frameRate?.max === "number" && capabilities.frameRate.max > 0) {
    const maxFrameRate = Math.min(CAMERA_MAX_FRAME_RATE, capabilities.frameRate.max);
    constraints.frameRate = {
      ideal: Math.min(CAMERA_TARGET_FRAME_RATE, maxFrameRate),
      max: maxFrameRate,
    };
  }

  if (Object.keys(constraints).length === 0) {
    return;
  }

  await track.applyConstraints(constraints);
}

function getTrackResolutionLabel(track: MediaStreamTrack | null) {
  if (!track) {
    return null;
  }

  const settings = track.getSettings();
  const width = typeof settings.width === "number" ? settings.width : 0;
  const height = typeof settings.height === "number" ? settings.height : 0;
  if (width > 0 && height > 0) {
    return `${width}x${height}`;
  }
  return null;
}

function getBackendHttpBase() {
  const envBase = import.meta.env.VITE_BACKEND_URL?.trim();
  if (envBase) {
    return normalizeBase(envBase);
  }

  if (import.meta.env.DEV) {
    return "http://127.0.0.1:7001";
  }

  return window.location.origin;
}

function getBackendWsBase(httpBase: string) {
  if (httpBase.startsWith("https://")) {
    return `wss://${httpBase.slice("https://".length)}`;
  }

  if (httpBase.startsWith("http://")) {
    return `ws://${httpBase.slice("http://".length)}`;
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}`;
}

const BACKEND_HTTP_BASE = getBackendHttpBase();
const BACKEND_WS_URL = `${getBackendWsBase(BACKEND_HTTP_BASE)}/ws/realtime`;

function hashString(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function colorForSubjectId(subjectId: string) {
  const numericId = Number.parseInt(subjectId, 10);
  const seed = Number.isFinite(numericId) ? numericId : hashString(subjectId);
  const hue = ((seed * 137.508) % 360 + 360) % 360;
  const lightness = 38 + (seed % 12);
  return `hsl(${hue.toFixed(1)} 68% ${lightness}%)`;
}

function projectionLabel(kind: ProjectionKind) {
  if (kind === "pca") {
    return "PCA";
  }
  if (kind === "umap") {
    return "UMAP";
  }
  return "t-SNE";
}

function compareSubjectIds(left: string, right: string) {
  const leftNumeric = Number.parseInt(left, 10);
  const rightNumeric = Number.parseInt(right, 10);

  if (Number.isFinite(leftNumeric) && Number.isFinite(rightNumeric) && leftNumeric !== rightNumeric) {
    return leftNumeric - rightNumeric;
  }

  return left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" });
}

function getVerifyConsensus(topMatches: VerifyMatch[]): VerifyConsensus | null {
  if (topMatches.length === 0) {
    return null;
  }

  const grouped = new Map<
    string,
    {
      subjectId: string;
      count: number;
      totalDistance: number;
      bestDistance: number;
      bestRank: number;
    }
  >();

  topMatches.forEach((match) => {
    const current = grouped.get(match.subject_id);
    if (current) {
      current.count += 1;
      current.totalDistance += match.distance;
      current.bestDistance = Math.min(current.bestDistance, match.distance);
      current.bestRank = Math.min(current.bestRank, match.rank);
      return;
    }

    grouped.set(match.subject_id, {
      subjectId: match.subject_id,
      count: 1,
      totalDistance: match.distance,
      bestDistance: match.distance,
      bestRank: match.rank,
    });
  });

  const sorted = [...grouped.values()].sort((left, right) => {
    if (left.count !== right.count) {
      return right.count - left.count;
    }

    const leftAverage = left.totalDistance / left.count;
    const rightAverage = right.totalDistance / right.count;
    if (leftAverage !== rightAverage) {
      return leftAverage - rightAverage;
    }

    if (left.bestDistance !== right.bestDistance) {
      return left.bestDistance - right.bestDistance;
    }

    if (left.bestRank !== right.bestRank) {
      return left.bestRank - right.bestRank;
    }

    return compareSubjectIds(left.subjectId, right.subjectId);
  });

  const winner = sorted[0];
  return {
    subjectId: winner.subjectId,
    count: winner.count,
    averageDistance: winner.totalDistance / winner.count,
    bestDistance: winner.bestDistance,
    bestRank: winner.bestRank,
  };
}

function buildVerifyStatus(verifyState: AppState["services"]["verify"]) {
  if (verifyState.last_error) {
    return verifyState.last_error;
  }

  const result = verifyState.last_result || "";
  const topMatches = verifyState.top_matches;
  const consensus = getVerifyConsensus(topMatches);

  if (result.startsWith("Verified:") && consensus) {
    return `Verified: ID ${consensus.subjectId} (${consensus.count}/${topMatches.length} votes)`;
  }

  if (result.startsWith("No match:")) {
    if (topMatches.length > 0) {
      const minDistance = Math.min(...topMatches.map((match) => match.distance));
      return `Unverified: min Euclid distance ${minDistance.toFixed(4)}`;
    }

    return "Unverified";
  }

  if (result.startsWith("Verify failed:")) {
    return result;
  }

  if (verifyState.enabled) {
    if (consensus) {
      return `Verify live: leading ID ${consensus.subjectId} (${consensus.count}/${topMatches.length} votes)`;
    }

    return "Verify mode is live. Waiting for matches.";
  }

  return result || "Verify mode is off.";
}

function verifyStatusClass(verifyState: AppState["services"]["verify"]) {
  if (verifyState.last_error) {
    return "is-error";
  }

  const result = buildVerifyStatus(verifyState);
  if (result.startsWith("Verified:")) {
    return "is-success";
  }
  if (result.startsWith("Unverified:") || result.startsWith("No match:") || result.startsWith("Verify failed:")) {
    return "is-error";
  }
  if (verifyState.enabled) {
    return "is-live";
  }
  return "";
}

function ProjectionScatterPlot({
  projection,
  kind,
}: {
  projection: ProjectionSnapshot | null;
  kind: ProjectionKind;
}) {
  const width = 880;
  const height = 420;
  const padding = 34;
  const label = projectionLabel(kind);

  if (!projection || projection.points.length === 0) {
    return (
      <div className="plot-empty">
        <span className="frame-label">{label}</span>
        <strong>{kind === "pca" ? "No registered vectors yet" : `No ${label} projection yet`}</strong>
        <span className="muted">
          {projection?.error || projection?.status ||
            (kind === "pca"
              ? "Start register mode and successful Qdrant inserts will appear here in realtime."
              : `Press Refresh to compute ${label} for the current registered vectors.`)}
        </span>
      </div>
    );
  }

  let minX = Math.min(...projection.points.map((point) => point.x));
  let maxX = Math.max(...projection.points.map((point) => point.x));
  let minY = Math.min(...projection.points.map((point) => point.y));
  let maxY = Math.max(...projection.points.map((point) => point.y));

  const xSpan = Math.max(maxX - minX, 1e-6);
  const ySpan = Math.max(maxY - minY, 1e-6);
  const span = Math.max(xSpan, ySpan);
  const pad = span * 0.18 + 0.05;

  minX -= pad;
  maxX += pad;
  minY -= pad;
  maxY += pad;

  const scaleX = (value: number) => padding + ((value - minX) / (maxX - minX || 1)) * (width - padding * 2);
  const scaleY = (value: number) => height - padding - ((value - minY) / (maxY - minY || 1)) * (height - padding * 2);

  const gridLines = [0.2, 0.4, 0.6, 0.8];

  return (
    <svg className="plot-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} scatter plot`}>
      <rect x="0" y="0" width={width} height={height} rx="24" fill="rgba(255,255,255,0.88)" />
      {gridLines.map((ratio) => (
        <line
          key={`v-${ratio}`}
          x1={padding + (width - padding * 2) * ratio}
          x2={padding + (width - padding * 2) * ratio}
          y1={padding}
          y2={height - padding}
          stroke="rgba(17,17,17,0.08)"
        />
      ))}
      {gridLines.map((ratio) => (
        <line
          key={`h-${ratio}`}
          x1={padding}
          x2={width - padding}
          y1={padding + (height - padding * 2) * ratio}
          y2={padding + (height - padding * 2) * ratio}
          stroke="rgba(17,17,17,0.08)"
        />
      ))}
      <rect
        x={padding}
        y={padding}
        width={width - padding * 2}
        height={height - padding * 2}
        rx="18"
        fill="none"
        stroke="rgba(17,17,17,0.12)"
      />
      {projection.points.map((point) => {
        const cx = scaleX(point.x);
        const cy = scaleY(point.y);
        const color = colorForSubjectId(point.subject_id);

        return (
          <g key={point.id}>
            {point.latest ? (
              <circle cx={cx} cy={cy} r="10" fill="none" stroke="#111111" strokeWidth="2.4" opacity="0.9" />
            ) : null}
            <circle cx={cx} cy={cy} r={point.latest ? 5.4 : 4.2} fill={color} opacity={point.latest ? 1 : 0.82} />
          </g>
        );
      })}
      <text x={padding} y="22" fill="#111111" fontSize="14" fontWeight="700">
        {label} projection of registered vectors
      </text>
      <text x={width - padding} y="22" fill="rgba(17,17,17,0.55)" fontSize="12" textAnchor="end">
        Latest point has outer ring
      </text>
    </svg>
  );
}

function App() {
  const [appState, setAppState] = useState<AppState>(initialState);
  const [message, setMessage] = useState("Connecting to ROI backend...");
  const [cameraMessage, setCameraMessage] = useState("Idle");
  const [isCameraRunning, setIsCameraRunning] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [roiImageSrc, setRoiImageSrc] = useState(DEFAULT_ROI_IMAGE);
  const [preprocessedImageSrc, setPreprocessedImageSrc] = useState(DEFAULT_PREPROCESSED_IMAGE);
  const [modeTab, setModeTab] = useState<"register" | "verify">("register");
  const [plotTab, setPlotTab] = useState<ProjectionKind>("pca");
  const [registerTargetId, setRegisterTargetId] = useState("1");
  const [verifyThreshold, setVerifyThreshold] = useState("35");
  const [verifyTopK, setVerifyTopK] = useState("10");
  const [pcaSnapshot, setPcaSnapshot] = useState<ProjectionSnapshot | null>(null);
  const [umapSnapshot, setUmapSnapshot] = useState<ProjectionSnapshot | null>(null);
  const [tsneSnapshot, setTsneSnapshot] = useState<ProjectionSnapshot | null>(null);
  const [isClearingDatabase, setIsClearingDatabase] = useState(false);
  const [showFloatingCamera, setShowFloatingCamera] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const floatingVideoRef = useRef<HTMLVideoElement | null>(null);
  const cameraFrameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const uploadInFlightRef = useRef(false);
  const lastFrameSentAtRef = useRef(0);
  const shouldReconnectRef = useRef(true);
  const appStateRef = useRef(appState);
  const registerState = appState.services.register;
  const verifyState = appState.services.verify;

  useEffect(() => {
    appStateRef.current = appState;
  }, [appState]);

  const streamIntervalMs = useMemo(() => {
    const configuredFps = Math.max(1, appState.camera.target_upload_fps || 16);
    const baseInterval = Math.max(50, Math.round(1000 / configuredFps));
    const backendLatency = appState.processed.last_processing_ms > 0 ? Math.round(appState.processed.last_processing_ms) : 0;
    return backendLatency > 0 ? Math.max(baseInterval, Math.min(260, backendLatency)) : baseInterval;
  }, [appState.camera.target_upload_fps, appState.processed.last_processing_ms]);

  const streamIntervalRef = useRef(streamIntervalMs);
  useEffect(() => {
    streamIntervalRef.current = streamIntervalMs;
  }, [streamIntervalMs]);

  function syncVideoStream(video: HTMLVideoElement | null, stream: MediaStream | null) {
    if (!video) {
      return;
    }

    if (video.srcObject !== stream) {
      video.srcObject = stream;
    }

    if (stream && video.paused) {
      void video.play().catch(() => {});
    }
  }

  function clearReconnectTimer() {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }

  function stopFrameLoop() {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }

  function queueReconnect(delayMs = 900) {
    if (!shouldReconnectRef.current || reconnectTimerRef.current !== null) {
      return;
    }

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      connectRealtime();
    }, delayMs);
  }

  function applyStreamMessage(payload: StreamMessage) {
    setAppState(payload.state);
    setRoiImageSrc(payload.images.roi || DEFAULT_ROI_IMAGE);
    setPreprocessedImageSrc(payload.images.preprocessed || DEFAULT_PREPROCESSED_IMAGE);
    if (payload.plot) {
      setPcaSnapshot(payload.plot);
    }
    if (payload.umap) {
      setUmapSnapshot(payload.umap);
    }
    if (payload.tsne) {
      setTsneSnapshot(payload.tsne);
    }
    const activeElementId =
      document.activeElement instanceof HTMLElement ? document.activeElement.id : "";
    if (payload.state.services.register.target_id !== null && activeElementId !== "register-target-id") {
      setRegisterTargetId(String(payload.state.services.register.target_id));
    }
    if (payload.state.services.verify.enabled) {
      setVerifyThreshold(String(payload.state.services.verify.threshold));
      setVerifyTopK(String(payload.state.services.verify.top_k));
    }
    if (payload.state.services.register.enabled) {
      setModeTab("register");
    } else if (payload.state.services.verify.enabled) {
      setModeTab("verify");
    }

    if (payload.state.backend.warmup_error) {
      setMessage(`ROI backend warmup error: ${payload.state.backend.warmup_error}`);
      return;
    }

    if (!payload.state.backend.warmup_ready) {
      setMessage("Preparing ROI backend...");
      return;
    }

    if (payload.state.processed.has_roi) {
      setMessage(`Realtime ROI stream live in ${payload.state.processed.last_processing_ms.toFixed(1)} ms`);
      return;
    }

    setMessage("Realtime backend connected. Show your palm to start ROI extraction.");
  }

  function sendSocketCommand(command: Record<string, unknown>) {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setMessage("Backend websocket is not connected.");
      return false;
    }

    socket.send(JSON.stringify(command));
    return true;
  }

  function connectRealtime() {
    const existing = socketRef.current;
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const socket = new WebSocket(BACKEND_WS_URL);
    socket.binaryType = "blob";
    socketRef.current = socket;

    socket.onopen = () => {
      setIsBackendConnected(true);
      setMessage("Realtime backend connected.");
      clearReconnectTimer();
    };

    socket.onmessage = (event) => {
      if (typeof event.data !== "string") {
        return;
      }

      try {
        applyStreamMessage(JSON.parse(event.data) as StreamMessage);
      } catch {
        setMessage("Received invalid realtime payload from backend.");
      }
    };

    socket.onerror = () => {
      setIsBackendConnected(false);
      setMessage("Realtime backend unavailable. Retrying...");
    };

    socket.onclose = () => {
      if (socketRef.current === socket) {
        socketRef.current = null;
      }
      setIsBackendConnected(false);
      queueReconnect();
    };
  }

  function startRegisterMode() {
    const normalized = registerTargetId.trim();
    if (!normalized) {
      setMessage("Register ID is required.");
      return;
    }

    const targetId = Number.parseInt(normalized, 10);
    if (Number.isNaN(targetId)) {
      setMessage("Register ID must be an integer.");
      return;
    }

    if (sendSocketCommand({ type: "register_start", targetId })) {
      setMessage(`Register mode requested for ID=${targetId}.`);
    }
  }

  function stopRegisterMode() {
    if (sendSocketCommand({ type: "register_stop" })) {
      setMessage("Register mode stopped.");
    }
  }

  function startVerifyMode() {
    const threshold = Number.parseFloat(verifyThreshold.trim());
    if (!Number.isFinite(threshold) || threshold < 0) {
      setMessage("Verify threshold must be a non-negative number.");
      return;
    }

    const topK = Number.parseInt(verifyTopK.trim(), 10);
    if (!Number.isFinite(topK) || topK <= 0) {
      setMessage("Verify top-k must be a positive integer.");
      return;
    }

    if (sendSocketCommand({ type: "verify_start", threshold, topK })) {
      setMessage(`Verify mode requested with threshold=${threshold.toFixed(4)} and top-${topK}.`);
    }
  }

  function stopVerifyMode() {
    if (sendSocketCommand({ type: "verify_stop" })) {
      setMessage("Verify mode stopped.");
    }
  }

  async function clearDatabase() {
    if (isClearingDatabase) {
      return;
    }

    if (!window.confirm("Clear all registered palm vectors from Qdrant?")) {
      return;
    }

    setIsClearingDatabase(true);
    try {
      const response = await fetch(`${BACKEND_HTTP_BASE}/api/database/clear`, {
        method: "POST",
      });
      const payload = (await response.json().catch(() => null)) as
        | {
            message?: string;
            detail?: string;
            plot?: ProjectionSnapshot;
            umap?: ProjectionSnapshot;
            tsne?: ProjectionSnapshot;
          }
        | null;

      if (!response.ok) {
        throw new Error(payload?.detail || payload?.message || "Failed to clear Qdrant vectors.");
      }

      if (payload?.plot) {
        setPcaSnapshot(payload.plot);
      }
      if (payload?.umap) {
        setUmapSnapshot(payload.umap);
      }
      if (payload?.tsne) {
        setTsneSnapshot(payload.tsne);
      }
      setMessage(payload?.message || "Cleared all registered vectors from Qdrant.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to clear Qdrant vectors.");
    } finally {
      setIsClearingDatabase(false);
    }
  }

  function refreshProjection(kind: Extract<ProjectionKind, "umap" | "tsne">) {
    if (sendSocketCommand({ type: "projection_refresh", kind })) {
      setMessage(`Refreshing ${projectionLabel(kind)}...`);
    }
  }

  async function sendFrameOverSocket() {
    if (!streamRef.current || !videoRef.current || !canvasRef.current || uploadInFlightRef.current) {
      return;
    }

    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setCameraMessage("Camera active, waiting for backend connection...");
      return;
    }

    if (socket.bufferedAmount > MAX_SOCKET_BUFFERED_BYTES) {
      return;
    }

    const video = videoRef.current;
    if (video.readyState < 2) {
      return;
    }

    const state = appStateRef.current;
    const width = video.videoWidth || state.camera.width;
    const height = video.videoHeight || state.camera.height;
    if (width <= 0 || height <= 0) {
      return;
    }

    const uploadWidth =
      state.backend.process_max_width > 0 ? Math.min(width, state.backend.process_max_width) : width;
    const uploadHeight = Math.max(1, Math.round((height * uploadWidth) / width));

    const canvas = canvasRef.current;
    canvas.width = uploadWidth;
    canvas.height = uploadHeight;

    const context = canvas.getContext("2d", { alpha: false });
    if (!context) {
      return;
    }

    context.drawImage(video, 0, 0, uploadWidth, uploadHeight);

    uploadInFlightRef.current = true;
    try {
      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob((value) => resolve(value), "image/png");
      });

      if (!blob) {
        return;
      }

      socket.send(blob);
    } catch (error) {
      setCameraMessage(error instanceof Error ? error.message : "Failed to stream frame");
    } finally {
      uploadInFlightRef.current = false;
    }
  }

  function runFrameLoop(now: number) {
    if (!streamRef.current) {
      animationFrameRef.current = null;
      return;
    }

    if (now - lastFrameSentAtRef.current >= streamIntervalRef.current) {
      lastFrameSentAtRef.current = now;
      void sendFrameOverSocket();
    }

    animationFrameRef.current = window.requestAnimationFrame(runFrameLoop);
  }

  function startFrameLoop() {
    stopFrameLoop();
    lastFrameSentAtRef.current = 0;
    animationFrameRef.current = window.requestAnimationFrame(runFrameLoop);
  }

  useEffect(() => {
    shouldReconnectRef.current = true;
    connectRealtime();

    return () => {
      shouldReconnectRef.current = false;
      clearReconnectTimer();
      stopFrameLoop();
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket) {
        socket.close();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const stream = streamRef.current;
    syncVideoStream(videoRef.current, stream);
    syncVideoStream(floatingVideoRef.current, stream);

    if (!isCameraRunning) {
      setShowFloatingCamera(false);
    }
  }, [isCameraRunning, showFloatingCamera]);

  useEffect(() => {
    if (!isCameraRunning) {
      setShowFloatingCamera(false);
      return;
    }

    const frame = cameraFrameRef.current;
    if (!frame) {
      return;
    }

    if (typeof IntersectionObserver === "undefined") {
      const browserWindow = globalThis as Window & typeof globalThis;
      const updateVisibility = () => {
        const rect = frame.getBoundingClientRect();
        const viewportHeight = browserWindow.innerHeight || document.documentElement.clientHeight || 0;
        const viewportWidth = browserWindow.innerWidth || document.documentElement.clientWidth || 0;
        const visibleWidth = Math.max(0, Math.min(rect.right, viewportWidth) - Math.max(rect.left, 0));
        const visibleHeight = Math.max(0, Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0));
        const visibleArea = visibleWidth * visibleHeight;
        const totalArea = Math.max(1, rect.width * rect.height);
        setShowFloatingCamera(visibleArea / totalArea < 0.5);
      };

      updateVisibility();
      browserWindow.addEventListener("scroll", updateVisibility, { passive: true });
      browserWindow.addEventListener("resize", updateVisibility);
      return () => {
        browserWindow.removeEventListener("scroll", updateVisibility);
        browserWindow.removeEventListener("resize", updateVisibility);
      };
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (!entry) {
          return;
        }
        setShowFloatingCamera(entry.intersectionRatio < 0.5);
      },
      {
        threshold: [0, 0.25, 0.5, 0.75, 1],
      },
    );

    observer.observe(frame);
    return () => {
      observer.disconnect();
    };
  }, [isCameraRunning]);

  useEffect(() => {
    if (!isCameraRunning) {
      return;
    }

    if (!isBackendConnected) {
      setCameraMessage("Camera active, waiting for backend connection...");
      return;
    }

    if (appState.processed.has_roi) {
      setCameraMessage(`ROI detected in ${appState.processed.last_processing_ms.toFixed(1)} ms`);
      return;
    }

    setCameraMessage("Streaming live video to backend...");
  }, [appState.processed.has_roi, appState.processed.last_processing_ms, isBackendConnected, isCameraRunning]);

  useEffect(() => {
    if (!registerState.enabled || !isBackendConnected) {
      return;
    }

    const normalized = registerTargetId.trim();
    if (!/^\d+$/.test(normalized)) {
      return;
    }

    const targetId = Number.parseInt(normalized, 10);
    if (registerState.target_id === targetId) {
      return;
    }

    const timerId = window.setTimeout(() => {
      if (sendSocketCommand({ type: "register_update", targetId })) {
        setMessage(`Register target updated to ID=${targetId}.`);
      }
    }, 250);

    return () => {
      window.clearTimeout(timerId);
    };
  }, [isBackendConnected, registerState.enabled, registerState.target_id, registerTargetId]);

  async function startCamera() {
    if (streamRef.current) {
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraMessage("This browser does not support camera capture.");
      return;
    }

    try {
      connectRealtime();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: CAMERA_IDEAL_WIDTH },
          height: { ideal: CAMERA_IDEAL_HEIGHT },
          frameRate: { ideal: CAMERA_TARGET_FRAME_RATE, max: CAMERA_MAX_FRAME_RATE },
        },
      });

      const videoTrack = stream.getVideoTracks()[0] ?? null;
      if (videoTrack) {
        try {
          await maximizeVideoTrackResolution(videoTrack);
        } catch {
          // Some browsers expose capabilities but reject max constraints; keep the opened stream.
        }
      }

      streamRef.current = stream;
      syncVideoStream(videoRef.current, stream);
      syncVideoStream(floatingVideoRef.current, stream);

      setIsCameraRunning(true);
      const resolutionLabel = getTrackResolutionLabel(videoTrack);
      if (resolutionLabel) {
        setCameraMessage(
          isBackendConnected
            ? `Camera active at ${resolutionLabel}`
            : `Camera active at ${resolutionLabel}, waiting for backend...`,
        );
      } else {
        setCameraMessage(isBackendConnected ? "Camera active" : "Camera active, waiting for backend...");
      }
      startFrameLoop();
    } catch (error) {
      setCameraMessage(error instanceof Error ? error.message : "Failed to start camera");
      await stopCamera();
    }
  }

  async function stopCamera() {
    stopFrameLoop();

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    uploadInFlightRef.current = false;
    setIsCameraRunning(false);
    setCameraMessage("Stopped");

    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop" }));
    }
  }

  const hasValidRegisterTarget = /^\d+$/.test(registerTargetId.trim());
  const registerDisabled = !isBackendConnected || !hasValidRegisterTarget || registerState.enabled;
  const parsedVerifyThreshold = Number.parseFloat(verifyThreshold.trim());
  const parsedVerifyTopK = Number.parseInt(verifyTopK.trim(), 10);
  const verifyDisabled =
    !isBackendConnected || !Number.isFinite(parsedVerifyThreshold) || parsedVerifyThreshold < 0 || !Number.isFinite(parsedVerifyTopK) || parsedVerifyTopK <= 0 || verifyState.enabled;
  const registerStatus = registerState.last_error
    ? registerState.last_error
    : registerState.enabled
      ? `Registering ID ${registerState.target_id} live. Change Register ID to switch target instantly.`
      : "Register mode is off.";
  const verifyConsensus = useMemo(() => getVerifyConsensus(verifyState.top_matches), [verifyState.top_matches]);
  const verifyStatus = buildVerifyStatus(verifyState);
  const verifyIsUnverified =
    !verifyState.last_error &&
    (verifyState.last_result || "").startsWith("No match:");
  const verifyOverlayClass = verifyStatusClass(verifyState);
  const activeModeValue = registerState.enabled ? "Register" : verifyState.enabled ? "Verify" : "Off";
  const activeModeDetail = registerState.enabled
    ? `Queued ${registerState.pending_count} | Inserted ${registerState.insert_count}`
    : verifyState.enabled
      ? `Top ${verifyState.top_k} | Pending ${verifyState.pending_count}`
      : "No live mode running";
  const activeProjection =
    plotTab === "pca" ? pcaSnapshot : plotTab === "umap" ? umapSnapshot : tsneSnapshot;
  const projectionRefreshDisabled =
    !isBackendConnected ||
    plotTab === "pca" ||
    Boolean(activeProjection && activeProjection.pending_count > 0);
  const projectionStatus = activeProjection?.error || activeProjection?.status || "Waiting for projection data.";
  const projectionStatusValue = activeProjection?.pending_count
    ? "Refreshing"
    : activeProjection?.error
      ? "Error"
      : activeProjection?.points.length
        ? "Ready"
        : "Idle";
  const projectionRefreshLabel =
    plotTab === "pca"
      ? ""
      : activeProjection?.pending_count
        ? `Refreshing ${projectionLabel(plotTab)}...`
        : `Refresh ${projectionLabel(plotTab)}`;

  const cards = [
    {
      label: "Frame Input",
      value: isCameraRunning ? "Streaming" : "Idle",
      detail: isBackendConnected ? "WebSocket live" : "Backend offline",
      active: isCameraRunning && isBackendConnected,
      icon: Camera,
    },
    {
      label: "ROI Output",
      value: appState.processed.has_roi ? "Detected" : "Waiting",
      detail: `Processed frame #${Math.max(appState.processed.last_processed_frame_id, 0)}`,
      active: appState.processed.has_roi,
      icon: ScanSearch,
    },
    {
      label: "Latency",
      value: `${appState.processed.last_processing_ms.toFixed(1)} ms`,
      detail: "ROI extraction to pushed output",
      active: appState.processed.last_processing_ms > 0,
      icon: Activity,
    },
    {
      label: "Mode",
      value: activeModeValue,
      detail: activeModeDetail,
      active: registerState.enabled || verifyState.enabled,
      icon: Cpu,
    },
  ];

  return (
    <main className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />
      <div className="sr-only" aria-live="polite">
        {message}
      </div>

      <section className="stats-grid">
        {cards.map((card) => {
          const Icon = card.icon;
          return (
            <article className={`stat-card ${card.active ? "is-active" : ""}`} key={card.label}>
              <div className="stat-top">
                <span className="stat-icon">
                  <Icon size={18} />
                </span>
                <span className="stat-label">{card.label}</span>
              </div>
              <div className="stat-value">{card.value}</div>
              <div className="stat-detail">{card.detail}</div>
            </article>
          );
        })}
      </section>

      <section className="workspace-stack">
        <div className="capture-grid">
          <article className="panel camera-panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Capture</p>
                <h2>Local Browser Camera</h2>
              </div>
              <Camera size={20} />
            </div>

            <div className="camera-actions">
              <button className="primary-button" onClick={() => void startCamera()} disabled={isCameraRunning}>
                Start Camera
              </button>
              <button className="secondary-button" onClick={() => void stopCamera()} disabled={!isCameraRunning}>
                Stop
              </button>
              <span className="camera-message">{cameraMessage}</span>
            </div>

            <div className="camera-frame" ref={cameraFrameRef}>
              <video ref={videoRef} autoPlay playsInline muted />
              <div className="camera-overlay camera-overlay-top">
                <div
                  className={`overlay-status-card ${registerState.last_error ? "is-error" : registerState.enabled ? "is-live" : ""}`}
                >
                  <span className="frame-label">Register</span>
                  <strong>{registerState.enabled ? `ID ${registerState.target_id}` : "Off"}</strong>
                  <span className="overlay-status-text">{registerStatus}</span>
                </div>
                <div
                  className={`overlay-status-card ${verifyOverlayClass}`}
                >
                  <span className="frame-label">Verify</span>
                  <strong>{verifyState.enabled ? "Active" : "Off"}</strong>
                  <span className="overlay-status-text">{verifyStatus}</span>
                </div>
              </div>

              <div className="camera-overlay camera-overlay-bottom">
                <div className="overlay-pill">{isCameraRunning ? "Camera Running" : "Camera Idle"}</div>
                <div className="overlay-pill soft">
                  {isBackendConnected ? "Realtime backend connected" : "Waiting for backend"}
                </div>
              </div>
            </div>

            <canvas ref={canvasRef} hidden />
          </article>

          <article className="panel outputs-panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Output</p>
                <h2>ROI Outputs</h2>
              </div>
              <ScanSearch size={20} />
            </div>

            <div className="outputs-meta">
              <span className="frame-label">Palm ROI + Preprocessed ROI</span>
              <div className={`status-chip ${appState.processed.has_roi ? "is-live" : ""}`}>
                {appState.processed.has_roi ? "Live" : "Waiting"}
              </div>
            </div>

            <div className="roi-stack">
              <div className="roi-tile">
                <div className="frame-label">Palm ROI</div>
                <img className="frame-image square" src={roiImageSrc} alt="Palm ROI" />
              </div>
              <div className="roi-tile">
                <div className="frame-label">Preprocessed ROI</div>
                <img className="frame-image square" src={preprocessedImageSrc} alt="Preprocessed palm ROI" />
              </div>
            </div>
          </article>
        </div>

        <div className="analysis-grid">
          <section className="panel plot-panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Plot</p>
                <h2>Embedding Maps</h2>
              </div>
              <Activity size={20} />
            </div>

            <div className="plot-layout">
              <div className="plot-toolbar">
                <div className="mode-tabs plot-tabs" role="tablist" aria-label="Projection tabs">
                  <button
                    type="button"
                    className={`mode-tab ${plotTab === "pca" ? "is-active" : ""}`}
                    onClick={() => setPlotTab("pca")}
                    role="tab"
                    aria-selected={plotTab === "pca"}
                  >
                    PCA
                  </button>
                  <button
                    type="button"
                    className={`mode-tab ${plotTab === "umap" ? "is-active" : ""}`}
                    onClick={() => setPlotTab("umap")}
                    role="tab"
                    aria-selected={plotTab === "umap"}
                  >
                    UMAP
                  </button>
                  <button
                    type="button"
                    className={`mode-tab ${plotTab === "tsne" ? "is-active" : ""}`}
                    onClick={() => setPlotTab("tsne")}
                    role="tab"
                    aria-selected={plotTab === "tsne"}
                  >
                    t-SNE
                  </button>
                </div>

                {plotTab !== "pca" ? (
                  <button
                    className="secondary-button plot-refresh-button"
                    onClick={() => refreshProjection(plotTab === "umap" ? "umap" : "tsne")}
                    disabled={projectionRefreshDisabled}
                  >
                    {projectionRefreshLabel}
                  </button>
                ) : null}
              </div>

              <div className="plot-stage">
                <ProjectionScatterPlot projection={activeProjection} kind={plotTab} />
              </div>

              <div className="plot-meta-grid">
                <div className="mode-card is-active">
                  <div className="mode-title">Registered Points</div>
                  <div className="mode-value">{appState.services.plot.count}</div>
                  <div className="mode-detail">
                    {appState.services.plot.max_points > 0
                      ? `Keeping latest ${appState.services.plot.max_points}`
                      : "Unlimited history"}
                  </div>
                </div>
                <div
                  className={`mode-card ${projectionStatusValue === "Refreshing" || Boolean(activeProjection?.error) ? "is-active" : ""}`}
                >
                  <div className="mode-title">{projectionLabel(plotTab)} Status</div>
                  <div className="mode-value">{projectionStatusValue}</div>
                  <div className="mode-detail">{projectionStatus}</div>
                </div>
              </div>
            </div>
          </section>

          <article className="panel mode-panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Modes</p>
                <h2>Register + Verify</h2>
              </div>
              <Cpu size={20} />
            </div>

            <div className="mode-tabs" role="tablist" aria-label="Mode tabs">
              <button
                type="button"
                className={`mode-tab ${modeTab === "register" ? "is-active" : ""}`}
                onClick={() => setModeTab("register")}
                role="tab"
                aria-selected={modeTab === "register"}
              >
                Register
              </button>
              <button
                type="button"
                className={`mode-tab ${modeTab === "verify" ? "is-active" : ""}`}
                onClick={() => setModeTab("verify")}
                role="tab"
                aria-selected={modeTab === "verify"}
              >
                Verify
              </button>
            </div>

            {modeTab === "register" ? (
              <div className="sidebar-section control-card tab-panel">
                <div className="section-header">
                  <div>
                    <div className="frame-label">Session</div>
                    <div className="section-title">Register Controls</div>
                  </div>
                  <div className={`status-chip ${registerState.enabled ? "is-live" : ""}`}>
                    {registerState.enabled ? "Active" : "Off"}
                  </div>
                </div>

                <div className="field">
                  <label className="frame-label" htmlFor="register-target-id">
                    Register ID
                  </label>
                  <input
                    id="register-target-id"
                    type="text"
                    inputMode="numeric"
                    value={registerTargetId}
                    onChange={(event) => setRegisterTargetId(event.target.value)}
                    placeholder="Enter subject ID"
                  />
                </div>

                <div className="action-row">
                  <button
                    className="primary-button"
                    onClick={startRegisterMode}
                    disabled={registerDisabled}
                  >
                    Start Register
                  </button>
                  <button className="secondary-button" onClick={stopRegisterMode} disabled={!registerState.enabled}>
                    Stop Register
                  </button>
                </div>

                <div className="metric-strip">
                  <div className="metric-chip">
                    <span className="frame-label">Target</span>
                    <strong>{registerState.target_id ?? "-"}</strong>
                  </div>
                  <div className="metric-chip">
                    <span className="frame-label">Queued</span>
                    <strong>{registerState.pending_count}</strong>
                  </div>
                  <div className="metric-chip">
                    <span className="frame-label">Inserted</span>
                    <strong>{registerState.insert_count}</strong>
                  </div>
                </div>

                <div className={`mode-card queue-card ${registerState.pending_count > 0 ? "is-active" : ""}`}>
                  <div className="mode-title">Register Queue</div>
                  <div className="mode-value">{registerState.pending_count}</div>
                  <div className="mode-detail">Capacity: {registerState.queue_maxsize}</div>
                </div>

                <button
                  className="secondary-button full-width-button"
                  onClick={() => void clearDatabase()}
                  disabled={!isBackendConnected || isClearingDatabase}
                >
                  {isClearingDatabase ? "Clearing Qdrant..." : "Clear All Registered Vectors"}
                </button>
              </div>
            ) : (
              <div className="sidebar-section control-card tab-panel">
                <div className="section-header">
                  <div>
                    <div className="frame-label">Verify</div>
                    <div className="section-title">Realtime Verify</div>
                  </div>
                  <div className={`status-chip ${verifyState.enabled ? "is-live" : ""}`}>
                    {verifyState.enabled ? "Active" : "Off"}
                  </div>
                </div>

                <div className="form-grid">
                  <div className="field">
                    <label className="frame-label" htmlFor="verify-threshold">
                      Threshold
                    </label>
                    <input
                      id="verify-threshold"
                      type="text"
                      inputMode="decimal"
                      value={verifyThreshold}
                      onChange={(event) => setVerifyThreshold(event.target.value)}
                      placeholder="35.0"
                    />
                  </div>
                  <div className="field">
                    <label className="frame-label" htmlFor="verify-top-k">
                      Top K
                    </label>
                    <input
                      id="verify-top-k"
                      type="text"
                      inputMode="numeric"
                      value={verifyTopK}
                      onChange={(event) => setVerifyTopK(event.target.value)}
                      placeholder="10"
                    />
                  </div>
                </div>

                <div className="action-row">
                  <button className="primary-button" onClick={startVerifyMode} disabled={verifyDisabled}>
                    Start Verify
                  </button>
                  <button className="secondary-button" onClick={stopVerifyMode} disabled={!verifyState.enabled}>
                    Stop Verify
                  </button>
                </div>

                <div className="metric-strip">
                  <div className="metric-chip">
                    <span className="frame-label">Threshold</span>
                    <strong>{verifyState.threshold.toFixed(2)}</strong>
                  </div>
                  <div className="metric-chip">
                    <span className="frame-label">Top K</span>
                    <strong>{verifyState.top_k}</strong>
                  </div>
                  <div className="metric-chip">
                    <span className="frame-label">Queued</span>
                    <strong>{verifyState.pending_count}</strong>
                  </div>
                </div>

                <div className={`mode-card queue-card ${verifyState.pending_count > 0 ? "is-active" : ""}`}>
                  <div className="mode-title">Verify Queue</div>
                  <div className="mode-value">{verifyState.pending_count}</div>
                  <div className="mode-detail">Capacity: {verifyState.queue_maxsize}</div>
                </div>

                <div className="verify-results-card">
                  <div className="section-header">
                    <div>
                      <div className="frame-label">Matches</div>
                      <div className="section-title">Top-K Verify Results</div>
                    </div>
                  </div>
                  {verifyState.top_matches.length > 0 ? (
                    <div className="verify-results">
                      {verifyState.top_matches.map((match) => (
                        <div
                          className={`verify-row ${
                            verifyConsensus && !verifyIsUnverified && match.subject_id === verifyConsensus.subjectId ? "is-primary" : ""
                          }`}
                          key={`${match.point_id}-${match.rank}`}
                        >
                          <span className="verify-rank">#{match.rank}</span>
                          <span className="verify-subject">ID {match.subject_id}</span>
                          <span className="verify-distance">{match.distance.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="verify-empty muted">No verify results yet.</div>
                  )}
                </div>
              </div>
            )}
          </article>
        </div>
      </section>

      {isCameraRunning && showFloatingCamera ? (
        <>
          <div className="floating-status-shell" aria-hidden="true">
            <div
              className={`overlay-status-card overlay-status-card-compact ${registerState.last_error ? "is-error" : registerState.enabled ? "is-live" : ""}`}
            >
              <span className="frame-label">Register</span>
              <strong>{registerState.enabled ? `ID ${registerState.target_id}` : "Off"}</strong>
              <span className="overlay-status-text">{registerStatus}</span>
            </div>
            <div
              className={`overlay-status-card overlay-status-card-compact ${verifyOverlayClass}`}
            >
              <span className="frame-label">Verify</span>
              <strong>{verifyState.enabled ? "Active" : "Off"}</strong>
              <span className="overlay-status-text">{verifyStatus}</span>
            </div>
          </div>

          <div className="floating-camera-shell" aria-hidden="true">
            <div className="floating-camera-label">Live Camera</div>
            <div className="floating-camera-frame">
              <video ref={floatingVideoRef} autoPlay playsInline muted />
            </div>
          </div>
        </>
      ) : null}
    </main>
  );
}

export default App;
