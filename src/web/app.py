import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.request import Request, urlopen
from urllib.parse import urlparse, urlunparse
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, stream_with_context, url_for
from werkzeug.utils import secure_filename

WEB_DIR = Path(__file__).resolve().parent
SRC_DIR = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATE_DIR = WEB_DIR / "templates"
UPLOAD_DIR = WEB_DIR / "runtime" / "uploads"
RESULT_DIR = STATIC_DIR / "generated" / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_process import cv2ImgAddText, detect_and_recognize, draw_detections
from test import run_image_file, run_video_file

ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"},
}

DEFAULT_CONF_THRES = 0.5
DEFAULT_CAMERA_INPUT = "http://192.168.0.16:8889/lubancat/"
DEFAULT_CAMERA_PORT = 8889
DEFAULT_MEDIAMTX_RTSP_PORT = 8554
DEFAULT_CAMERA_FRAME_INTERVAL = 3
DEFAULT_CAMERA_JPEG_QUALITY = 85
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5000
CAMERA_SESSIONS = {}
CAMERA_SESSIONS_LOCK = threading.Lock()
ACTIVE_CAMERA_STREAMS = 0
ACTIVE_CAMERA_STREAMS_LOCK = threading.Lock()
APP_SHUTDOWN_EVENT = threading.Event()
STREAM_RESPONSE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024


def ensure_runtime_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def error_response(message, status_code=400):
    response = jsonify({"ok": False, "message": message})
    response.status_code = status_code
    return response


def get_extension(filename):
    return Path(filename).suffix.lower()


def is_allowed_file(filename, media_type):
    return get_extension(filename) in ALLOWED_EXTENSIONS[media_type]


def parse_conf_thres(value):
    raw_value = str(value).strip() if value is not None else ""
    if not raw_value:
        return DEFAULT_CONF_THRES

    try:
        conf_thres = float(raw_value)
    except ValueError as exc:
        raise ValueError("检测阈值必须是数字") from exc

    if not 0.01 <= conf_thres <= 1.0:
        raise ValueError("检测阈值必须在 0.01 到 1.0 之间")
    return conf_thres


def parse_form_conf_thres():
    return parse_conf_thres(request.form.get("conf_thres", str(DEFAULT_CONF_THRES)))


def build_media_name(filename, media_type):
    sanitized = secure_filename(Path(filename).stem)
    return sanitized or media_type


def build_static_url(path):
    relative_path = path.relative_to(STATIC_DIR).as_posix()
    return url_for("static", filename=relative_path)


def normalize_camera_source(raw_value):
    value = (raw_value or "").strip()
    if not value:
        raise ValueError("请输入摄像头 IP")

    candidate = value if "://" in value else f"http://{value}"
    parsed = urlparse(candidate)
    if not parsed.hostname:
        raise ValueError("摄像头 IP 或 URL 格式不正确")

    scheme = parsed.scheme or "http"
    port = parsed.port or DEFAULT_CAMERA_PORT
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    auth_prefix = ""
    if parsed.username:
        auth_prefix = parsed.username
        if parsed.password:
            auth_prefix = f"{auth_prefix}:{parsed.password}"
        auth_prefix = f"{auth_prefix}@"

    path = parsed.path or ""
    query = parsed.query or ""
    netloc = f"{auth_prefix}{hostname}:{port}"
    return urlunparse((scheme, netloc, path, "", query, ""))


def fetch_http_source_metadata(source_url, timeout=5):
    req = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as response:
        preview = response.read(4096)
        return {
            "status": int(response.status),
            "content_type": response.headers.get_content_type(),
            "server": response.headers.get("Server", ""),
            "preview_text": preview.decode("utf-8", "replace"),
        }


def is_mediatx_webrtc_page(source_url, metadata):
    parsed = urlparse(source_url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False

    preview_text = metadata.get("preview_text", "")
    server_name = metadata.get("server", "").lower()
    return (
        metadata.get("content_type") == "text/html"
        and (
            "mediamtx" in server_name
            or "MediaMTXWebRTCReader" in preview_text
            or "new URL('whep'" in preview_text
        )
    )


def derive_mediatx_capture_candidates(source_url):
    parsed = urlparse(source_url)
    hostname = parsed.hostname
    if not hostname:
        return []

    path = parsed.path.rstrip("/")
    if not path:
        return []

    auth_prefix = ""
    if parsed.username:
        auth_prefix = parsed.username
        if parsed.password:
            auth_prefix = f"{auth_prefix}:{parsed.password}"
        auth_prefix = f"{auth_prefix}@"

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    rtsp_netloc = f"{auth_prefix}{hostname}:{DEFAULT_MEDIAMTX_RTSP_PORT}"
    return [
        urlunparse(("rtsp", rtsp_netloc, path, "", "", "")),
    ]


def build_capture_candidates(source_url):
    parsed = urlparse(source_url)
    candidates = []

    if parsed.scheme.lower() == "rtsp":
        candidates.append(source_url)
    elif parsed.scheme.lower() in {"http", "https"}:
        metadata = None
        try:
            metadata = fetch_http_source_metadata(source_url)
        except Exception:
            metadata = None

        if metadata and is_mediatx_webrtc_page(source_url, metadata):
            candidates.extend(derive_mediatx_capture_candidates(source_url))

        candidates.append(source_url)
    else:
        candidates.append(source_url)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def open_stream_capture(source_url):
    scheme = urlparse(source_url).scheme.lower()
    if scheme == "rtsp":
        return cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source_url)


def probe_camera_source(source_url):
    candidates = build_capture_candidates(source_url)
    if not candidates:
        raise RuntimeError("无法从当前输入推导出可识别的视频流地址")

    errors = []
    for candidate in candidates:
        cap = open_stream_capture(candidate)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频流: {candidate}")

            for _ in range(30):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    height, width = frame.shape[:2]
                    return {
                        "width": int(width),
                        "height": int(height),
                        "capture_url": candidate,
                    }
            raise RuntimeError(f"视频流已连接，但没有读取到有效帧: {candidate}")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            cap.release()

    raise RuntimeError("；".join(errors))


def serialize_detections(detections):
    serialized = []
    for det in detections:
        serialized.append(
            {
                "bbox": [int(value) for value in det["bbox"]],
                "plate": det["plate"] or "",
                "conf": float(det["conf"]),
            }
        )
    return serialized


def summarize_counter(counter):
    return [{"plate": plate, "count": int(count)} for plate, count in counter.most_common()]


def create_camera_session(camera_input, source_url, probe_info):
    session_id = uuid.uuid4().hex
    session = {
        "session_id": session_id,
        "camera_input": camera_input,
        "source_url": probe_info["capture_url"],
        "browser_url": source_url,
        "source_label": probe_info["capture_url"].replace("http://", "").replace("https://", ""),
        "width": int(probe_info["width"]),
        "height": int(probe_info["height"]),
        "frames_processed": 0,
        "recognized_plates": [],
        "last_detections": [],
        "last_error": None,
        "active": False,
        "updated_at": time.time(),
    }
    with CAMERA_SESSIONS_LOCK:
        CAMERA_SESSIONS[session_id] = session
    return session


def get_camera_session(session_id):
    with CAMERA_SESSIONS_LOCK:
        session = CAMERA_SESSIONS.get(session_id)
        if session is None:
            return None
        return dict(session)


def update_camera_session(session_id, **updates):
    with CAMERA_SESSIONS_LOCK:
        session = CAMERA_SESSIONS.get(session_id)
        if session is None:
            return None
        session.update(updates)
        session["updated_at"] = time.time()
        return dict(session)


def mark_all_camera_sessions_inactive():
    with CAMERA_SESSIONS_LOCK:
        current_time = time.time()
        for session in CAMERA_SESSIONS.values():
            session["active"] = False
            session["updated_at"] = current_time


def increment_active_camera_streams():
    global ACTIVE_CAMERA_STREAMS
    with ACTIVE_CAMERA_STREAMS_LOCK:
        ACTIVE_CAMERA_STREAMS += 1


def decrement_active_camera_streams():
    global ACTIVE_CAMERA_STREAMS
    with ACTIVE_CAMERA_STREAMS_LOCK:
        ACTIVE_CAMERA_STREAMS = max(ACTIVE_CAMERA_STREAMS - 1, 0)


def wait_for_camera_streams(timeout=3.0, poll_interval=0.05):
    deadline = time.time() + max(timeout, 0.0)
    while time.time() < deadline:
        with ACTIVE_CAMERA_STREAMS_LOCK:
            if ACTIVE_CAMERA_STREAMS == 0:
                return True
        time.sleep(poll_interval)

    with ACTIVE_CAMERA_STREAMS_LOCK:
        return ACTIVE_CAMERA_STREAMS == 0


def serialize_camera_session(session):
    return {
        "sessionId": session["session_id"],
        "cameraInput": session["camera_input"],
        "sourceUrl": session["source_url"],
        "browserUrl": session["browser_url"],
        "sourceLabel": session["source_label"],
        "sourceWidth": int(session["width"]),
        "sourceHeight": int(session["height"]),
        "active": bool(session["active"]),
        "framesProcessed": int(session["frames_processed"]),
        "recognizedPlates": list(session["recognized_plates"]),
        "lastDetections": list(session["last_detections"]),
        "lastError": session["last_error"],
        "updatedAt": float(session["updated_at"]),
    }


class GracefulThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True


def create_info_frame(message, width=960, height=540):
    frame = np.full((height, width, 3), 242, dtype=np.uint8)
    frame = cv2ImgAddText(frame, "RK3568 LPR Camera Stream", (28, 28), textColor=(45, 36, 24), textSize=32)

    current_y = 100
    for line in message.splitlines():
        frame = cv2ImgAddText(frame, line[:48], (28, current_y), textColor=(139, 47, 24), textSize=28)
        current_y += 48
    return frame


def encode_mjpeg_frame(frame):
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), DEFAULT_CAMERA_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("无法将摄像头帧编码为 JPEG")
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + encoded.tobytes()
        + b"\r\n"
    )


def camera_stream_generator(session_id, detect_mode=False, conf_thres=DEFAULT_CONF_THRES, frame_interval=DEFAULT_CAMERA_FRAME_INTERVAL):
    if APP_SHUTDOWN_EVENT.is_set():
        return

    session = get_camera_session(session_id)
    if session is None:
        yield encode_mjpeg_frame(create_info_frame("摄像头会话不存在"))
        return

    cap = open_stream_capture(session["source_url"])
    if not cap.isOpened():
        update_camera_session(
            session_id,
            active=False,
            last_error=f"无法打开摄像头流: {session['source_url']}",
        )
        yield encode_mjpeg_frame(create_info_frame("无法打开摄像头流"))
        cap.release()
        return

    increment_active_camera_streams()
    frame_interval = max(int(frame_interval), 1)
    last_detections = []
    recognized_counter = Counter()
    fps = 0.0
    previous_time = time.time()

    if detect_mode:
        update_camera_session(
            session_id,
            active=True,
            frames_processed=0,
            recognized_plates=[],
            last_detections=[],
            last_error=None,
        )

    try:
        frame_idx = 0
        while not APP_SHUTDOWN_EVENT.is_set():
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                error_message = "摄像头流中断，无法继续读取帧"
                if APP_SHUTDOWN_EVENT.is_set():
                    break
                if detect_mode:
                    update_camera_session(session_id, active=False, last_error=error_message)
                try:
                    yield encode_mjpeg_frame(create_info_frame(error_message))
                except (GeneratorExit, ConnectionError, BrokenPipeError):
                    pass
                break

            frame_idx += 1
            display = frame.copy()

            if detect_mode:
                should_detect = frame_idx == 1 or frame_idx % frame_interval == 0
                if should_detect:
                    last_detections = detect_and_recognize(frame, conf_thres=conf_thres)
                    current_frame_plates = set()
                    for det in last_detections:
                        plate = det["plate"].strip() if det["plate"] else ""
                        if plate and plate not in current_frame_plates:
                            recognized_counter[plate] += 1
                            current_frame_plates.add(plate)

                display = draw_detections(display, last_detections)
                current_time = time.time()
                delta = current_time - previous_time
                previous_time = current_time
                if delta > 0:
                    fps = fps * 0.9 + (1.0 / delta) * 0.1

                # overlay_text = f"camera frame={frame_idx} detect={int(should_detect)} fps={fps:.1f}"
                # display = cv2ImgAddText(display, overlay_text, (10, 10), textColor=(255, 255, 0), textSize=24)
                update_camera_session(
                    session_id,
                    active=True,
                    frames_processed=int(frame_idx),
                    recognized_plates=summarize_counter(recognized_counter),
                    last_detections=serialize_detections(last_detections),
                    last_error=None,
                )
            else:
                display = cv2ImgAddText(display, f"source frame={frame_idx}", (10, 10), textColor=(255, 255, 0), textSize=24)

            try:
                yield encode_mjpeg_frame(display)
            except (GeneratorExit, ConnectionError, BrokenPipeError):
                break
    finally:
        cap.release()
        decrement_active_camera_streams()
        if detect_mode:
            update_camera_session(session_id, active=False)


def build_stream_response(generator):
    return Response(
        stream_with_context(generator),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=STREAM_RESPONSE_HEADERS,
    )


@app.get("/")
def index():
    ensure_runtime_dirs()
    return render_template("index.html", default_camera_ip=DEFAULT_CAMERA_INPUT)


@app.post("/api/detect")
def detect():
    ensure_runtime_dirs()

    media_type = request.form.get("media_type", "").strip().lower()
    if media_type not in ALLOWED_EXTENSIONS:
        return error_response("请先上传图片或视频")

    media_file = request.files.get("file")
    if media_file is None or not media_file.filename:
        return error_response("未接收到上传文件")

    if not is_allowed_file(media_file.filename, media_type):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS[media_type]))
        return error_response(f"不支持的文件格式，可用格式：{allowed}")

    try:
        conf_thres = parse_form_conf_thres()
    except ValueError as exc:
        return error_response(str(exc))

    request_id = uuid.uuid4().hex
    media_name = build_media_name(media_file.filename, media_type)
    upload_ext = get_extension(media_file.filename)
    upload_path = UPLOAD_DIR / f"{request_id}_{media_name}{upload_ext}"
    media_file.save(upload_path)

    try:
        if media_type == "image":
            result_path = RESULT_DIR / f"{media_name}_{request_id}_detected.jpg"
            result = run_image_file(
                image_path=str(upload_path),
                output_path=str(result_path),
                conf_thres=conf_thres,
            )
            return jsonify(
                {
                    "ok": True,
                    "mediaType": "image",
                    "message": "图片检测完成",
                    "resultUrl": build_static_url(Path(result["output_path"])),
                    "detections": result["detections"],
                    "recognizedPlates": result["recognized_plates"],
                }
            )

        result_path = RESULT_DIR / f"{media_name}_{request_id}_detected.mp4"
        result = run_video_file(
            video_path=str(upload_path),
            output_path=str(result_path),
            conf_thres=conf_thres,
        )
        return jsonify(
            {
                "ok": True,
                "mediaType": "video",
                "message": "视频检测完成",
                "resultUrl": build_static_url(Path(result["output_path"])),
                "recognizedPlates": result["recognized_plates"],
                "framesProcessed": result["frames_processed"],
                "h264Ready": result["h264_ready"],
                "transcodeWarning": result["transcode_warning"],
                "encoder": result["encoder"],
            }
        )
    except Exception as exc:
        return error_response(f"检测失败：{exc}", status_code=500)


@app.post("/api/camera/connect")
def connect_camera():
    payload = request.get_json(silent=True) or {}
    camera_input = payload.get("ip", DEFAULT_CAMERA_INPUT)

    try:
        source_url = normalize_camera_source(camera_input)
        probe_info = probe_camera_source(source_url)
    except (RuntimeError, ValueError) as exc:
        return error_response(f"摄像头连接失败：{exc}", status_code=400)

    session = create_camera_session(camera_input=str(camera_input).strip(), source_url=source_url, probe_info=probe_info)
    return jsonify(
        {
            "ok": True,
            "message": "摄像头连接成功",
            "sessionId": session["session_id"],
            "sourceLabel": session["source_label"],
            "sourceUrl": session["source_url"],
            "browserUrl": session["browser_url"],
            "sourceWidth": session["width"],
            "sourceHeight": session["height"],
            "sourceStreamUrl": url_for("camera_raw_stream", session_id=session["session_id"]),
            "detectStreamUrl": url_for("camera_detect_stream", session_id=session["session_id"]),
            "statusUrl": url_for("camera_status", session_id=session["session_id"]),
        }
    )


@app.get("/api/camera/raw-stream")
def camera_raw_stream():
    session_id = request.args.get("session_id", "").strip()
    if not session_id or get_camera_session(session_id) is None:
        return error_response("摄像头会话不存在", status_code=404)
    return build_stream_response(camera_stream_generator(session_id, detect_mode=False))


@app.get("/api/camera/detect-stream")
def camera_detect_stream():
    session_id = request.args.get("session_id", "").strip()
    if not session_id or get_camera_session(session_id) is None:
        return error_response("摄像头会话不存在", status_code=404)

    try:
        conf_thres = parse_conf_thres(request.args.get("conf_thres", str(DEFAULT_CONF_THRES)))
    except ValueError as exc:
        return error_response(str(exc), status_code=400)

    frame_interval = request.args.get("frame_interval", str(DEFAULT_CAMERA_FRAME_INTERVAL)).strip()
    try:
        frame_interval = max(int(frame_interval), 1)
    except ValueError:
        return error_response("frame_interval 必须是正整数", status_code=400)

    return build_stream_response(
        camera_stream_generator(
            session_id,
            detect_mode=True,
            conf_thres=conf_thres,
            frame_interval=frame_interval,
        )
    )


@app.get("/api/camera/status")
def camera_status():
    session_id = request.args.get("session_id", "").strip()
    session = get_camera_session(session_id)
    if session is None:
        return error_response("摄像头会话不存在", status_code=404)
    return jsonify({"ok": True, "camera": serialize_camera_session(session)})


@app.errorhandler(413)
def file_too_large(_error):
    return error_response("上传文件过大，请控制在 512MB 以内", status_code=413)


def run_web_server(host=DEFAULT_SERVER_HOST, port=DEFAULT_SERVER_PORT):
    server = make_server(
        host,
        port,
        app,
        server_class=GracefulThreadingWSGIServer,
        handler_class=WSGIRequestHandler,
    )
    print(f"* Running On http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        APP_SHUTDOWN_EVENT.set()
        mark_all_camera_sessions_inactive()
        wait_for_camera_streams()
        server.server_close()


if __name__ == "__main__":
    ensure_runtime_dirs()
    run_web_server()
