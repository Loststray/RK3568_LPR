import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

WEB_DIR = Path(__file__).resolve().parent
SRC_DIR = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATE_DIR = WEB_DIR / "templates"
UPLOAD_DIR = WEB_DIR / "runtime" / "uploads"
RESULT_DIR = STATIC_DIR / "generated" / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from test import run_image_file, run_video_file

ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"},
}

DEFAULT_CONF_THRES = 0.5

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


def parse_conf_thres():
    raw_value = request.form.get("conf_thres", str(DEFAULT_CONF_THRES)).strip()
    if not raw_value:
        return DEFAULT_CONF_THRES

    try:
        conf_thres = float(raw_value)
    except ValueError as exc:
        raise ValueError("检测阈值必须是数字") from exc

    if not 0.01 <= conf_thres <= 1.0:
        raise ValueError("检测阈值必须在 0.01 到 1.0 之间")
    return conf_thres


def build_media_name(filename, media_type):
    sanitized = secure_filename(Path(filename).stem)
    return sanitized or media_type


def build_static_url(path):
    relative_path = path.relative_to(STATIC_DIR).as_posix()
    return url_for("static", filename=relative_path)


@app.get("/")
def index():
    ensure_runtime_dirs()
    return render_template("index.html")


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
        conf_thres = parse_conf_thres()
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


@app.errorhandler(413)
def file_too_large(_error):
    return error_response("上传文件过大，请控制在 512MB 以内", status_code=413)


if __name__ == "__main__":
    ensure_runtime_dirs()
    app.run(host="0.0.0.0", port=5000, debug=False)
