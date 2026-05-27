const elements = {
    uploadImageBtn: document.getElementById("upload-image-btn"),
    uploadVideoBtn: document.getElementById("upload-video-btn"),
    connectCameraBtn: document.getElementById("connect-camera-btn"),
    detectBtn: document.getElementById("detect-btn"),
    imageInput: document.getElementById("image-input"),
    videoInput: document.getElementById("video-input"),
    cameraIpInput: document.getElementById("camera-ip-input"),
    sourcePreview: document.getElementById("source-preview"),
    resultPreview: document.getElementById("result-preview"),
    sourceBadge: document.getElementById("source-badge"),
    resultBadge: document.getElementById("result-badge"),
    statusText: document.getElementById("status-text"),
    summaryList: document.getElementById("summary-list"),
    detailList: document.getElementById("detail-list"),
    confThres: document.getElementById("conf-thres"),
    confThresValue: document.getElementById("conf-thres-value"),
};

const state = {
    file: null,
    mediaType: null,
    sourceObjectUrl: null,
    cameraSessionId: null,
    cameraSourceStreamUrl: null,
    cameraDetectStreamUrl: null,
    cameraStatusUrl: null,
    cameraStatusTimer: null,
};

function setStatus(message, tone) {
    elements.statusText.textContent = message;
    elements.statusText.className = `status-text status-${tone}`;
}

function setEmpty(container, message, extraClass = "") {
    container.className = extraClass ? `${extraClass} empty-list` : "empty-list";
    container.innerHTML = `<p>${message}</p>`;
}

function releaseSourcePreviewUrl() {
    if (state.sourceObjectUrl) {
        URL.revokeObjectURL(state.sourceObjectUrl);
        state.sourceObjectUrl = null;
    }
}

function clearCameraPolling() {
    if (state.cameraStatusTimer) {
        clearInterval(state.cameraStatusTimer);
        state.cameraStatusTimer = null;
    }
}

function clearCameraState() {
    clearCameraPolling();
    state.cameraSessionId = null;
    state.cameraSourceStreamUrl = null;
    state.cameraDetectStreamUrl = null;
    state.cameraStatusUrl = null;
}

function resetResultPanels() {
    clearCameraPolling();
    elements.resultBadge.textContent = "等待检测";
    elements.resultPreview.className = "media-stage empty-stage";
    elements.resultPreview.innerHTML = "<p>点击“检测车牌”后展示标注结果</p>";
    setEmpty(elements.summaryList, "检测完成后会列出识别到的车牌和出现次数", "summary-list");
    setEmpty(elements.detailList, "图片模式会列出检测框坐标和置信度，视频与摄像头模式显示帧处理与实时识别信息", "detail-list");
}

function resetSourcePanel() {
    releaseSourcePreviewUrl();
    elements.sourcePreview.className = "media-stage empty-stage";
    elements.sourcePreview.innerHTML = "<p>上传后会在这里预览原始图片或视频，接入摄像头后会显示实时源流</p>";
    elements.sourceBadge.textContent = "未选择";
}

function resetFileInputs() {
    elements.imageInput.value = "";
    elements.videoInput.value = "";
}

function formatFileMeta(file, mediaType) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    return `${mediaType === "image" ? "图片" : "视频"} | ${file.name} | ${sizeMb} MB`;
}

function createMediaMarkup(src, mediaType, alt = "媒体预览") {
    if (mediaType === "image") {
        return `<img src="${src}" alt="${alt}">`;
    }
    if (mediaType === "camera") {
        return `<img src="${src}" alt="${alt}" class="stream-frame">`;
    }
    return `<video src="${src}" controls playsinline></video>`;
}

function renderSourcePreview(file, mediaType) {
    releaseSourcePreviewUrl();
    state.sourceObjectUrl = URL.createObjectURL(file);
    elements.sourcePreview.className = "media-stage";
    elements.sourcePreview.innerHTML = createMediaMarkup(state.sourceObjectUrl, mediaType, "原始媒体预览");
    elements.sourceBadge.textContent = formatFileMeta(file, mediaType);
}

function renderResultPreview(resultUrl, mediaType) {
    const cacheBustedUrl = `${resultUrl}?t=${Date.now()}`;
    elements.resultPreview.className = "media-stage";
    elements.resultPreview.innerHTML = createMediaMarkup(cacheBustedUrl, mediaType, "检测结果预览");
    elements.resultBadge.textContent = mediaType === "image" ? "图片已标注" : "视频已生成";
}

function renderSummary(recognizedPlates) {
    if (!recognizedPlates || recognizedPlates.length === 0) {
        setEmpty(elements.summaryList, "没有识别到明确的车牌文本", "summary-list");
        return;
    }

    elements.summaryList.className = "summary-list";
    elements.summaryList.innerHTML = recognizedPlates
        .map(
            (item) => `
                <div class="plate-chip">
                    <strong>${item.plate}</strong>
                    <span>出现 ${item.count} 次</span>
                </div>
            `
        )
        .join("");
}

function renderImageDetails(detections) {
    if (!detections || detections.length === 0) {
        setEmpty(elements.detailList, "图片中未检测到车牌框", "detail-list");
        return;
    }

    elements.detailList.className = "detail-list";
    elements.detailList.innerHTML = detections
        .map((item, index) => {
            const [x1, y1, x2, y2] = item.bbox;
            const label = item.plate || "未识别";
            return `
                <div class="detail-item">
                    <strong>#${index + 1} ${label}</strong>
                    <span>bbox: [${x1}, ${y1}, ${x2}, ${y2}] | conf: ${item.conf.toFixed(2)}</span>
                </div>
            `;
        })
        .join("");
}

function renderVideoDetails(payload) {
    const parts = [`处理帧数 ${payload.framesProcessed}`];
    if (payload.encoder) {
        parts.push(`编码器 ${payload.encoder}`);
    }
    if (!payload.h264Ready && payload.transcodeWarning) {
        parts.push("已保留原始 MP4 输出");
    }

    elements.detailList.className = "detail-list";
    elements.detailList.innerHTML = `
        <div class="detail-item">
            <strong>视频检测完成</strong>
            <span>${parts.join(" | ")}</span>
        </div>
        ${
            payload.transcodeWarning
                ? `<p class="video-note">${payload.transcodeWarning}</p>`
                : ""
        }
    `;
}

function renderCameraDetails(camera) {
    const detailItems = [
        `
            <div class="detail-item">
                <strong>实时识别状态</strong>
                <span>${camera.active ? "检测中" : "待机"} | 已处理 ${camera.framesProcessed} 帧</span>
            </div>
        `,
        `
            <div class="detail-item">
                <strong>流地址</strong>
                <span>${camera.sourceUrl}</span>
            </div>
        `,
        `
            <div class="detail-item">
                <strong>分辨率</strong>
                <span>${camera.sourceWidth} x ${camera.sourceHeight}</span>
            </div>
        `,
    ];

    if (camera.lastDetections && camera.lastDetections.length > 0) {
        camera.lastDetections.forEach((item, index) => {
            const [x1, y1, x2, y2] = item.bbox;
            const label = item.plate || "未识别";
            detailItems.push(`
                <div class="detail-item">
                    <strong>最近检测 #${index + 1} ${label}</strong>
                    <span>bbox: [${x1}, ${y1}, ${x2}, ${y2}] | conf: ${item.conf.toFixed(2)}</span>
                </div>
            `);
        });
    }

    if (camera.lastError) {
        detailItems.push(`<p class="video-note">${camera.lastError}</p>`);
    }

    elements.detailList.className = "detail-list";
    elements.detailList.innerHTML = detailItems.join("");
}

function switchToFileMode(file, mediaType) {
    clearCameraState();
    state.file = file;
    state.mediaType = mediaType;
    resetFileInputs();
    renderSourcePreview(file, mediaType);
    resetResultPanels();
    elements.detectBtn.disabled = false;
    setStatus(`已选择${mediaType === "image" ? "图片" : "视频"}，可以开始检测`, "idle");
}

function switchToCameraMode(payload) {
    releaseSourcePreviewUrl();
    state.file = null;
    state.mediaType = "camera";
    state.cameraSessionId = payload.sessionId;
    state.cameraSourceStreamUrl = payload.sourceStreamUrl;
    state.cameraDetectStreamUrl = payload.detectStreamUrl;
    state.cameraStatusUrl = payload.statusUrl;
    resetFileInputs();
    resetResultPanels();

    elements.sourcePreview.className = "media-stage";
    elements.sourcePreview.innerHTML = createMediaMarkup(
        `${payload.sourceStreamUrl}&t=${Date.now()}`,
        "camera",
        "摄像头原始流"
    );
    elements.sourceBadge.textContent = `摄像头 | ${payload.sourceLabel}`;
    elements.detectBtn.disabled = false;
    setStatus(`摄像头已连接：${payload.sourceLabel}，可以开始实时识别`, "success");
}

async function connectCamera() {
    const cameraIp = elements.cameraIpInput.value.trim();
    if (!cameraIp) {
        setStatus("请输入摄像头 IP", "error");
        return;
    }

    elements.connectCameraBtn.disabled = true;
    elements.detectBtn.disabled = true;
    state.file = null;
    state.mediaType = null;
    clearCameraState();
    setStatus(`正在连接摄像头流 ${cameraIp} ...`, "busy");

    try {
        const response = await fetch("/api/camera/connect", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ ip: cameraIp }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            throw new Error(payload.message || "摄像头连接失败");
        }
        switchToCameraMode(payload);
    } catch (error) {
        resetSourcePanel();
        resetResultPanels();
        setStatus(error.message, "error");
    } finally {
        elements.connectCameraBtn.disabled = false;
    }
}

async function runFileDetection() {
    if (!state.file || !state.mediaType) {
        setStatus("请先上传图片或视频", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", state.file);
    formData.append("media_type", state.mediaType);
    formData.append("conf_thres", elements.confThres.value);

    elements.detectBtn.disabled = true;
    setStatus("正在执行检测，请稍候…", "busy");

    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            throw new Error(payload.message || "检测失败");
        }

        renderResultPreview(payload.resultUrl, payload.mediaType);
        renderSummary(payload.recognizedPlates);
        if (payload.mediaType === "image") {
            renderImageDetails(payload.detections);
        } else {
            renderVideoDetails(payload);
        }
        setStatus(payload.message, "success");
    } catch (error) {
        resetResultPanels();
        setStatus(error.message, "error");
    } finally {
        elements.detectBtn.disabled = false;
    }
}

async function fetchCameraStatus() {
    if (!state.cameraStatusUrl) {
        return;
    }

    try {
        const response = await fetch(`${state.cameraStatusUrl}&t=${Date.now()}`);
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            throw new Error(payload.message || "无法获取摄像头状态");
        }

        const { camera } = payload;
        renderSummary(camera.recognizedPlates);
        renderCameraDetails(camera);
        if (camera.lastError) {
            setStatus(camera.lastError, "error");
        } else if (camera.active) {
            setStatus(`正在从摄像头 ${camera.sourceLabel} 实时识别车牌`, "busy");
        } else if (camera.framesProcessed > 0) {
            setStatus(`摄像头流已连接，最近一次识别已处理 ${camera.framesProcessed} 帧`, "success");
        }
    } catch (error) {
        clearCameraPolling();
        setStatus(error.message, "error");
    }
}

function startCameraPolling() {
    clearCameraPolling();
    fetchCameraStatus();
    state.cameraStatusTimer = window.setInterval(fetchCameraStatus, 1500);
}

function runCameraDetection() {
    if (!state.cameraSessionId || !state.cameraDetectStreamUrl) {
        setStatus("请先接入摄像头", "error");
        return;
    }

    const detectStreamUrl = `${state.cameraDetectStreamUrl}&conf_thres=${encodeURIComponent(elements.confThres.value)}&t=${Date.now()}`;
    elements.resultPreview.className = "media-stage";
    elements.resultPreview.innerHTML = createMediaMarkup(detectStreamUrl, "camera", "摄像头检测结果");
    elements.resultBadge.textContent = "实时检测中";
    setStatus("正在从摄像头实时识别车牌…", "busy");
    startCameraPolling();
}

function runDetection() {
    if (state.mediaType === "camera") {
        runCameraDetection();
        return;
    }
    runFileDetection();
}

elements.uploadImageBtn.addEventListener("click", () => elements.imageInput.click());
elements.uploadVideoBtn.addEventListener("click", () => elements.videoInput.click());
elements.connectCameraBtn.addEventListener("click", connectCamera);
elements.detectBtn.addEventListener("click", runDetection);

elements.imageInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) {
        switchToFileMode(file, "image");
    }
});

elements.videoInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) {
        switchToFileMode(file, "video");
    }
});

elements.confThres.addEventListener("input", () => {
    elements.confThresValue.textContent = Number(elements.confThres.value).toFixed(2);
});

window.addEventListener("beforeunload", () => {
    clearCameraPolling();
    releaseSourcePreviewUrl();
});

resetSourcePanel();
resetResultPanels();
