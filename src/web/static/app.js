const elements = {
    uploadImageBtn: document.getElementById("upload-image-btn"),
    uploadVideoBtn: document.getElementById("upload-video-btn"),
    detectBtn: document.getElementById("detect-btn"),
    imageInput: document.getElementById("image-input"),
    videoInput: document.getElementById("video-input"),
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

function resetResultPanels() {
    elements.resultBadge.textContent = "等待检测";
    elements.resultPreview.className = "media-stage empty-stage";
    elements.resultPreview.innerHTML = "<p>点击“检测车牌”后展示标注结果</p>";
    setEmpty(elements.summaryList, "检测完成后会列出识别到的车牌和出现次数", "summary-list");
    setEmpty(elements.detailList, "图片模式会列出检测框坐标和置信度，视频模式显示帧处理信息", "detail-list");
}

function formatFileMeta(file, mediaType) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    return `${mediaType === "image" ? "图片" : "视频"} · ${file.name} · ${sizeMb} MB`;
}

function createMediaMarkup(src, mediaType) {
    if (mediaType === "image") {
        return `<img src="${src}" alt="预览图片">`;
    }
    return `<video src="${src}" controls playsinline></video>`;
}

function renderSourcePreview(file, mediaType) {
    releaseSourcePreviewUrl();
    state.sourceObjectUrl = URL.createObjectURL(file);
    elements.sourcePreview.className = "media-stage";
    elements.sourcePreview.innerHTML = createMediaMarkup(state.sourceObjectUrl, mediaType);
    elements.sourceBadge.textContent = formatFileMeta(file, mediaType);
}

function renderResultPreview(resultUrl, mediaType) {
    const cacheBustedUrl = `${resultUrl}?t=${Date.now()}`;
    elements.resultPreview.className = "media-stage";
    elements.resultPreview.innerHTML = createMediaMarkup(cacheBustedUrl, mediaType);
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
                    <span>bbox: [${x1}, ${y1}, ${x2}, ${y2}] · conf: ${item.conf.toFixed(2)}</span>
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
            <span>${parts.join(" · ")}</span>
        </div>
        ${
            payload.transcodeWarning
                ? `<p class="video-note">${payload.transcodeWarning}</p>`
                : ""
        }
    `;
}

function handleFileSelection(file, mediaType) {
    state.file = file;
    state.mediaType = mediaType;
    elements.detectBtn.disabled = false;
    renderSourcePreview(file, mediaType);
    resetResultPanels();
    setStatus(`已选择${mediaType === "image" ? "图片" : "视频"}，可以开始检测`, "idle");
}

async function runDetection() {
    if (!state.file || !state.mediaType) {
        setStatus("请先上传图片或视频", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", state.file);
    formData.append("media_type", state.mediaType);
    formData.append("conf_thres", elements.confThres.value);

    elements.detectBtn.disabled = true;
    setStatus("正在调用 src/test.py 执行检测，请稍候…", "busy");

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

elements.uploadImageBtn.addEventListener("click", () => elements.imageInput.click());
elements.uploadVideoBtn.addEventListener("click", () => elements.videoInput.click());
elements.detectBtn.addEventListener("click", runDetection);

elements.imageInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) {
        elements.videoInput.value = "";
        handleFileSelection(file, "image");
    }
});

elements.videoInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) {
        elements.imageInput.value = "";
        handleFileSelection(file, "video");
    }
});

elements.confThres.addEventListener("input", () => {
    elements.confThresValue.textContent = Number(elements.confThres.value).toFixed(2);
});

resetResultPanels();
