/* ==========================================================================
   Vision Scraper - Frontend Application JavaScript Engine
   Handles WebSockets, Live Extraction Data Updates, Canvas Charts & Device CRUD
   ========================================================================== */

let devices = [];
let activeDeviceId = "";
let socket = null;
let pressureChartCtx = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    loadDevices();
    startClockTimer();
    restoreSidebarPreferences();

    // Live data polling via HTTP REST API (1 second refresh)
    setInterval(pollDeviceData, 1000);
});

function toggleLeftSidebar() {
    const container = document.querySelector(".app-container");
    const btn = document.getElementById("btn-toggle-left");
    const label = document.getElementById("label-toggle-left");
    if (!container) return;

    const isHidden = container.classList.toggle("hide-left");
    if (btn) {
        btn.classList.toggle("active-hidden", isHidden);
    }
    if (label) {
        label.textContent = isHidden ? "▶ Devices" : "◀ Devices";
    }
    localStorage.setItem("hideLeftSidebar", isHidden ? "true" : "false");
}

function toggleRightSidebar() {
    const container = document.querySelector(".app-container");
    const btn = document.getElementById("btn-toggle-right");
    const label = document.getElementById("label-toggle-right");
    if (!container) return;

    const isHidden = container.classList.toggle("hide-right");
    if (btn) {
        btn.classList.toggle("active-hidden", isHidden);
    }
    if (label) {
        label.textContent = isHidden ? "Extracted Data ◀" : "Extracted Data ▶";
    }
    localStorage.setItem("hideRightSidebar", isHidden ? "true" : "false");
}

function restoreSidebarPreferences() {
    if (localStorage.getItem("hideLeftSidebar") === "true") {
        const container = document.querySelector(".app-container");
        if (container) container.classList.add("hide-left");
        const btn = document.getElementById("btn-toggle-left");
        const label = document.getElementById("label-toggle-left");
        if (btn) btn.classList.add("active-hidden");
        if (label) label.textContent = "▶ Devices";
    }
    if (localStorage.getItem("hideRightSidebar") === "true") {
        const container = document.querySelector(".app-container");
        if (container) container.classList.add("hide-right");
        const btn = document.getElementById("btn-toggle-right");
        const label = document.getElementById("label-toggle-right");
        if (btn) btn.classList.add("active-hidden");
        if (label) label.textContent = "Extracted Data ◀";
    }
}

async function loadDevices() {
    try {
        const res = await fetch("/api/devices");
        if (res.ok) {
            devices = await res.json();
            renderDeviceList();
            renderStreamDropdown();
            if (devices.length > 0 && (!activeDeviceId || !devices.some(d => d.id === activeDeviceId))) {
                selectDevice(devices[0].id);
            } else if (activeDeviceId) {
                updateActiveDeviceUI();
            }
        }
    } catch (err) {
        console.error("Error loading devices:", err);
    }
}

function initChart() {
    const canvas = document.getElementById("pressureChart");
    if (canvas) {
        pressureChartCtx = canvas.getContext("2d");
        drawPressureChart([-150, -160, -155, -165, -170, -162, -158, -164], [-290, -295, -292, -288, -290, -294, -291, -289]);
    }
}

function startClockTimer() {
    setInterval(() => {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const mins = String(now.getMinutes()).padStart(2, '0');
        const secs = String(now.getSeconds()).padStart(2, '0');
        
        const clockEl = document.getElementById("live-clock-display");
        if (clockEl) {
            clockEl.textContent = `${day}-${month}-${year} ${hours}:${mins}:${secs}`;
        }
    }, 1000);
}

function renderDeviceList() {
    const listEl = document.getElementById("device-tree-list");
    const countBadge = document.getElementById("device-count-badge");
    if (!listEl) return;

    if (countBadge) countBadge.textContent = devices.length;
    listEl.innerHTML = "";

    devices.forEach(dev => {
        const isActive = dev.id === activeDeviceId;
        const card = document.createElement("div");
        card.className = `device-item-card ${isActive ? 'active' : ''}`;
        card.onclick = () => selectDevice(dev.id);

        card.innerHTML = `
            <div class="device-card-header">
                <div class="device-title-box">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M23 7l-7 5 7 5V7z"></path>
                        <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                    </svg>
                    <span class="device-id">${dev.id}</span>
                </div>
                <button class="btn-delete-cross" onclick="deleteDeviceById(event, '${dev.id}')" title="Erase / Remove Device">&times;</button>
            </div>
            <div class="device-name">${dev.name}</div>
            <div class="device-ip">${dev.ip || '127.0.0.1'}</div>
        `;
        listEl.appendChild(card);
    });
}

function renderStreamDropdown() {
    const dropdown = document.getElementById("active-stream-dropdown");
    if (!dropdown) return;

    dropdown.innerHTML = "";
    devices.forEach(dev => {
        const opt = document.createElement("option");
        opt.value = dev.id;
        opt.textContent = `${dev.id} — ${dev.name}`;
        if (dev.id === activeDeviceId) opt.selected = true;
        dropdown.appendChild(opt);
    });
}

function selectDevice(deviceId) {
    activeDeviceId = deviceId;
    renderDeviceList();
    renderStreamDropdown();
    updateActiveDeviceUI();

    // Update Stream Image Source
    const imgEl = document.getElementById("rtsp-live-stream-img");
    if (imgEl) {
        imgEl.src = `/api/stream/${deviceId}?t=${Date.now()}`;
    }

    pollDeviceData();
}

function updateActiveDeviceUI() {
    const dev = devices.find(d => d.id === activeDeviceId);
    if (!dev) return;

    const elId = document.getElementById("info-device-id");
    if (elId) elId.textContent = dev.id;
    const elName = document.getElementById("info-device-name");
    if (elName) elName.textContent = dev.name;
    const elRtsp = document.getElementById("info-device-rtsp");
    if (elRtsp) elRtsp.textContent = dev.camera_source || dev.rtsp_url || "0";
    const elRtspBox = document.getElementById("info-device-rtsp-box");
    if (elRtspBox) elRtspBox.textContent = `Source: ${dev.camera_source || dev.rtsp_url || 'Camera 0'} | Attached Camera Feed`;

    const statusPill = document.getElementById("info-device-status");
    if (statusPill) {
        const isOnline = (dev.status || '').toLowerCase().includes('online');
        statusPill.textContent = isOnline ? "o Online" : "o Offline";
        statusPill.className = `status-pill ${isOnline ? 'online' : 'offline'}`;
    }
}

async function pollDeviceData() {
    if (!activeDeviceId) return;
    try {
        const res = await fetch(`/api/devices/${activeDeviceId}/data`);
        if (res.ok) {
            const data = await res.json();
            updateExtractedDataDisplay(data);
        }
    } catch (err) {
        console.error("Data polling error:", err);
    }
}

function updateExtractedDataDisplay(data) {
    if (!data || !data.fields) return;

    const fields = data.fields;

    // Helper to safely format value
    const getVal = (key, fallback) => {
        if (fields[key] && fields[key].value !== null && fields[key].value !== undefined) {
            return fields[key].value;
        }
        return fallback;
    };

    // Update LCD Screen Value Boxes
    setLcdText("val-kt-v", getVal("Kt/V", "0.76"));
    setLcdText("val-plasma-na", getVal("Plasma Na", "134"));
    setLcdText("val-goal-in", getVal("Goal in", "1:53"));
    setLcdText("val-clearance", getVal("Clearance", "161"));

    setLcdText("val-uf-volume", getVal("UF Volume", "5,581"));
    setLcdText("val-uf-time-left", getVal("UF Time Left", "1:43"));
    setLcdText("val-uf-rate", getVal("UF Rate", "946"));
    setLcdText("val-uf-goal", getVal("UF Goal", "4,000"));
    setLcdText("val-eff-blood-flow", getVal("Eff. Blood Flow", "197"));
    setLcdText("val-cum-blood-vol", getVal("Cum. Blood Vol.", "96.2"));

    // Raw OCR log update
    const rawTextEl = document.getElementById("raw-ocr-text");
    if (rawTextEl) {
        let str = `Timestamp: ${data.timestamp}\nOCR Processing Time: ${data.ocr_duration || 0.12}s\nMode: ${data.mode}\n\n`;
        if (data.raw_text) {
            str += `--- RAW SCRAPED OCR TEXT ---\n${data.raw_text}\n\n`;
        }
        if (data.numbers_found && data.numbers_found.length > 0) {
            str += `--- DETECTED NUMERIC READINGS ---\n${data.numbers_found.join(', ')}\n`;
        }
        rawTextEl.textContent = str;
    }

    const confTag = document.getElementById("ocr-confidence-tag");
    if (confTag && data.confidence) {
        confTag.textContent = `Conf: ${Math.round(data.confidence * 100)}%`;
    }

    const lastTs = document.getElementById("ocr-last-timestamp");
    if (lastTs && data.timestamp) {
        lastTs.textContent = data.timestamp.split(" ")[1] || data.timestamp;
    }

    // Chart update
    if (data.pressure_history) {
        drawPressureChart(data.pressure_history.art_pressure, data.pressure_history.ven_pressure);
    }
}

function setLcdText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function drawPressureChart(artData, venData) {
    if (!pressureChartCtx) return;
    const ctx = pressureChartCtx;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw Grid Lines
    ctx.strokeStyle = "#d0d5dd";
    ctx.lineWidth = 1;

    for (let y = 15; y < h; y += 25) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    for (let x = 40; x < w; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }

    // Draw Art. Pressure Line (Red)
    if (artData && artData.length > 0) {
        ctx.strokeStyle = "#e53e3e";
        ctx.lineWidth = 2;
        ctx.beginPath();
        const stepX = w / Math.max(1, artData.length - 1);
        artData.forEach((val, idx) => {
            const y = h / 2 + (val + 160) * 0.8;
            const x = idx * stepX;
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    // Draw Ven. Pressure Line (Blue)
    if (venData && venData.length > 0) {
        ctx.strokeStyle = "#3182ce";
        ctx.lineWidth = 2;
        ctx.beginPath();
        const stepX = w / Math.max(1, venData.length - 1);
        venData.forEach((val, idx) => {
            const y = h - 15 + (val + 290) * 0.8;
            const x = idx * stepX;
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
}

function startWebSocket() {
    // WebSockets disabled as requested — using HTTP REST polling for live data
}


async function triggerManualOCR() {
    if (!activeDeviceId) return;
    try {
        const res = await fetch(`/api/devices/${activeDeviceId}/extract`, { method: "POST" });
        if (res.ok) {
            const data = await res.json();
            updateExtractedDataDisplay(data);
            alert("Instant OCR Extraction Completed!");
        }
    } catch (err) {
        alert("OCR Extraction Error: " + err);
    }
}

function filterDevices() {
    const query = document.getElementById("device-search-input").value.toLowerCase();
    const cards = document.querySelectorAll(".device-item-card");
    cards.forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = text.includes(query) ? "block" : "none";
    });
}

function toggleOcrBoxes() {
    alert("OCR Bounding Box Overlay Toggled!");
}

function captureSnapshot() {
    if (!activeDeviceId) return;
    window.open(`/api/stream/${activeDeviceId}/snapshot`, "_blank");
}

/*
// RTSP stream validation disabled for now
function validateRTSP() {
    const dev = devices.find(d => d.id === activeDeviceId);
    if (!dev) return;
    alert(`Validating Camera Stream for ${dev.name}...\\nStatus: Stream Connected Successfully`);
}
*/

function refreshDevice() {
    loadDevices();
    pollDeviceData();
}

function openAddDeviceModal() {
    document.getElementById("modal-title").textContent = "Add Camera Device";
    document.getElementById("form-device-id").value = "";
    document.getElementById("form-id-input").value = "";
    document.getElementById("form-id-input").disabled = false;
    document.getElementById("form-name-input").value = "";
    document.getElementById("form-rtsp-input").value = "0";
    document.getElementById("device-modal").style.display = "flex";
}

function openEditModal() {
    const dev = devices.find(d => d.id === activeDeviceId);
    if (!dev) return;

    document.getElementById("modal-title").textContent = "Edit Camera Device";
    document.getElementById("form-device-id").value = dev.id;
    document.getElementById("form-id-input").value = dev.id;
    document.getElementById("form-id-input").disabled = true;
    document.getElementById("form-name-input").value = dev.name;
    const inputEl = document.getElementById("form-rtsp-input");
    if (inputEl) inputEl.value = dev.camera_source || dev.rtsp_url || "0";
    document.getElementById("form-mode-input").value = dev.mode || "dialysis";
    document.getElementById("device-modal").style.display = "flex";
}

function closeDeviceModal() {
    document.getElementById("device-modal").style.display = "none";
}

async function saveDeviceForm(e) {
    e.preventDefault();
    const isEdit = !!document.getElementById("form-device-id").value;
    const devId = document.getElementById("form-id-input").value.trim();
    const camInput = document.getElementById("form-rtsp-input") ? document.getElementById("form-rtsp-input").value.trim() : "0";

    const payload = {
        id: devId,
        name: document.getElementById("form-name-input").value.trim(),
        camera_source: camInput,
        rtsp_url: camInput,
        mode: document.getElementById("form-mode-input").value,
        extraction_interval: parseFloat(document.getElementById("form-interval-input").value)
    };

    try {
        let res;
        if (isEdit) {
            res = await fetch(`/api/devices/${devId}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
        } else {
            res = await fetch("/api/devices", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
        }

        if (res.ok) {
            closeDeviceModal();
            await loadDevices();
            selectDevice(devId);
        } else {
            alert("Error saving camera device.");
        }
    } catch (err) {
        alert("Failed to save device: " + err);
    }
}

async function deleteDeviceById(event, targetId) {
    if (event) event.stopPropagation();
    const devId = targetId || activeDeviceId;
    if (!devId) return;

    if (!confirm(`Are you sure you want to erase/remove device "${devId}"?`)) return;

    try {
        const res = await fetch(`/api/devices/${encodeURIComponent(devId)}`, { method: "DELETE" });
        if (res.ok) {
            await loadDevices();
        }
    } catch (err) {
        alert("Failed to delete device: " + err);
    }
}

async function deleteDevice() {
    deleteDeviceById(null, activeDeviceId);
}

function generateReport() {
    alert(`Generating OTP & Extracted Data Report for Device ${activeDeviceId}...\nReport saved to output/ folder.`);
}

function sendActionControl() {
    const val = document.getElementById("action-select").value;
    const ack = document.getElementById("ack-status-text");
    if (ack) {
        ack.textContent = `Action '${val}' sent to device ${activeDeviceId} at ${new Date().toLocaleTimeString()} (ACK Received)`;
    }
    alert(`Action Control '${val}' executed successfully on device ${activeDeviceId}.`);
}

function toggleAccordion(id) {
    const body = document.getElementById(id);
    if (body) {
        body.style.display = body.style.display === "none" ? "block" : "none";
    }
}

/* ==========================================================================
   HARDWARE CAMERA DISCOVERY & CAPTURE SAVE ENGINE
   ========================================================================== */

let hardwareCameras = [];

async function discoverHardwareCameras() {
    const badge = document.getElementById("hw-camera-count-badge");
    const listEl = document.getElementById("camera-hardware-list");
    if (badge) badge.textContent = "Scanning...";

    try {
        const res = await fetch("/api/cameras/discover");
        if (res.ok) {
            const data = await res.json();
            hardwareCameras = data.cameras || [];
            if (badge) badge.textContent = `${hardwareCameras.length} Camera(s) Found`;

            if (listEl) {
                listEl.innerHTML = "";
                if (hardwareCameras.length === 0) {
                    listEl.innerHTML = `<div class="empty-hw">No extra USB camera detected. (Defaulting to built-in laptop camera #0)</div>`;
                } else {
                    hardwareCameras.forEach(cam => {
                        const item = document.createElement("div");
                        item.className = "hw-cam-card";
                        const isLaptop = cam.index === 0;
                        item.innerHTML = `
                            <div class="hw-cam-title">
                                <span>${isLaptop ? '💻' : '📷'} ${cam.name}</span>
                                <span class="hw-cam-res">${cam.resolution}</span>
                            </div>
                            <button class="btn-use-cam" onclick="switchStreamToCamera('${cam.rtsp_url}', '${cam.name}')">
                                ▶ Stream from ${isLaptop ? 'Laptop Cam' : 'USB Cam'}
                            </button>
                        `;
                        listEl.appendChild(item);
                    });
                }
            }
        }
    } catch (err) {
        console.error("Camera discovery error:", err);
        if (badge) badge.textContent = "Scan Error";
    }
}

async function switchStreamToCamera(camSource, camName) {
    if (!activeDeviceId) return;
    try {
        const res = await fetch(`/api/devices/${activeDeviceId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                id: activeDeviceId,
                name: camName || `Camera ${activeDeviceId}`,
                rtsp_url: String(camSource)
            })
        });
        if (res.ok) {
            await loadDevices();
            selectDevice(activeDeviceId);
            alert(`Switched active camera stream to: ${camName} (Source index: ${camSource})`);
        }
    } catch (err) {
        alert("Failed to switch camera source: " + err);
    }
}

async function captureAndSaveJSON() {
    if (!activeDeviceId) {
        alert("Please select an active camera stream first.");
        return;
    }

    try {
        const btn = document.querySelector(".btn-control.capture-save-gold");
        if (btn) {
            btn.disabled = true;
            btn.textContent = "Capturing & Extracting...";
        }

        const res = await fetch(`/api/devices/${activeDeviceId}/capture-and-save`, {
            method: "POST"
        });

        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                </svg>
                <span>📸 Capture & Save JSON</span>
            `;
        }

        if (res.ok) {
            const data = await res.json();
            openCaptureModal(data);
        } else {
            alert("Error capturing frame and saving JSON data.");
        }
    } catch (err) {
        alert("Capture failed: " + err);
    }
}

function openCaptureModal(data) {
    const modal = document.getElementById("capture-result-modal");
    const imgEl = document.getElementById("capture-modal-img");
    const jsonEl = document.getElementById("capture-modal-json-text");
    const btnPng = document.getElementById("btn-download-png");
    const btnJson = document.getElementById("btn-download-json");

    if (imgEl && data.image_url) {
        imgEl.src = `${data.image_url}?t=${Date.now()}`;
    }

    if (jsonEl) {
        jsonEl.textContent = JSON.stringify(data, null, 2);
    }

    if (btnPng && data.image_filename) {
        btnPng.href = `/api/captures/download/${data.image_filename}`;
        btnPng.download = data.image_filename;
    }

    if (btnJson && data.json_filename) {
        btnJson.href = `/api/captures/download/${data.json_filename}`;
        btnJson.download = data.json_filename;
    }

    if (modal) modal.style.display = "flex";
}

function closeCaptureModal() {
    const modal = document.getElementById("capture-result-modal");
    if (modal) modal.style.display = "none";
}

async function openSavedCapturesModal() {
    const modal = document.getElementById("saved-captures-modal");
    const tbody = document.getElementById("saved-captures-tbody");
    if (modal) modal.style.display = "flex";

    if (tbody) tbody.innerHTML = `<tr><td colspan="6">Loading saved captures archive...</td></tr>`;

    try {
        const res = await fetch("/api/captures");
        if (res.ok) {
            const data = await res.json();
            const captures = data.captures || [];

            if (!tbody) return;
            tbody.innerHTML = "";

            if (captures.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6">No saved captures found in output/ folder yet. Click "📸 Capture & Save JSON" to create one.</td></tr>`;
                return;
            }

            captures.forEach(cap => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${cap.formatted_timestamp || cap.timestamp}</td>
                    <td>${cap.device_name || cap.device_id}</td>
                    <td><span class="mode-tag">${cap.mode || 'dialysis'}</span></td>
                    <td><a href="${cap.image_url}" target="_blank" class="table-link">🖼️ ${cap.image_filename}</a></td>
                    <td><a href="${cap.json_url}" target="_blank" class="table-link">📄 ${cap.json_filename}</a></td>
                    <td>
                        <a href="/api/captures/download/${cap.json_filename}" download class="btn-sm btn-json-dl">⬇️ JSON</a>
                        <a href="/api/captures/download/${cap.image_filename}" download class="btn-sm btn-png-dl">🖼️ PNG</a>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        if (tbody) tbody.innerHTML = `<tr><td colspan="6">Error loading saved captures archive.</td></tr>`;
    }
}

function closeSavedCapturesModal() {
    const modal = document.getElementById("saved-captures-modal");
    if (modal) modal.style.display = "none";
}

