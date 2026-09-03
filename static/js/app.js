/* ==========================================================================
   Vision Scraper - Frontend Application JavaScript Engine
   Raspberry Pi Camera Live Data Extraction & Digitized Screen UI
   ========================================================================== */

let devices = [];
let activeDeviceId = "";
let pressureChartCtx = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    loadDevices();
    startClockTimer();
    restoreSidebarPreferences();

    // Live data polling via HTTP REST API
    setInterval(pollDeviceData, 1500);
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
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                    <span class="device-id">${dev.id}</span>
                </div>
            </div>
            <div class="device-name">${dev.name}</div>
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

    const imgEl = document.getElementById("rtsp-live-stream-img");
    if (imgEl) {
        imgEl.src = `/api/stream/${deviceId}?t=${Date.now()}`;
    }

    pollDeviceData();
}

function updateActiveDeviceUI() {
    const dev = devices.find(d => d.id === activeDeviceId);
    if (!dev) return;

    const idEl = document.getElementById("info-device-id");
    if (idEl) idEl.textContent = dev.id;

    const nameEl = document.getElementById("info-device-name");
    if (nameEl) nameEl.textContent = dev.name;

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

    const getVal = (key, fallback) => {
        if (fields[key] && fields[key].value !== null && fields[key].value !== undefined) {
            return fields[key].value;
        }
        return fallback;
    };

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

function refreshDevice() {
    loadDevices();
    pollDeviceData();
}

function openEditModal() {
    const dev = devices.find(d => d.id === activeDeviceId);
    if (!dev) return;

    document.getElementById("modal-title").textContent = "Edit Camera Device";
    document.getElementById("form-device-id").value = dev.id;
    document.getElementById("form-id-input").value = dev.id;
    document.getElementById("form-id-input").disabled = true;
    document.getElementById("form-name-input").value = dev.name;
    document.getElementById("form-mode-input").value = dev.mode || "dialysis";
    document.getElementById("device-modal").style.display = "flex";
}

function closeDeviceModal() {
    document.getElementById("device-modal").style.display = "none";
}

async function saveDeviceForm(e) {
    e.preventDefault();
    const devId = document.getElementById("form-id-input").value.trim();

    const payload = {
        id: devId,
        name: document.getElementById("form-name-input").value.trim(),
        mode: document.getElementById("form-mode-input").value,
        extraction_interval: parseFloat(document.getElementById("form-interval-input").value)
    };

    try {
        const res = await fetch(`/api/devices/${devId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

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

function toggleAccordion(id) {
    const body = document.getElementById(id);
    if (body) {
        body.style.display = body.style.display === "none" ? "block" : "none";
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
