(() => {
  const video = document.getElementById('cameraVideo');
  const poseCanvas = document.getElementById('poseCanvas');
  const poseCtx = poseCanvas.getContext('2d');
  const captureCanvas = document.getElementById('captureCanvas');
  const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: false });
  const startButton = document.getElementById('startButton');
  const uploadButton = document.getElementById('uploadButton');
  const videoFileInput = document.getElementById('videoFileInput');
  const stopButton = document.getElementById('stopButton');
  const resetButton = document.getElementById('resetButton');
  const cameraSelect = document.getElementById('cameraSelect');
  const fpsSelect = document.getElementById('fpsSelect');
  const soundToggle = document.getElementById('soundToggle');
  const placeholder = document.getElementById('cameraPlaceholder');
  const videoStage = document.getElementById('videoStage');
  const fallFlash = document.getElementById('fallFlash');
  const serverChip = document.getElementById('serverChip');
  const statusCard = document.getElementById('statusCard');
  const statusText = document.getElementById('statusText');
  const statusDescription = document.getElementById('statusDescription');
  const probabilityText = document.getElementById('probabilityText');
  const probabilityFill = document.getElementById('probabilityFill');
  const thresholdMark = document.getElementById('thresholdMark');
  const thresholdText = document.getElementById('thresholdText');
  const fallCount = document.getElementById('fallCount');
  const fpsValue = document.getElementById('fpsValue');
  const deviceText = document.getElementById('deviceText');
  const eventList = document.getElementById('eventList');
  const clearLogButton = document.getElementById('clearLogButton');
  const sessionLabel = document.getElementById('sessionLabel');

  let stream = null;
  let videoObjectUrl = null;
  let mediaMode = 'idle'; // idle | camera | video
  let running = false;
  let requestInFlight = false;
  let previousStatus = 'IDLE';
  let audioContext = null;
  let peakProbability = 0;
  let latestFallCount = 0;
  let videoEndLogged = false;
  let scheduleTimer = null;
  let isInitialVideoPlay = false;

  // Accessibility: Gắn aria-live đúng vị trí
  statusCard.removeAttribute('aria-live');
  statusCard.removeAttribute('aria-atomic');
  const statusTextContainer = statusText.parentElement;
  if (statusTextContainer) {
    statusTextContainer.setAttribute('aria-live', 'polite');
    statusTextContainer.setAttribute('aria-atomic', 'true');
  }

  const sessionId = (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
  if (sessionLabel) {
    sessionLabel.textContent = `Session: ${sessionId.slice(0, 8)}`;
  }

  function getCookie(name) {
    const entry = document.cookie.split(';').map(v => v.trim()).find(v => v.startsWith(`${name}=`));
    return entry ? decodeURIComponent(entry.split('=').slice(1).join('=')) : '';
  }

  function setServerState(kind, text) {
    if (!serverChip) return;
    serverChip.classList.remove('ready', 'error');
    if (kind) serverChip.classList.add(kind);
    const label = serverChip.querySelector('b');
    if (label) label.textContent = text;
  }

  async function listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    const selected = cameraSelect.value;
    cameraSelect.innerHTML = cameras.length ? '' : '<option value="">Camera mặc định</option>';
    cameras.forEach((camera, index) => {
      const option = document.createElement('option');
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      if (camera.deviceId === selected) option.selected = true;
      cameraSelect.appendChild(option);
    });
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      addEvent('Trình duyệt không hỗ trợ getUserMedia.', 'fall');
      return;
    }

    startButton.disabled = true;
    startButton.classList.add('is-loading');
    uploadButton.disabled = true;

    try {
      stopCurrentMedia(false);
      await resetSession(true);

      const deviceId = cameraSelect.value;
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 20, max: 30 }
        },
        audio: false
      });
      video.removeAttribute('src');
      video.srcObject = stream;
      video.controls = false;
      await video.play();
      await listCameras();
      resizeCanvases();

      mediaMode = 'camera';
      running = true;
      placeholder.classList.add('hidden');
      startButton.disabled = true;
      uploadButton.disabled = false;
      stopButton.disabled = false;
      setServerState('ready', 'Hệ thống sẵn sàng');
      addEvent('Đã bật camera và bắt đầu phân tích.', 'normal');
      scheduleNext(0);
    } catch (error) {
      startButton.disabled = false;
      uploadButton.disabled = false;
      addEvent(`Không mở được camera: ${error.message}`, 'fall');
      updateStatus({ status: 'NO_PERSON', persons: [] });
    } finally {
      startButton.classList.remove('is-loading');
    }
  }

  function chooseDemoVideo() {
    videoFileInput.value = '';
    videoFileInput.click();
  }

  async function loadDemoVideo(file) {
    if (!file) return;
    const supportedExtension = /\.(mp4|webm|mov|m4v)$/i.test(file.name);
    if (!file.type.startsWith('video/') && !supportedExtension) {
      addEvent('Định dạng video không được hỗ trợ. Hãy dùng MP4, WebM hoặc MOV.', 'fall');
      return;
    }

    startButton.disabled = true;
    uploadButton.disabled = true;
    try {
      stopCurrentMedia(false);
      await resetSession(true);

      videoObjectUrl = URL.createObjectURL(file);
      video.srcObject = null;
      video.src = videoObjectUrl;
      video.controls = true;
      video.muted = true;
      videoEndLogged = false;

      await waitForVideoMetadata();
      resizeCanvases();
      mediaMode = 'video';
      running = true;
      placeholder.classList.add('hidden');
      startButton.disabled = false;
      uploadButton.disabled = false;
      stopButton.disabled = false;
      setServerState('ready', 'Video Demo Ready');

      const duration = Number.isFinite(video.duration) ? formatDuration(video.duration) : 'không xác định';
      addEvent(`Đã tải video “${file.name}” · thời lượng ${duration}.`, 'normal');

      isInitialVideoPlay = true;

      try {
        await video.play();
      } catch (_) {
        addEvent('Video đã sẵn sàng. Nhấn nút Play trên video để bắt đầu.', 'normal');
      }
    } catch (error) {
      startButton.disabled = false;
      uploadButton.disabled = false;
      stopButton.disabled = true;
      addEvent(`Không mở được video: ${error.message}`, 'fall');
      stopCurrentMedia(false);
    }
  }

  function waitForVideoMetadata() {
    if (video.readyState >= 1) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const onLoaded = () => { cleanup(); resolve(); };
      const onError = () => { cleanup(); reject(new Error('Trình duyệt không giải mã được video.')); };
      const cleanup = () => {
        video.removeEventListener('loadedmetadata', onLoaded);
        video.removeEventListener('error', onError);
      };
      video.addEventListener('loadedmetadata', onLoaded, { once: true });
      video.addEventListener('error', onError, { once: true });
      video.load();
    });
  }

  function stopCurrentMedia(announce = true) {
    const oldMode = mediaMode;
    running = false;
    requestInFlight = false;
    isInitialVideoPlay = false;
    if (scheduleTimer !== null) {
      window.clearTimeout(scheduleTimer);
      scheduleTimer = null;
    }

    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.pause();
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
    video.controls = false;

    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    videoObjectUrl = null;
    mediaMode = 'idle';

    poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
    placeholder.classList.remove('hidden');
    startButton.disabled = false;
    uploadButton.disabled = false;
    stopButton.disabled = true;
    videoStage.classList.remove('fall');
    fallFlash.classList.remove('visible');
    setStatusVisual('IDLE', 'CHỜ DỮ LIỆU', 'Hệ thống đã dừng.');

    if (announce && oldMode !== 'idle') {
      addEvent(oldMode === 'video' ? 'Đã dừng video demo.' : 'Đã dừng camera.', 'normal');
    }
  }

  function resizeCanvases() {
    const sourceW = video.videoWidth || 640;
    const sourceH = video.videoHeight || 480;
    const targetW = 640;
    const targetH = Math.round(targetW * sourceH / sourceW);
    captureCanvas.width = targetW;
    captureCanvas.height = targetH;
    poseCanvas.width = targetW;
    poseCanvas.height = targetH;
  }

  function scheduleNext(delay = null) {
    if (!running) return;
    const interval = delay === null ? Number(fpsSelect.value) : delay;
    if (scheduleTimer !== null) window.clearTimeout(scheduleTimer);
    scheduleTimer = window.setTimeout(() => {
      scheduleTimer = null;
      sendFrame();
    }, interval);
  }

  async function sendFrame() {
    if (!running) return;
    if (mediaMode === 'video' && (video.paused || video.ended)) {
      scheduleNext(100);
      return;
    }
    if (requestInFlight || video.readyState < 2) {
      scheduleNext();
      return;
    }

    requestInFlight = true;
    const started = performance.now();
    try {
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const blob = await new Promise(resolve => captureCanvas.toBlob(resolve, 'image/jpeg', 0.72));
      if (!blob) throw new Error('Không tạo được JPEG từ nguồn hình ảnh.');

      const form = new FormData();
      form.append('image', blob, mediaMode === 'video' ? 'video-frame.jpg' : 'camera.jpg');
      form.append('session_id', sessionId);

      const response = await fetch('/predict/', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        body: form
      });

      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);

      drawPose(data);
      updateStatus(data);
    } catch (error) {
      addEvent(`Lỗi frame: ${error.message}`, 'fall', true);
    } finally {
      requestInFlight = false;
      const elapsed = performance.now() - started;
      scheduleNext(Math.max(20, Number(fpsSelect.value) - elapsed));
    }
  }

  function drawPose(data) {
    poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
    const keypoints = data.keypoints || [];
    if (keypoints.length === 0) return;

    const sx = poseCanvas.width / (data.image_width || 640);
    const sy = poseCanvas.height / (data.image_height || 480);

    const connections = [
      [0, 1], [0, 2], [1, 3], [2, 4],
      [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
      [5, 11], [6, 12], [11, 12],
      [11, 13], [13, 15], [12, 14], [14, 16]
    ];

    const isFall = data.is_fall;
    const strokeColor = isFall ? '#ff5267' : '#34d6d1';
    const jointColor = isFall ? '#ff9aa7' : '#e9ffff';

    poseCtx.lineWidth = 2.5;
    poseCtx.lineCap = 'round';
    poseCtx.strokeStyle = strokeColor;

    connections.forEach(([a, b]) => {
      const p1 = keypoints[a];
      const p2 = keypoints[b];
      if (p1 && p2 && p1[2] > 0.15 && p2[2] > 0.15) {
        poseCtx.beginPath();
        poseCtx.moveTo(p1[0] * sx, p1[1] * sy);
        poseCtx.lineTo(p2[0] * sx, p2[1] * sy);
        poseCtx.stroke();
      }
    });

    keypoints.forEach(pt => {
      if (pt[2] > 0.15) {
        poseCtx.beginPath();
        poseCtx.fillStyle = jointColor;
        poseCtx.arc(pt[0] * sx, pt[1] * sy, 3.5, 0, Math.PI * 2);
        poseCtx.fill();
      }
    });
  }

  function setStatusVisual(kind, title, description) {
    statusCard.classList.remove('status-idle', 'status-normal', 'status-fall');
    statusCard.classList.add(kind === 'FALL' ? 'status-fall' : kind === 'NORMAL' ? 'status-normal' : 'status-idle');
    statusText.textContent = title;
    statusDescription.textContent = description;
  }

  function updateStatus(data) {
    const probability = Number(data.fall_probability || 0);
    const threshold = Number(data.threshold || 45.77);
    peakProbability = Math.max(peakProbability, probability);
    latestFallCount = Number(data.fall_count || 0);

    probabilityText.textContent = `${probability.toFixed(1)}%`;
    probabilityFill.style.width = `${Math.min(100, probability)}%`;
    probabilityFill.style.background = probability >= threshold ? '#ff5267' : probability >= threshold * 0.6 ? '#f6bf56' : '#41df8d';
    thresholdMark.style.left = `${threshold}%`;
    thresholdText.textContent = `${threshold.toFixed(2)}%`;
    fallCount.textContent = latestFallCount;
    fpsValue.textContent = Number(data.server_fps || 0).toFixed(1);
    if (data.device) deviceText.textContent = data.device === 'CUDA' ? 'NVIDIA CUDA' : data.device;

    const isFall = Boolean(data.is_fall);
    const hasPerson = Boolean(data.has_person);

    if (isFall) {
      setStatusVisual('FALL', 'FALL DETECTED', 'Cảnh báo: Phát hiện sự cố té ngã!');
      videoStage.classList.add('fall');
      fallFlash.classList.add('visible');

      if (previousStatus !== 'FALL') {
        const timeLabel = mediaMode === 'video' ? ` · mốc ${formatDuration(video.currentTime)}` : '';
        addEvent(`Phát hiện té ngã${timeLabel} · xác suất ${probability.toFixed(1)}%`, 'fall');
        beepAlarm();
      }
      previousStatus = 'FALL';
    } else if (hasPerson) {
      setStatusVisual('NORMAL', 'NORMAL', 'Đang giám sát tư thế bình thường.');
      videoStage.classList.remove('fall');
      fallFlash.classList.remove('visible');
      if (previousStatus === 'FALL') addEvent('Trạng thái đã trở lại bình thường.', 'normal');
      previousStatus = 'NORMAL';
    } else {
      setStatusVisual('IDLE', 'KHÔNG THẤY NGƯỜI', 'Hãy bảo đảm đối tượng nằm trong khung hình quan sát.');
      videoStage.classList.remove('fall');
      fallFlash.classList.remove('visible');
      previousStatus = 'NO_PERSON';
    }
  }

  async function resetSession(silent = false) {
    try {
      const form = new FormData();
      form.append('session_id', sessionId);
      await fetch('/reset/', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        body: form
      });
      fallCount.textContent = '0';
      fpsValue.textContent = '—';
      probabilityText.textContent = '0.0%';
      probabilityFill.style.width = '0%';
      previousStatus = 'IDLE';
      peakProbability = 0;
      latestFallCount = 0;
      videoEndLogged = false;
      poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
      if (!silent) addEvent('Đã đặt lại phiên phân tích và bộ đếm.', 'normal');
    } catch (error) {
      if (!silent) addEvent(error.message, 'fall');
    }
  }

  function handleVideoEnded() {
    if (mediaMode !== 'video' || videoEndLogged) return;
    running = false;
    if (scheduleTimer !== null) {
      window.clearTimeout(scheduleTimer);
      scheduleTimer = null;
    }
    videoEndLogged = true;
    stopButton.disabled = false;
    videoStage.classList.remove('fall');
    fallFlash.classList.remove('visible');
    setStatusVisual('IDLE', 'VIDEO HOÀN TẤT', `Đã phát hiện ${latestFallCount} sự kiện Fall.`);
    addEvent(`Hoàn tất video · ${latestFallCount} sự kiện Fall · xác suất cao nhất ${peakProbability.toFixed(1)}%.`, latestFallCount > 0 ? 'fall' : 'normal');
  }

  async function handleVideoPlay() {
    if (mediaMode !== 'video') return;

    if (isInitialVideoPlay) {
      isInitialVideoPlay = false;
    } else if (video.ended || video.currentTime < 0.2) {
      await resetSession(true);
    }

    videoEndLogged = false;
    running = true;
    stopButton.disabled = false;
    scheduleNext(0);
  }

  function formatDuration(seconds) {
    const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
    const minutes = Math.floor(safeSeconds / 60);
    const remainder = safeSeconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
  }

  function addEvent(message, type = 'normal', dedupe = false) {
    if (dedupe && eventList.firstElementChild?.dataset?.message === message) return;
    const empty = eventList.querySelector('.empty-log');
    if (empty) empty.remove();
    const row = document.createElement('div');
    row.className = `event-item ${type}`;
    row.dataset.message = message;
    const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
    row.innerHTML = `<i></i><div>${escapeHtml(message)}</div><span>${now}</span>`;
    eventList.prepend(row);
    while (eventList.children.length > 30) eventList.lastElementChild.remove();
  }

  function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value;
    return div.innerHTML;
  }

  let lastBeepTime = 0;
  const BEEP_COOLDOWN_MS = 1000;

  function beepAlarm() {
    if (!soundToggle.checked) return;

    const now = Date.now();
    if (now - lastBeepTime < BEEP_COOLDOWN_MS) return;
    lastBeepTime = now;

    try {
      audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
      [0, .22, .44].forEach(offset => {
        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();
        osc.frequency.value = 880;
        gain.gain.setValueAtTime(.001, audioContext.currentTime + offset);
        gain.gain.exponentialRampToValueAtTime(.2, audioContext.currentTime + offset + .02);
        gain.gain.exponentialRampToValueAtTime(.001, audioContext.currentTime + offset + .16);
        osc.connect(gain).connect(audioContext.destination);
        osc.start(audioContext.currentTime + offset);
        osc.stop(audioContext.currentTime + offset + .18);
      });
    } catch (_) { /* AudioContext policy */ }
  }

  // Gắn các sự kiện điều khiển UI
  startButton.addEventListener('click', startCamera);
  uploadButton.addEventListener('click', chooseDemoVideo);
  videoFileInput.addEventListener('change', event => loadDemoVideo(event.target.files?.[0]));
  stopButton.addEventListener('click', () => stopCurrentMedia(true));
  resetButton.addEventListener('click', () => resetSession(false));
  clearLogButton.addEventListener('click', () => { eventList.innerHTML = '<div class="empty-log">Chưa có sự kiện.</div>'; });
  cameraSelect.addEventListener('change', async () => { if (mediaMode === 'camera') { stopCurrentMedia(false); await startCamera(); } });
  video.addEventListener('ended', handleVideoEnded);
  video.addEventListener('play', handleVideoPlay);
  window.addEventListener('resize', () => { if (video.videoWidth) resizeCanvases(); });
  window.addEventListener('beforeunload', () => stopCurrentMedia(false));

  // Trạng thái ban đầu
  setServerState('ready', 'Model sẵn sàng');
})();