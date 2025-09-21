   // Global configuration
   let config = {
    apiBaseUrl: 'http://localhost:5000',
    cameraAvailable: true
};

// Camera and emotion detection variables
let videoStream = null;
let isCameraActive = false;
let emotionDetectionInterval = null;
let lastEmotion = 'unknown';
let emotionHistory = [];

// Load configuration on page load
async function loadConfig() {
    try {
        const response = await fetch('/config');
        config = await response.json();
        console.log('Configuration loaded:', config);
        updateApiStatus();
    } catch (e) {
        console.error('Failed to load configuration:', e);
        updateApiStatus('Error loading configuration');
    }
}

// Update API status display
function updateApiStatus(message = null) {
    const statusElement = document.getElementById('apiStatusText');
    const statusContainer = document.getElementById('apiStatus');
    
    if (message) {
        statusElement.textContent = message;
        statusContainer.style.backgroundColor = '#ffebee';
        statusContainer.style.color = '#c62828';
        return;
    }
    
    // Use the status from config if available
    if (config.status === 'ngrok' || config.isNgrok) {
        statusElement.textContent = `Using Ngrok API: ${config.apiBaseUrl}`;
        statusContainer.style.backgroundColor = '#e3f2fd';
        statusContainer.style.color = '#1565c0';
    } else if (config.apiBaseUrl === 'http://localhost:5000') {
        statusElement.textContent = 'Using Local Processing';
        statusContainer.style.backgroundColor = '#e8f5e8';
        statusContainer.style.color = '#2e7d32';
    } else {
        statusElement.textContent = `Using API: ${config.apiBaseUrl}`;
        statusContainer.style.backgroundColor = '#fff3e0';
        statusContainer.style.color = '#ef6c00';
    }
}

const pieCtx = document.getElementById('pieChart').getContext('2d');
const lineCtx = document.getElementById('timelineChart').getContext('2d');

const emotionColors = {
    angry: '#e74c3c',
    disgust: '#16a085',
    fear: '#8e44ad',
    happy: '#f1c40f',
    sad: '#3498db',
    surprise: '#e67e22',
    neutral: '#95a5a6'
};

let pieChart = new Chart(pieCtx, {
    type: 'pie',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [] }]},
    options: { responsive: true, plugins: { legend: { position: 'bottom' }}}
});

let timelineChart = new Chart(lineCtx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
        responsive: true,
        scales: { x: { title: { display: true, text: 'Time' }}, y: { beginAtZero: true, precision: 0 }},
        plugins: { legend: { position: 'bottom' }}
    }
});

// Camera functions
async function startCamera() {
    try {
        const videoElement = document.getElementById('videoElement');
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        
        console.log('Requesting camera access...');
        
        // Request camera access
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user' // Front camera
            },
            audio: false
        });
        
        console.log('Camera access granted');
        videoElement.srcObject = videoStream;
        
        // Wait for video to load and play
        videoElement.onloadedmetadata = () => {
            console.log('Video metadata loaded');
            document.getElementById('videoReady').textContent = 'Yes';
            videoElement.play().then(() => {
                console.log('Video playing');
                isCameraActive = true;
                document.getElementById('cameraStatus').textContent = 'Active';
                document.getElementById('streamActive').textContent = 'Yes';
                
                // Update UI
                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-block';
                
                // Start emotion detection
                startEmotionDetection();
                
                console.log('Camera started successfully');
                updateApiStatus('Camera aktif - Analisis emosi dimulai');
            }).catch(e => {
                console.error('Error playing video:', e);
                updateApiStatus('Error: Video tidak bisa diputar');
                document.getElementById('cameraStatus').textContent = 'Error';
            });
        };
        
        // Fallback timeout
        setTimeout(() => {
            if (!isCameraActive) {
                console.log('Fallback: Starting camera...');
                isCameraActive = true;
                document.getElementById('cameraStatus').textContent = 'Active (Fallback)';
                document.getElementById('streamActive').textContent = 'Yes';
                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-block';
                startEmotionDetection();
                updateApiStatus('Camera aktif - Analisis emosi dimulai');
            }
        }, 3000);
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        updateApiStatus('Error: Tidak bisa mengakses camera - ' + error.message);
        alert('Tidak bisa mengakses camera. Pastikan browser memiliki izin camera dan camera tidak digunakan aplikasi lain.');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    isCameraActive = false;
    
    // Stop emotion detection
    if (emotionDetectionInterval) {
        clearInterval(emotionDetectionInterval);
        emotionDetectionInterval = null;
    }
    
    // Update UI
    document.getElementById('startCameraBtn').style.display = 'inline-block';
    document.getElementById('stopCameraBtn').style.display = 'none';
    
    // Clear video display
    const videoFeed = document.getElementById('videoFeed');
    videoFeed.src = '';
    
    console.log('Camera stopped');
    updateApiStatus('Camera dihentikan');
}

function startEmotionDetection() {
    if (emotionDetectionInterval) {
        clearInterval(emotionDetectionInterval);
    }
    
    emotionDetectionInterval = setInterval(async () => {
        if (!isCameraActive) return;
        
        try {
            const videoElement = document.getElementById('videoElement');
            const canvasElement = document.getElementById('canvasElement');
            
            // Check if video is ready and has video
            if (videoElement.readyState < 2 || videoElement.videoWidth === 0) {
                console.log('Video not ready yet...');
                return;
            }
            
            // Capture frame from video
            const ctx = canvasElement.getContext('2d');
            ctx.drawImage(videoElement, 0, 0, 640, 480);
            
            // Convert to base64
            const base64 = canvasElement.toDataURL('image/jpeg', 0.8).split(',')[1];
            
            // Send for emotion analysis with face detection
            const result = await analyzeEmotionWithFaces(base64);
            
            if (result) {
                lastEmotion = result.overall_emotion;
                emotionHistory.push(result.overall_emotion);
                if (emotionHistory.length > 10) {
                    emotionHistory.shift(); // Keep only last 10
                }
                
                // Update display with emotion overlay and bounding boxes
                updateVideoDisplayWithEmotion(result.overall_emotion, result);
            } else {
                // Fallback - just show video without analysis
                updateVideoDisplayWithEmotion('analyzing...', null);
            }
            
        } catch (error) {
            console.error('Error in emotion detection:', error);
            // Show error on display
            updateVideoDisplayWithEmotion('error', null);
        }
    }, 3000); // Analyze every 3 seconds (slower for stability)
}

async function analyzeEmotion(base64Image) {
    try {
        const response = await fetch('/analyze_emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image })
        });
        
        if (response.ok) {
            const result = await response.json();
            return result.emotion;
        }
    } catch (error) {
        console.error('Error analyzing emotion:', error);
    }
    return null;
}

async function analyzeEmotionWithFaces(base64Image) {
    try {
        const response = await fetch('/analyze_emotion_faces', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image })
        });
        
        if (response.ok) {
            const result = await response.json();
            return result;
        }
    } catch (error) {
        console.error('Error analyzing emotion with faces:', error);
    }
    return null;
}

function updateVideoDisplayWithEmotion(emotion, faceData = null) {
    const videoElement = document.getElementById('videoElement');
    const canvasElement = document.getElementById('canvasElement');
    const videoFeed = document.getElementById('videoFeed');
    
    try {
        // Draw video frame with emotion overlay
        const ctx = canvasElement.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, 640, 480);
        
        // Draw bounding boxes for detected faces
        if (faceData && faceData.faces && Array.isArray(faceData.faces)) {
            faceData.faces.forEach((face, index) => {
                const { x, y, width, height, emotion: faceEmotion, confidence, person } = face;
                
                // Draw bounding box
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, width, height);
                
                // Draw person name or number
                ctx.fillStyle = '#00FF00';
                ctx.font = 'bold 20px Arial';
                const personText = person && person !== 'Unknown' ? person : `Person ${index + 1}`;
                ctx.fillText(personText, x, y - 10);
                
                // Draw emotion and confidence
                ctx.fillStyle = '#FFFF00';
                ctx.font = '16px Arial';
                const emotionText = `${faceEmotion} (${Math.round(confidence * 100)}%)`;
                ctx.fillText(emotionText, x, y + height + 20);
            });
        }
        
        // Add overall emotion text
        ctx.fillStyle = emotion === 'error' ? '#FF0000' : '#00FF00';
        ctx.font = '24px Arial';
        ctx.fillText(`Emotion: ${emotion}`, 30, 50);
        
        // Add recognized person info
        const recognizedPerson = faceData && faceData.recognized_person ? faceData.recognized_person : 'Unknown';
        ctx.fillStyle = '#FFFF00';
        ctx.font = '20px Arial';
        ctx.fillText(`Person: ${recognizedPerson}`, 30, 80);
        
        // Add person count
        const personCount = faceData && faceData.faces ? faceData.faces.length : 0;
        ctx.fillStyle = '#FF00FF';
        ctx.font = '18px Arial';
        ctx.fillText(`Faces: ${personCount}`, 30, 110);
        
        // Add mode indicator
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '16px Arial';
        ctx.fillText('Mode: Client-Side', 30, 140);
        
        // Update the display image
        videoFeed.src = canvasElement.toDataURL();
        
    } catch (error) {
        console.error('Error updating video display:', error);
        // Show error message
        const ctx = canvasElement.getContext('2d');
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, 640, 480);
        ctx.fillStyle = '#FF0000';
        ctx.font = '24px Arial';
        ctx.fillText('Error: Camera display issue', 50, 240);
        videoFeed.src = canvasElement.toDataURL();
    }
}

// Function to send frame to ngrok API for emotion analysis (legacy)
async function analyzeFrameWithNgrok(frame) {
    try {
        if (config.apiBaseUrl === 'http://localhost:5000') {
            return null; // Use local processing
        }
        
        // Convert frame to base64
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = frame.videoWidth || frame.width;
        canvas.height = frame.videoHeight || frame.height;
        ctx.drawImage(frame, 0, 0);
        const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        // Send to ngrok API
        const response = await fetch(`${config.apiBaseUrl}/analyze_emotion`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64 })
        });
        
        if (response.ok) {
            return await response.json();
        }
    } catch (e) {
        console.error('Error sending frame to ngrok API:', e);
    }
    return null;
}

async function refreshAnalytics() {
    try {
        const filter = document.getElementById('identityFilter').value;
        const q = filter ? ('?identity=' + encodeURIComponent(filter)) : '';
        const res = await fetch('/analytics/data' + q);
        const data = await res.json();

        document.getElementById('sessionStart').textContent = data.sessionStart ? `Session start: ${data.sessionStart}` : '';

        // Populate identity filter options on first load or when identities change
        const sel = document.getElementById('identityFilter');
        if (Array.isArray(data.identities)) {
            const current = sel.value;
            const options = [''].concat(data.identities);
            // Rebuild options
            sel.innerHTML = '';
            for (const id of options) {
                const opt = document.createElement('option');
                opt.value = id;
                opt.textContent = id || 'All';
                sel.appendChild(opt);
            }
            // Restore previous selection if available
            if (options.includes(current)) sel.value = current;
        }

        // Pie data
        const labels = Object.keys(data.counts).sort();
        const values = labels.map(l => data.counts[l]);
        const bg = labels.map(l => emotionColors[l] || '#7f8c8d');
        pieChart.data.labels = labels;
        pieChart.data.datasets[0].data = values;
        pieChart.data.datasets[0].backgroundColor = bg;
        pieChart.update();

        // Timeline: count per minute per emotion
        const buckets = {}; // { 'HH:MM': {emotion: count} }
        const ordered = (data.timeline || []).slice().sort((a,b) => a.t.localeCompare(b.t));
        const times = [];
        for (const row of ordered) {
            const t = row.t.slice(11, 16); // HH-MM
            const tLabel = t.replace('-', ':');
            if (!buckets[tLabel]) buckets[tLabel] = {};
            buckets[tLabel][row.emotion] = (buckets[tLabel][row.emotion] || 0) + 1;
            if (!times.includes(tLabel)) times.push(tLabel);
        }

        const emotions = Array.from(new Set(ordered.map(r => r.emotion))).sort();
        const datasets = emotions.map(em => ({
            label: em,
            data: times.map(tm => (buckets[tm]?.[em] || 0)),
            borderColor: emotionColors[em] || '#7f8c8d',
            backgroundColor: 'transparent'
        }));

        timelineChart.data.labels = times;
        timelineChart.data.datasets = datasets;
        timelineChart.update();

        // Summary table per identity-emotion
        const tbody = document.querySelector('#summaryTable tbody');
        tbody.innerHTML = '';
        const pic = data.perIdentityCounts || {};
        const identities = Object.keys(pic).sort();
        for (const id of identities) {
            const emoCounts = pic[id];
            const emos = Object.keys(emoCounts).sort();
            for (const em of emos) {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td style="padding:6px; border-bottom:1px solid #f0f0f0;">${id}</td>
                                <td style=\"padding:6px; border-bottom:1px solid #f0f0f0;\">${em}</td>
                                <td style=\"padding:6px; border-bottom:1px solid #f0f0f0;\">${emoCounts[em]}</td>`;
                tbody.appendChild(tr);
            }
        }
    } catch (e) {
        console.error('Analytics refresh error', e);
    }
}

// Initialize on page load
loadConfig().then(() => {
    refreshAnalytics();
    setInterval(refreshAnalytics, 5000);
});

// Add event listeners
document.getElementById('identityFilter').addEventListener('change', refreshAnalytics);
document.getElementById('startCameraBtn').addEventListener('click', startCamera);
document.getElementById('stopCameraBtn').addEventListener('click', stopCamera);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
});
