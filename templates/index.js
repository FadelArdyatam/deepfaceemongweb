   // Global configuration
   let config = {
    apiBaseUrl: 'http://localhost:5000',
    cameraAvailable: true
};

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

// Function to send frame to ngrok API for emotion analysis
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

document.getElementById('identityFilter').addEventListener('change', refreshAnalytics);
