#!/usr/bin/env node
/**
 * Hand Tracking WebSocket Server
 *
 * Receives hand landmark data from iPhone and:
 * 1. Logs it to console with visualization
 * 2. Forwards to any connected web clients
 *
 * Usage:
 *   node hand-tracking-server.js
 *   # Then connect iPhone app to ws://<your-mac-ip>:8765
 *   # Open http://localhost:8766 for web visualization
 */

const { WebSocketServer, WebSocket } = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');

// NOTE: 8765/8767 are often used by other local tooling. Default to 8768 to avoid collisions.
// Override via env:
//   HAND_TRACKING_PHONE_PORT=8765 HAND_TRACKING_WEB_PORT=8766 node hand-tracking-server.js
const PHONE_PORT = Number(process.env.HAND_TRACKING_PHONE_PORT || 8768);  // iPhone connects here
const WEB_PORT = Number(process.env.HAND_TRACKING_WEB_PORT || 8766);      // Web visualization

// Store latest hand data
let latestHandData = null;
let webClients = new Set();
let phoneClients = new Set();

function getLocalIps() {
    // Get local IP for convenience
    const { networkInterfaces } = require('os');
    const nets = networkInterfaces();
    const ips = [];
    for (const name of Object.keys(nets)) {
        for (const net of nets[name]) {
            if (net.family === 'IPv4' && !net.internal) {
                ips.push(net.address);
            }
        }
    }
    return ips;
}

function broadcastStatus() {
    const msg = JSON.stringify({
        type: 'bridge_status',
        phonePort: PHONE_PORT,
        webPort: WEB_PORT,
        phoneConnected: phoneClients.size > 0,
        ips: getLocalIps(),
        lastHandFrameAt: latestHandData?.frameTimestamp || null,
    });
    for (const client of webClients) {
        if (client.readyState === WebSocket.OPEN) {
            client.send(msg);
        }
    }
}

// ============ iPhone WebSocket Server ============

const phoneServer = new WebSocketServer({ port: PHONE_PORT });

console.log(`📱 iPhone WebSocket server listening on ws://0.0.0.0:${PHONE_PORT}`);
console.log(`   Connect your iPhone app to: ws://<your-mac-ip>:${PHONE_PORT}`);

phoneServer.on('connection', (ws, req) => {
    console.log(`\n✅ iPhone connected from ${req.socket.remoteAddress}`);
    phoneClients.add(ws);
    broadcastStatus();

    ws.on('message', (data) => {
        try {
            const message = JSON.parse(data.toString());
            latestHandData = message;

            // Log hand detection
            if (message.hands && message.hands.length > 0) {
                const hand = message.hands[0];
                const wrist = hand.landmarks['VNHLKJWRIST'];
                const indexTip = hand.landmarks['VNHLKJINDEXTIP'];
                const hasDepth = hand.hasLiDARDepth;

                // Simple ASCII visualization
                process.stdout.write('\r');
                process.stdout.write(`🖐️  Hands: ${message.hands.length} | `);
                if (wrist) {
                    process.stdout.write(`Wrist: (${wrist.x.toFixed(2)}, ${wrist.y.toFixed(2)}`);
                    if (hasDepth && wrist.z !== 0) {
                        process.stdout.write(`, ${wrist.z.toFixed(2)}m`);
                    }
                    process.stdout.write(') | ');
                }
                if (indexTip) {
                    process.stdout.write(`Index: (${indexTip.x.toFixed(2)}, ${indexTip.y.toFixed(2)}`);
                    if (hasDepth && indexTip.z !== 0) {
                        process.stdout.write(`, ${indexTip.z.toFixed(2)}m`);
                    }
                    process.stdout.write(')');
                }
                process.stdout.write(`  [LiDAR: ${hasDepth ? '✓' : '✗'}]`);
                process.stdout.write('    '); // Clear any trailing chars
            }

            // Forward to web clients
            const jsonStr = JSON.stringify(message);
            for (const client of webClients) {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(jsonStr);
                }
            }

        } catch (e) {
            console.error('Parse error:', e.message);
        }
    });

    ws.on('close', () => {
        console.log('\n📱 iPhone disconnected');
        phoneClients.delete(ws);
        broadcastStatus();
    });

    ws.on('error', (err) => {
        console.error('iPhone WebSocket error:', err.message);
    });
});

// ============ Web Visualization Server ============

const webHtml = `<!DOCTYPE html>
<html>
<head>
    <title>Hand Tracking Test</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            margin-bottom: 20px;
            background: linear-gradient(90deg, #4ecdc4, #f72585);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat {
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stat-label { font-size: 12px; color: #888; }
        .stat-value { font-size: 24px; font-weight: bold; }
        .connected { color: #4ade80; }
        .disconnected { color: #f87171; }
        #canvas {
            background: rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 16px;
        }
        .info {
            margin-top: 20px;
            color: #888;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>🖐️ Hand Tracking Test</h1>

    <div class="status">
        <div class="stat">
            <div class="stat-label">Status</div>
            <div class="stat-value" id="status">Connecting...</div>
        </div>
        <div class="stat">
            <div class="stat-label">Hands</div>
            <div class="stat-value" id="hands">0</div>
        </div>
        <div class="stat">
            <div class="stat-label">LiDAR</div>
            <div class="stat-value" id="lidar">-</div>
        </div>
        <div class="stat">
            <div class="stat-label">FPS</div>
            <div class="stat-value" id="fps">0</div>
        </div>
    </div>

    <canvas id="canvas" width="800" height="600"></canvas>

    <div class="info">
        iPhone should connect to: ws://&lt;your-mac-ip&gt;:8765
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const statusEl = document.getElementById('status');
        const handsEl = document.getElementById('hands');
        const lidarEl = document.getElementById('lidar');
        const fpsEl = document.getElementById('fps');

        let frameCount = 0;
        let lastFpsTime = Date.now();

        // Finger connections
        const fingers = [
            ['VNHLKJWRIST', 'VNHLKJTHUMBCMC', 'VNHLKJTHUMBMP', 'VNHLKJTHUMBIP', 'VNHLKJTHUMBTIP'],
            ['VNHLKJWRIST', 'VNHLKJINDEXMCP', 'VNHLKJINDEXPIP', 'VNHLKJINDEXDIP', 'VNHLKJINDEXTIP'],
            ['VNHLKJWRIST', 'VNHLKJMIDDLEMCP', 'VNHLKJMIDDLEPIP', 'VNHLKJMIDDLEDIP', 'VNHLKJMIDDLETIP'],
            ['VNHLKJWRIST', 'VNHLKJRINGMCP', 'VNHLKJRINGPIP', 'VNHLKJRINGDIP', 'VNHLKJRINGTIP'],
            ['VNHLKJWRIST', 'VNHLKJLITTLEMCP', 'VNHLKJLITTLEPIP', 'VNHLKJLITTLEDIP', 'VNHLKJLITTLETIP'],
        ];

        function connect() {
            const ws = new WebSocket('ws://localhost:${WEB_PORT}/ws');

            ws.onopen = () => {
                statusEl.textContent = 'Connected';
                statusEl.className = 'stat-value connected';
            };

            ws.onclose = () => {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'stat-value disconnected';
                setTimeout(connect, 1000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                render(data);
                updateFPS();
            };
        }

        function updateFPS() {
            frameCount++;
            const now = Date.now();
            if (now - lastFpsTime >= 1000) {
                fpsEl.textContent = frameCount;
                frameCount = 0;
                lastFpsTime = now;
            }
        }

        function render(data) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!data.hands || data.hands.length === 0) {
                handsEl.textContent = '0';
                return;
            }

            handsEl.textContent = data.hands.length;
            lidarEl.textContent = data.hands[0].hasLiDARDepth ? '✓' : '✗';
            lidarEl.style.color = data.hands[0].hasLiDARDepth ? '#4ade80' : '#f87171';

            const colors = ['#4ecdc4', '#f72585'];

            data.hands.forEach((hand, handIdx) => {
                const color = colors[handIdx % colors.length];
                const landmarks = hand.landmarks;

                // Draw fingers
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.lineCap = 'round';

                fingers.forEach(finger => {
                    ctx.beginPath();
                    let started = false;

                    finger.forEach(jointName => {
                        const lm = landmarks[jointName];
                        if (lm) {
                            const x = lm.x * canvas.width;
                            const y = lm.y * canvas.height;
                            if (started) {
                                ctx.lineTo(x, y);
                            } else {
                                ctx.moveTo(x, y);
                                started = true;
                            }
                        }
                    });

                    ctx.stroke();
                });

                // Draw palm
                ctx.strokeStyle = color + '80';
                ctx.beginPath();
                let started = false;
                ['VNHLKJINDEXMCP', 'VNHLKJMIDDLEMCP', 'VNHLKJRINGMCP', 'VNHLKJLITTLEMCP'].forEach(j => {
                    const lm = landmarks[j];
                    if (lm) {
                        const x = lm.x * canvas.width;
                        const y = lm.y * canvas.height;
                        if (started) ctx.lineTo(x, y);
                        else { ctx.moveTo(x, y); started = true; }
                    }
                });
                ctx.stroke();

                // Draw joints
                Object.entries(landmarks).forEach(([name, lm]) => {
                    const x = lm.x * canvas.width;
                    const y = lm.y * canvas.height;
                    const isTip = name.includes('TIP');
                    const radius = isTip ? 8 : 5;

                    ctx.beginPath();
                    ctx.arc(x, y, radius, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();

                    // Show depth at fingertips
                    if (isTip && lm.z !== 0) {
                        ctx.fillStyle = 'white';
                        ctx.font = '10px monospace';
                        ctx.fillText(lm.z.toFixed(2) + 'm', x + 10, y - 5);
                    }
                });
            });
        }

        connect();
    </script>
</body>
</html>`;

// HTTP + WebSocket server for web visualization
const httpServer = http.createServer((req, res) => {
    if (req.url === '/' || req.url === '/index.html') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(webHtml);
    } else {
        res.writeHead(404);
        res.end('Not found');
    }
});

const webWss = new WebSocketServer({ server: httpServer, path: '/ws' });

webWss.on('connection', (ws) => {
    console.log('🌐 Web client connected');
    webClients.add(ws);
    // Send current status immediately
    try {
        ws.send(JSON.stringify({
            type: 'bridge_status',
            phonePort: PHONE_PORT,
            webPort: WEB_PORT,
            phoneConnected: phoneClients.size > 0,
            ips: getLocalIps(),
            lastHandFrameAt: latestHandData?.frameTimestamp || null,
        }));
    } catch {}

    ws.on('close', () => {
        console.log('🌐 Web client disconnected');
        webClients.delete(ws);
    });
});

httpServer.listen(WEB_PORT, () => {
    console.log(`\n🌐 Web visualization at http://localhost:${WEB_PORT}`);
});

console.log('\n📡 Your Mac IP addresses:');
for (const ip of getLocalIps()) {
    console.log(`   ${ip}`);
}
console.log('\n');
