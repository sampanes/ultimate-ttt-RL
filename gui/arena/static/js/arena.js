
// ============================================
//  STATE
// ============================================

const AGENT_COLORS = [
    '#3b9eff', '#22d3ee', '#22c55e', '#f59e0b', '#ef4444',
    '#ec4899', '#14b8a6', '#f97316', '#a78bfa', '#34d399'
];

// ============================================
//  INLINE SVG ICONS  (all-ASCII source, no emoji -> no cp1252 crash path;
//  rendered as crisp vector glyphs. currentColor = inherits the text color
//  of whatever container they sit in.)
// ============================================

const _SVG = (inner) =>
    `<svg class="ico-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`;

const ICONS = {
    grid: _SVG('<path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>'),
    clock: _SVG('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'),
    pause: _SVG('<rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/>'),
    play: _SVG('<polygon points="6 4 20 12 6 20 6 4"/>'),
    plus: _SVG('<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>'),
    activity: _SVG('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
    trophy: _SVG('<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>'),
    'trending-up': _SVG('<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>'),
    list: _SVG('<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>'),
    trash: _SVG('<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/>'),
    check: _SVG('<polyline points="20 6 9 17 4 12"/>'),
    x: _SVG('<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>'),
    info: _SVG('<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'),
    zap: _SVG('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'),
    shield: _SVG('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'),
    package: _SVG('<line x1="16.5" y1="9.4" x2="7.5" y2="4.21"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>'),
    power: _SVG('<path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/>'),
    link: _SVG('<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'),
};

function icon(name) { return ICONS[name] || ''; }

// Replace every <span data-ico="name"> placeholder in the static markup with its SVG.
function hydrateIcons(root = document) {
    root.querySelectorAll('[data-ico]').forEach(el => { el.innerHTML = icon(el.dataset.ico); });
}

// Event-log icon is derived from the event TYPE here, client-side -- so it stays clean
// regardless of any legacy "[^]"/"[#]" text placeholder the backend wrote into the state.
const EVENT_ICONS = {
    spawn: 'plus', retire: 'trash', elo: 'trending-up', pause: 'pause',
    resume: 'play', system: 'info', chunk: 'package', start: 'power', stop: 'power',
};

let appState = {
    status: 'Stopped',          // "Running" | "Paused" | "Stopped"
    pendingRetireAgent: null,
    agents: [],                 // last agent list from /api/status
    events: [],                 // last events list from /api/events (for the Live tab)
    retirementThreshold: 30,    // from /api/status
};

// Tracks events already rendered so polls don't duplicate them.
// Keys are "isoTime|msg". Capped at 200 entries (mirrors server-side cap of 100 events).
const _seenEventKeys = new Set();

// ============================================
//  API
// ============================================

async function apiFetch(endpoint, method = 'GET', body = null) {
    const opts = { method, headers: {} };
    if (body) {
        opts.body = JSON.stringify(body);
        opts.headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(endpoint, opts);
    return res.json();
}

// ============================================
//  CHART SETUP
// ============================================

let eloChart;

function initChart() {
    const ctx = document.getElementById('eloChart').getContext('2d');
    eloChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: '#8b97a8',
                        font: { size: 11 },
                        boxWidth: 12,
                        padding: 10,
                        filter: (item) => item.text !== '__hidden__'
                    }
                },
                tooltip: {
                    backgroundColor: '#131722',
                    borderColor: '#29313f',
                    borderWidth: 1,
                    titleColor: '#e6edf3',
                    bodyColor: '#8b97a8',
                    padding: 10,
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y} ELO`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(41,49,63,0.5)' },
                    ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 6 }
                },
                y: {
                    grid: { color: 'rgba(41,49,63,0.5)' },
                    ticks: { color: '#64748b', font: { size: 10 } },
                    title: { display: true, text: 'ELO Rating', color: '#64748b', font: { size: 11 } }
                }
            },
            animation: { duration: 400 },
            elements: {
                point: { radius: 2, hoverRadius: 5 },
                line: { tension: 0.35, borderWidth: 2 }
            }
        }
    });
}

function updateChart(historyData) {
    const activeAgents = appState.agents.filter(a => a.status !== 'retired').slice(0, 8);

    const datasets = activeAgents.map((agent, idx) => {
        const color = AGENT_COLORS[idx % AGENT_COLORS.length];
        const data = historyData[agent.name] || [];
        return {
            label: agent.name,
            data,
            borderColor: color,
            backgroundColor: color + '18',
            fill: false,
            pointBackgroundColor: color
        };
    });

    eloChart.data.datasets = datasets;

    const maxLen = Math.max(...datasets.map(d => d.data.length), 1);
    eloChart.data.labels = Array.from({ length: maxLen }, (_, i) => {
        const back = maxLen - 1 - i;
        return back === 0 ? 'Now' : `T-${back}`;
    });

    eloChart.update('none');
}

// ============================================
//  LEADERBOARD RENDER
// ============================================

function getRowClass(agent) {
    if (agent.status === 'retired') return 'row-retired';
    const pct = agent.chunks_stagnant / appState.retirementThreshold;
    if (pct > 0.5) return 'row-stagnant';
    return 'row-active';
}

function getStatusPill(agent) {
    if (agent.status === 'retired') return `<span class="status-pill pill-retired"><span class="pill-dot"></span> Retired</span>`;
    const pct = agent.chunks_stagnant / appState.retirementThreshold;
    if (pct > 0.5) return `<span class="status-pill pill-stagnant"><span class="pill-dot"></span> Stagnant</span>`;
    return `<span class="status-pill pill-active"><span class="pill-dot"></span> Active</span>`;
}

function getRankBadge(rank) {
    if (rank === null) return `<span class="rank-badge rank-n">--</span>`;
    const cls = rank <= 3 ? `rank-${rank}` : 'rank-n';
    return `<span class="rank-badge ${cls}">${rank}</span>`;
}

function getStagnantBar(count) {
    const pct = Math.min(100, Math.round(count / appState.retirementThreshold * 100));
    let color = '#22c55e';
    if (pct > 75) color = '#ef4444';
    else if (pct > 50) color = '#eab308';
    else if (pct > 25) color = '#f59e0b';
    return `
    <div class="stagnant-bar-wrap">
      <div class="stagnant-bar-bg">
        <div class="stagnant-bar-fill" style="width:${pct}%; background:${color};"></div>
      </div>
      <span class="stagnant-val">${pct}%</span>
    </div>`;
}

function renderLeaderboard(agents) {
    const tbody = document.getElementById('leaderboardBody');
    if (!agents || agents.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--clr-muted);padding:2rem;">No agents found</td></tr>';
        return;
    }

    tbody.innerHTML = agents.map((agent, idx) => {
        const color = AGENT_COLORS[idx % AGENT_COLORS.length];
        const rowClass = getRowClass(agent);
        const retireBtn = agent.status !== 'retired'
            ? `<button class="btn btn-danger" onclick="showRetireConfirm(${agent.id}, '${agent.name}')">${icon('trash')} Retire</button>`
            : `<span style="font-size:0.78rem;color:var(--clr-muted);">--</span>`;

        return `
      <tr class="${rowClass}">
        <td>${getRankBadge(agent.rank)}</td>
        <td>
          <div style="display:flex;align-items:center;gap:0;">
            <span class="agent-dot" style="background:${color};"></span>
            <div>
              <span style="font-weight:600;">${agent.name}</span>
              <div style="font-size:0.72rem;color:var(--clr-muted);">${agent.arch || ''}</div>
            </div>
          </div>
        </td>
        <td><span class="elo-value" style="color:${color};">${agent.elo.toLocaleString()}</span></td>
        <td style="min-width:120px;">${getStagnantBar(agent.chunks_stagnant)}</td>
        <td>${getStatusPill(agent)}</td>
        <td>${retireBtn}</td>
      </tr>`;
    }).join('');

    document.getElementById('leaderboardUpdated').textContent = 'Updated ' + new Date().toLocaleTimeString();
}

// ============================================
//  STATS CARDS
// ============================================

function updateStats(agents) {
    const total = agents.length;
    const pctThreshold = 0.5;
    const active = agents.filter(a => a.status === 'active' &&
        a.chunks_stagnant / appState.retirementThreshold <= pctThreshold).length;
    const stagnant = agents.filter(a => a.status === 'active' &&
        a.chunks_stagnant / appState.retirementThreshold > pctThreshold).length;
    const retired = agents.filter(a => a.status === 'retired').length;

    document.getElementById('statTotal').textContent = total;
    document.getElementById('statActive').textContent = active;
    document.getElementById('statStagnant').textContent = stagnant;
    document.getElementById('statRetired').textContent = retired;
}

// Count a number up/down to a new value -- gives the headline stats a live "ticking"
// feel when ELO or game count changes. Purely visual; no state stored.
function animateValue(el, to) {
    if (!el) return;
    const from = Number(String(el.dataset.raw ?? el.textContent).replace(/[^0-9.-]/g, '')) || 0;
    to = Number(to) || 0;
    el.dataset.raw = to;
    if (from === to) { el.textContent = to.toLocaleString(); return; }
    const steps = 18, start = performance.now(), dur = 600;
    function tick(now) {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);           // ease-out cubic
        const v = Math.round(from + (to - from) * eased);
        el.textContent = v.toLocaleString();
        if (t < 1) requestAnimationFrame(tick);
        else el.textContent = to.toLocaleString();
    }
    requestAnimationFrame(tick);
}

// ============================================
//  EVENT LOG
// ============================================

function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// Third arg is ignored (legacy text-icon slot); the glyph is chosen from the type.
function addEvent(type, _legacyIcon, msg) {
    const log = document.getElementById('eventLog');
    const item = document.createElement('div');
    item.className = `event-item event-${type}`;
    item.innerHTML = `
    <span class="event-icon">${icon(EVENT_ICONS[type] || 'info')}</span>
    <div class="event-content">
      <div class="event-msg">${msg}</div>
      <div class="event-time">${formatTime(new Date())}</div>
    </div>`;
    log.insertBefore(item, log.firstChild);
    while (log.children.length > 50) log.removeChild(log.lastChild);
}

function clearLog() {
    document.getElementById('eventLog').innerHTML = '';
    addEvent('system', null, 'Event log cleared.');
}

// ============================================
//  TOAST
// ============================================

function showToast(msg, type = 'success', duration = 3000) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const tColor = type === 'success' ? 'var(--clr-green)' : 'var(--clr-red)';
    toast.innerHTML = `<span style="color:${tColor};">${icon(type === 'success' ? 'check' : 'x')}</span> ${msg}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============================================
//  LIVE TAB  (Hall of Fame + Now Training)
//  Pure derived views over data already polled -- no new storage, no new files.
// ============================================

// Top all-time by PEAK ELO (best_elo), across active AND retired agents.
function renderHallOfFame(agents) {
    const body = document.getElementById('hofBody');
    if (!body) return;
    const ranked = (agents || [])
        .slice()
        .sort((a, b) => (b.best_elo ?? b.elo) - (a.best_elo ?? a.elo))
        .slice(0, 8);

    if (ranked.length === 0) {
        body.innerHTML = '<div class="hof-empty">No agents yet.</div>';
        return;
    }

    body.innerHTML = ranked.map((a, i) => {
        const peak = Math.round(a.best_elo ?? a.elo);
        const color = AGENT_COLORS[i % AGENT_COLORS.length];
        const medal = i === 0 ? 'hof-gold' : i === 1 ? 'hof-silver' : i === 2 ? 'hof-bronze' : 'hof-plain';
        const crown = i === 0 ? `<span class="hof-crown" style="color:#facc15;">${icon('trophy')}</span>` : '';
        const live = a.status === 'active'
            ? '<span class="hof-live">LIVE</span>'
            : '<span class="hof-gone">retired</span>';
        return `
        <div class="hof-row ${medal}">
          <span class="hof-rank">${i + 1}</span>
          <span class="agent-dot" style="background:${color};"></span>
          <div class="hof-name">
            <div style="font-weight:600;">${crown}${a.name}</div>
            <div class="hof-arch">${a.arch || ''}</div>
          </div>
          <div class="hof-elo" style="color:${color};">${peak.toLocaleString()}<span class="hof-elo-label">peak</span></div>
          ${live}
        </div>`;
    }).join('');
}

// Parse the recent "chunk" events into per-agent latest results -> live W/L/D cards.
// Event msg format: "<name> chunk <n>: W=<w> L=<l> D=<d> WR=<wr>% ELO=<elo>"
const _CHUNK_RE = /^(.+?) chunk (\d+): W=(\d+) L=(\d+) D=(\d+) WR=([\d.]+)% ELO=(-?\d+)/;
let _lastChunkByAgent = {};

function renderNowTraining(events) {
    const wrap = document.getElementById('nowTraining');
    if (!wrap) return;

    const latest = {};   // name -> parsed result (events are oldest-first, so last wins)
    for (const ev of (events || [])) {
        if (ev.type !== 'chunk' || !ev.msg) continue;
        const m = _CHUNK_RE.exec(ev.msg);
        if (!m) continue;
        latest[m[1]] = {
            name: m[1], chunk: +m[2], w: +m[3], l: +m[4], d: +m[5],
            wr: parseFloat(m[6]), elo: +m[7],
        };
    }

    const rows = Object.values(latest).sort((a, b) => b.elo - a.elo);
    if (rows.length === 0) {
        wrap.innerHTML = '<div class="hof-empty">Waiting for the next chunk result...</div>';
        return;
    }

    wrap.innerHTML = rows.map((r, i) => {
        const color = AGENT_COLORS[i % AGENT_COLORS.length];
        const flash = _lastChunkByAgent[r.name] !== undefined && r.chunk > _lastChunkByAgent[r.name] ? 'nt-flash' : '';
        return `
        <div class="nt-card ${flash}">
          <div class="nt-head">
            <span class="nt-name"><span class="agent-dot" style="background:${color};"></span>${r.name}</span>
            <span class="nt-chunk">chunk ${r.chunk.toLocaleString()}</span>
          </div>
          <div class="nt-elo" style="color:${color};">${r.elo.toLocaleString()} <span class="nt-elo-label">ELO</span></div>
          <div class="nt-bar"><div class="nt-bar-fill" style="width:${Math.max(0, Math.min(100, r.wr)).toFixed(1)}%;"></div></div>
          <div class="nt-wld">
            <span class="nt-w">${r.w} W</span>
            <span class="nt-l">${r.l} L</span>
            <span class="nt-d">${r.d} D</span>
            <span class="nt-wr">${r.wr.toFixed(1)}% WR</span>
          </div>
        </div>`;
    }).join('');

    _lastChunkByAgent = {};
    for (const r of rows) _lastChunkByAgent[r.name] = r.chunk;
}

// ============================================
//  MODAL HELPERS
// ============================================

function openModal(id) { document.getElementById(id).classList.add('active'); }
function closeModal(id) { document.getElementById(id).classList.remove('active'); }

document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.classList.remove('active');
    });
});

// ============================================
//  PAUSE / RESUME
// ============================================

function showPauseConfirm() {
    const isRunning = appState.status === 'Running';
    document.getElementById('pauseModalTitle').innerHTML = isRunning
        ? `${icon('pause')} Pause Training?`
        : `${icon('play')} Resume Training?`;
    document.getElementById('pauseModalDesc').textContent = isRunning
        ? 'This will pause all active training matches. Agents retain their current ELO scores. You can resume at any time.'
        : 'This will resume all training matches and ELO scoring. The arena will become live again.';
    const btn = document.getElementById('pauseConfirmBtn');
    btn.innerHTML = isRunning ? `${icon('pause')} Confirm Pause` : `${icon('play')} Confirm Resume`;
    btn.className = isRunning ? 'btn btn-confirm-pause' : 'btn btn-confirm-resume';
    openModal('pauseModal');
}

async function executePauseResume() {
    closeModal('pauseModal');
    const isRunning = appState.status === 'Running';
    const endpoint = isRunning ? '/api/control/pause' : '/api/control/resume';
    try {
        const result = await apiFetch(endpoint, 'POST');
        if (result.success) {
            const action = isRunning ? 'paused' : 'resumed';
            showToast(`Training ${action} successfully.`);
            addEvent(isRunning ? 'pause' : 'resume', null, `Training ${action} by operator.`);
            pollStatus();
        }
    } catch (e) {
        showToast('Request failed.', 'error');
    }
}

function updateHeaderState() {
    const badge = document.getElementById('statusBadge');
    const text = document.getElementById('statusText');
    const btn = document.getElementById('pauseResumeBtn');

    if (appState.status === 'Running') {
        badge.className = 'status-badge running';
        text.textContent = 'Running';
        btn.className = 'btn btn-pause';
        btn.innerHTML = `${icon('pause')} Pause Training`;
    } else if (appState.status === 'Paused') {
        badge.className = 'status-badge paused';
        text.textContent = 'Paused';
        btn.className = 'btn btn-resume';
        btn.innerHTML = `${icon('play')} Resume Training`;
    } else {
        badge.className = 'status-badge paused';
        text.textContent = 'Stopped';
        btn.className = 'btn btn-resume';
        btn.innerHTML = `${icon('play')} Resume Training`;
    }
}

// ============================================
//  RETIRE
// ============================================

function showRetireConfirm(id, name) {
    appState.pendingRetireAgent = { id, name };
    document.getElementById('retireModalDesc').textContent =
        `Agent "${name}" will be permanently retired and removed from future matchmaking. This action cannot be undone.`;
    openModal('retireModal');
}

async function executeRetire() {
    closeModal('retireModal');
    const { id, name } = appState.pendingRetireAgent;
    try {
        const result = await apiFetch(`/api/control/retire/${id}`, 'POST');
        if (result.success) {
            showToast(`Agent ${name} has been retired.`);
            addEvent('retire', null, `Agent ${name} retired by operator.`);
            pollStatus();
            pollHistory();
        } else {
            showToast(result.error || 'Retire failed.', 'error');
        }
    } catch (e) {
        showToast('Request failed.', 'error');
    }
}

// ============================================
//  SPAWN
// ============================================

function showSpawnModal() {
    document.getElementById('spawnName').value = '';
    document.getElementById('spawnConfig').value = 'clone_best';
    document.getElementById('spawnElo').value = '1000';
    openModal('spawnModal');
}

async function executeSpawn() {
    const name = document.getElementById('spawnName').value.trim();
    const configVal = document.getElementById('spawnConfig').value;
    if (!name) {
        document.getElementById('spawnName').style.borderColor = 'var(--clr-red)';
        setTimeout(() => document.getElementById('spawnName').style.borderColor = '', 1500);
        return;
    }
    closeModal('spawnModal');

    // Parse "clone_best" or "random_3" -> { mode, depth }
    let mode, depth = null;
    if (configVal === 'clone_best') {
        mode = 'clone_best';
    } else {
        mode = 'random';
        const parts = configVal.split('_');
        if (parts.length === 2 && !isNaN(parts[1])) depth = parseInt(parts[1]);
    }

    const elo = parseInt(document.getElementById('spawnElo').value, 10) || 1000;
    try {
        const result = await apiFetch('/api/control/spawn', 'POST', { name, mode, depth, elo });
        if (result.success) {
            showToast(`Agent "${result.agent.name}" spawned!`);
            addEvent('spawn', null, `New agent "${result.agent.name}" deployed (${result.agent.arch}).`);
            pollStatus();
            pollHistory();
        } else {
            showToast(result.error || 'Spawn failed.', 'error');
        }
    } catch (e) {
        showToast('Request failed.', 'error');
    }
}

// ============================================
//  POLLING LOOP
// ============================================

async function pollStatus() {
    try {
        const data = await apiFetch('/api/status');
        appState.status = data.status;
        appState.agents = data.agents || [];
        appState.retirementThreshold = data.retirement_threshold || 30;

        document.getElementById('uptimeDisplay').textContent = data.uptime;
        updateHeaderState();
        updateStats(data.agents);
        animateValue(document.getElementById('statBestElo'), data.best_elo || 0);
        animateValue(document.getElementById('statGames'), data.total_games || 0);
        renderLeaderboard(data.agents);
        renderHallOfFame(appState.agents);
    } catch (e) {
        console.error('[Arena] Status poll error:', e);
    }
}

async function pollHistory() {
    try {
        const data = await apiFetch('/api/history');
        updateChart(data.history);
    } catch (e) {
        console.error('[Arena] History poll error:', e);
    }
}

async function pollEvents() {
    try {
        const data = await apiFetch('/api/events');
        const events = data.events || [];
        appState.events = events;
        renderNowTraining(events);
        // Events are oldest-first; iterate in order so newest ends up on top.
        for (const ev of events) {
            const key = ev.time + '|' + ev.msg;
            if (!_seenEventKeys.has(key)) {
                _seenEventKeys.add(key);
                addEvent(ev.type, ev.icon, ev.msg);
                // Prevent unbounded growth -- trim oldest keys if set exceeds server cap.
                if (_seenEventKeys.size > 200) {
                    _seenEventKeys.delete(_seenEventKeys.values().next().value);
                }
            }
        }
    } catch (e) {
        console.error('[Arena] Events poll error:', e);
    }
}

// ============================================
//  TRAINING METRICS TAB (folds the old standalone live_metrics_plot in)
//  Reads /api/metrics (raw JSONL of loss/epsilon/winrate from a single-net run).
// ============================================

let trainingChart = null;
let _metricsTimer = null;
let _metricsLastLen = -1;
let _metricsInFlight = false;

function _movingAverage(data, windowSize = 8) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - windowSize + 1);
        const w = data.slice(start, i + 1);
        result.push(w.reduce((s, v) => s + v, 0) / w.length);
    }
    return result;
}

function _downsample(arr, maxPoints) {
    if (arr.length <= maxPoints) return arr.slice();
    const stride = Math.ceil(arr.length / maxPoints);
    const out = [];
    for (let i = 0; i < arr.length; i += stride) {
        const s = arr.slice(i, i + stride).filter(v => !Number.isNaN(v));   // NaN-safe: a bucket straddling old(no stage/elo)->new keeps its real values; all-NaN -> NaN (true gap)
        out.push(s.length ? s.reduce((a, b) => a + b, 0) / s.length : NaN);
    }
    return out;
}

function resetTrainingZoom() {
    if (trainingChart && trainingChart.resetZoom) trainingChart.resetZoom();
}

function initTrainingChart() {
    const ctx = document.getElementById('trainingCanvas').getContext('2d');
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Loss', yAxisID: 'yLoss',      // unbounded -> its own right axis
                    borderColor: '#ef4444', pointBackgroundColor: '#ef4444',
                    data: [], fill: false, showLine: false, pointRadius: 2, pointHoverRadius: 4, borderWidth: 0
                },
                {
                    label: 'Epsilon', yAxisID: 'y',
                    borderColor: '#22d3ee', pointBackgroundColor: '#22d3ee',
                    data: [], fill: false, borderWidth: 2, pointRadius: 0
                },
                {
                    label: 'Winrate', yAxisID: 'y',
                    borderColor: '#22c55e', pointBackgroundColor: '#22c55e',
                    data: [], fill: false, borderWidth: 2, pointRadius: 0
                },
                {
                    label: 'Stage', yAxisID: 'yStage',
                    borderColor: '#a78bfa', pointBackgroundColor: '#a78bfa',
                    data: [], fill: false, borderWidth: 2, pointRadius: 0,
                    stepped: true, spanGaps: false
                },
                {
                    label: 'ELO', yAxisID: 'yElo',
                    borderColor: '#f59e0b', pointBackgroundColor: '#f59e0b',
                    data: [], fill: false, borderWidth: 2, pointRadius: 0,
                    spanGaps: false
                },
                {
                    label: 'Value EV', yAxisID: 'y',     // explained variance in (-inf,1]; shares the 0-1 axis
                    borderColor: '#ec4899', pointBackgroundColor: '#ec4899',
                    data: [], fill: false, borderWidth: 2, pointRadius: 0,
                    borderDash: [4, 3], spanGaps: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(41,49,63,0.5)' },
                    ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 12 },
                    title: { display: true, text: 'training step (downsampled)', color: '#64748b', font: { size: 11 } }
                },
                y: {
                    position: 'left', beginAtZero: true, max: 1,
                    grid: { color: 'rgba(41,49,63,0.5)' },
                    ticks: { color: '#64748b', font: { size: 10 } },
                    title: { display: true, text: 'epsilon / win rate', color: '#64748b', font: { size: 11 } }
                },
                yLoss: {
                    position: 'right', beginAtZero: true,
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#64748b', font: { size: 10 } },
                    title: { display: true, text: 'loss', color: '#ef4444', font: { size: 11 } }
                },
                yStage: {
                    position: 'left', min: 0, max: 8, offset: true,
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#a78bfa', font: { size: 10 }, stepSize: 1, precision: 0 },
                    title: { display: true, text: 'stage', color: '#a78bfa', font: { size: 11 } }
                },
                yElo: {
                    position: 'right', offset: true,
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#f59e0b', font: { size: 10 } },
                    title: { display: true, text: 'ELO', color: '#f59e0b', font: { size: 11 } }
                }
            },
            plugins: {
                legend: { labels: { usePointStyle: true, color: '#8b97a8', font: { size: 12 }, padding: 14 } },
                tooltip: {
                    backgroundColor: '#131722', borderColor: '#29313f', borderWidth: 1,
                    titleColor: '#e6edf3', bodyColor: '#8b97a8', padding: 10
                },
                zoom: {
                    pan: { enabled: true, mode: 'x', modifierKey: 'ctrl' },
                    zoom: { pinch: { enabled: true }, wheel: { enabled: true, modifierKey: 'ctrl' }, mode: 'x' }
                }
            }
        }
    });
}

async function fetchMetrics() {
    if (_metricsInFlight) return;
    _metricsInFlight = true;
    try {
        const res = await fetch('/api/metrics', { cache: 'no-store' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const text = await res.text();

        if (text.length === _metricsLastLen) return;   // append-only log unchanged -> skip all work
        _metricsLastLen = text.length;

        const summaryEl = document.getElementById('metricsSummary');
        const lines = text.trim().split('\n').filter(Boolean);
        if (lines.length === 0) {
            if (summaryEl) summaryEl.textContent = 'no data yet';
            return;
        }
        const data = lines.map(l => JSON.parse(l));

        const MAX_POINTS = 5000;
        const loss = _downsample(_movingAverage(data.map(d => d.loss), 8), MAX_POINTS);
        const eps = _downsample(data.map(d => d.epsilon), MAX_POINTS);
        const win = _downsample(data.map(d => d.winrate), MAX_POINTS);
        const stage = _downsample(data.map(d => d.stage ?? NaN), MAX_POINTS);   // missing on old logs -> gap (NOT moving-averaged)
        const elo = _downsample(data.map(d => d.elo ?? NaN), MAX_POINTS);
        const ev = _downsample(data.map(d => d.explained_var ?? NaN), MAX_POINTS);  // value-head quality; gap on old logs

        if (!trainingChart) initTrainingChart();
        trainingChart.data.labels = loss.map((_, i) => i);
        trainingChart.data.datasets[0].data = loss;
        trainingChart.data.datasets[1].data = eps;
        trainingChart.data.datasets[2].data = win;
        trainingChart.data.datasets[3].data = stage;
        trainingChart.data.datasets[4].data = elo;
        trainingChart.data.datasets[5].data = ev;
        trainingChart.update();

        const last = data[data.length - 1];
        if (summaryEl) {
            // stage 0 is valid -> test !== null, not ?? (which would treat 0 as missing).
            const stageStr = (last.stage ?? null) !== null ? last.stage : '--';
            const eloStr   = (last.elo   ?? null) !== null ? Math.round(last.elo) : '--';
            const evStr    = (last.explained_var ?? null) !== null ? last.explained_var.toFixed(2) : '--';
            summaryEl.textContent =
                `loss ${(last.loss ?? 0).toFixed(3)} . epsilon=${(last.epsilon ?? 0).toFixed(3)} . win ${((last.winrate ?? 0) * 100).toFixed(1)}% . stage ${stageStr} . ELO ${eloStr} . EV ${evStr}`;
        }
    } catch (e) {
        console.error('[Arena] Metrics poll error:', e);
    } finally {
        _metricsInFlight = false;
    }
}

// ============================================
//  TABS
// ============================================

function showTab(name) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    const panel = document.getElementById('tab-' + name);
    const btn = document.querySelector(`.tab-btn[data-tab="${name}"]`);
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');

    if (name === 'training') {
        // Lazy-init so the canvas is visible at creation (Chart.js mis-sizes a hidden canvas).
        if (!trainingChart) initTrainingChart();
        fetchMetrics();                                 // immediate refresh on open
        if (!_metricsTimer) _metricsTimer = setInterval(fetchMetrics, 2000);
        if (trainingChart) trainingChart.resize();
    } else if (_metricsTimer) {
        clearInterval(_metricsTimer);                   // stop polling while the tab is hidden
        _metricsTimer = null;
    }

    if (name === 'live') {
        // Render straight from the last poll so the tab is populated instantly;
        // the 5s pollStatus/pollEvents loops keep it ticking.
        renderHallOfFame(appState.agents);
        renderNowTraining(appState.events);
    }
}

// ============================================
//  INIT
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    hydrateIcons();                 // swap every data-ico placeholder for its SVG
    initChart();
    addEvent('system', null, 'Connected to Arena server.');

    pollStatus();
    pollHistory();
    pollEvents();

    setInterval(pollStatus, 5000);
    setInterval(pollHistory, 5000);
    setInterval(pollEvents, 5000);

    console.log('[Arena] Dashboard initialized.');
});
