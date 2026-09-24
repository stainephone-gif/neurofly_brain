// NeuroFly gallery screen: 138k neurons as a living point cloud.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';

// colours per super_class (see meta.super_classes for the order)
const CLASS_COLORS = {
  optic: '#3a6fd8', central: '#a06cff', sensory: '#4fd18a', visual_projection: '#2fbfc7',
  visual_centrifugal: '#7fd6ff', descending: '#ff9f43', ascending: '#ffd166', sensory_ascending: '#bfe36b',
  motor: '#ff5a5a', endocrine: '#ff7ad9', '': '#6b6b6b',
};
const CLASS_RU = {
  optic: 'зрительная доля', central: 'центральный мозг', sensory: 'сенсорные', visual_projection: 'зрительная проекция',
  visual_centrifugal: 'центрифугальные', descending: 'нисходящие', ascending: 'восходящие', sensory_ascending: 'сенсорно-восходящие',
  motor: 'мотонейроны', endocrine: 'эндокринные', '': 'без аннотации',
};

const state = {
  meta: null, n: 0, pos: null, cls: null,
  act: null, flag: null, mode: 'look', scope: 'type',
  selected: null, pendingPre: null, status: null, ws: null, lastFrameWall: 0,
};

// ------------------------------------------------------------------ scene
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05060a);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 1, 20000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.autoRotate = true; controls.autoRotateSpeed = 0.35;
controls.addEventListener('start', () => { controls.autoRotate = false; idleSince = performance.now(); });
let idleSince = performance.now();

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

let points, pointMat, synLines, extraLines, skeletonLines, selectMarker;

function buildPoints() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(state.pos, 3));
  const color = new Float32Array(state.n * 3);
  const c = new THREE.Color();
  for (let i = 0; i < state.n; i++) {
    c.set(CLASS_COLORS[state.meta.super_classes[state.cls[i]]] || '#777');
    color[3 * i] = c.r; color[3 * i + 1] = c.g; color[3 * i + 2] = c.b;
  }
  g.setAttribute('base', new THREE.BufferAttribute(color, 3));
  state.act = new Float32Array(state.n);
  state.flag = new Float32Array(state.n);
  g.setAttribute('act', new THREE.BufferAttribute(state.act, 1).setUsage(THREE.DynamicDrawUsage));
  g.setAttribute('flag', new THREE.BufferAttribute(state.flag, 1).setUsage(THREE.DynamicDrawUsage));
  pointMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uScale: { value: innerHeight / 2 } },
    vertexShader: `
      attribute vec3 base; attribute float act; attribute float flag;
      varying vec3 vColor; varying float vAlpha;
      uniform float uScale;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float a = clamp(act, 0.0, 1.0);
        vec3 col = mix(base * 0.95, vec3(1.0, 0.95, 0.75), a);
        float size = 2.1 + a * 7.0;
        if (flag > 1.5) { col = mix(vec3(0.55, 0.12, 0.12), vec3(1.0, 0.3, 0.3), a); size = 2.2; }
        else if (flag > 0.5) { col = mix(vec3(0.2, 0.7, 1.0), vec3(0.8, 1.0, 1.0), a); size += 1.5; }
        vColor = col;
        vAlpha = 0.42 + a * 0.58 + step(0.5, flag) * 0.3;
        gl_PointSize = size * uScale / -mv.z * 2.2;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor; varying float vAlpha;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        float a = smoothstep(0.5, 0.1, d) * vAlpha;
        gl_FragColor = vec4(vColor, a);
      }`,
  });
  points = new THREE.Points(g, pointMat);
  scene.add(points);

  selectMarker = new THREE.Mesh(new THREE.SphereGeometry(4, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.9 }));
  selectMarker.visible = false; scene.add(selectMarker);
}

function frameCamera() {
  const [lo, hi] = state.meta.bounds;
  const center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
  controls.target.copy(center);
  camera.position.set(center.x, center.y - 60, center.z + 980);
  camera.up.set(0, -1, 0); // EM space: y grows downwards
  camera.lookAt(center);
}

async function loadMesh(name, opts) {
  try {
    const buf = await (await fetch(`/api/mesh/${name}`)).arrayBuffer();
    const [nv, nf] = new Uint32Array(buf, 0, 2);
    const v = new Float32Array(buf, 8, nv * 3);
    const f = new Uint32Array(buf, 8 + nv * 12, nf * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(v, 3));
    g.setIndex(new THREE.BufferAttribute(f, 1));
    g.computeVertexNormals();
    const surf = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: opts.color, transparent: true, opacity: opts.opacity, depthWrite: false, side: THREE.DoubleSide }));
    const wire = new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: opts.color, transparent: true, opacity: opts.wire }));
    scene.add(surf); scene.add(wire);
  } catch (e) { console.warn('mesh', name, e); }
}

// ------------------------------------------------------------ picking
const _v = new THREE.Vector3();
function pick(clientX, clientY) {
  const w = innerWidth, h = innerHeight;
  const nx = (clientX / w) * 2 - 1, ny = -(clientY / h) * 2 + 1;
  camera.updateMatrixWorld();
  const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = m.elements, p = state.pos;
  let best = -1, bestD = (14 / w * 2) ** 2, bestZ = Infinity;
  for (let i = 0; i < state.n; i++) {
    const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
    const cw = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (cw <= 0) continue;
    const cx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / cw;
    const cy = (e[1] * x + e[5] * y + e[9] * z + e[13]) / cw;
    const dx = cx - nx, dy = (cy - ny) * (h / w);
    const d = dx * dx + dy * dy;
    if (d < bestD * 1.0001 && (d < bestD * 0.5 || cw < bestZ)) { best = i; bestD = Math.max(d, bestD * 0.5); bestZ = cw; }
  }
  return best;
}

let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt) return;
  const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
  downAt = null;
  if (moved > 6 || !state.pos) return;
  const i = pick(e.clientX, e.clientY);
  if (i < 0) return;
  onNeuronClick(i);
});

function onNeuronClick(i) {
  const rate = +$('#rate-slider').value, w = +$('#w-slider').value, sign = $('#sign').classList.contains('on') ? 1 : -1;
  if (state.mode === 'activate') {
    send({ cmd: state.status && stimOf(i) ? 'deactivate' : 'activate', idx: [i], rate, expand: state.scope });
  } else if (state.mode === 'silence') {
    send({ cmd: state.status && state.silSet.has(i) ? 'unsilence' : 'silence', idx: [i], expand: state.scope });
  } else if (state.mode === 'connect') {
    if (state.pendingPre === null) { state.pendingPre = i; setStatus('выбран источник; щёлкните по цели'); select(i); return; }
    send({ cmd: 'connect', pre: state.pendingPre, post: i, n: w, sign });
    setStatus(`связь ${state.pendingPre} → ${i}: ${sign > 0 ? '+' : '−'}${w} синапсов`);
    state.pendingPre = null;
  }
  select(i);
}

function select(i) {
  state.selected = i;
  selectMarker.position.set(state.pos[3 * i], state.pos[3 * i + 1], state.pos[3 * i + 2]);
  selectMarker.visible = true;
  send({ cmd: 'info', idx: i });
}

// ------------------------------------------------------------- lines
function setLines(holder, segs, colors) {
  if (holder.obj) { scene.remove(holder.obj); holder.obj.geometry.dispose(); }
  if (!segs.length) { holder.obj = null; return; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(segs, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  holder.obj = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: holder.opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
  scene.add(holder.obj);
}
synLines = { obj: null, opacity: 0.55 }; extraLines = { obj: null, opacity: 0.95 }; skeletonLines = { obj: null, opacity: 0.9 };

function showSynapses(info) {
  const rows = info.out.slice(0, 150).map(([j, w]) => [info.idx, j, w]).concat(info.in.slice(0, 80).map(([j, w]) => [j, info.idx, w]));
  const segs = new Float32Array(rows.length * 6), col = new Float32Array(rows.length * 6);
  rows.forEach(([a, b, w], k) => {
    segs.set([state.pos[3 * a], state.pos[3 * a + 1], state.pos[3 * a + 2], state.pos[3 * b], state.pos[3 * b + 1], state.pos[3 * b + 2]], 6 * k);
    const c = w > 0 ? [0.4, 1.0, 0.6] : [1.0, 0.35, 0.35];
    col.set([...c, ...c], 6 * k);
  });
  setLines(synLines, segs, col);
}

function drawExtra(list) {
  const segs = new Float32Array(list.length * 6), col = new Float32Array(list.length * 6);
  list.forEach(([a, b, w], k) => {
    segs.set([state.pos[3 * a], state.pos[3 * a + 1], state.pos[3 * a + 2], state.pos[3 * b], state.pos[3 * b + 1], state.pos[3 * b + 2]], 6 * k);
    const c = w > 0 ? [1.0, 0.85, 0.3] : [1.0, 0.4, 0.7];
    col.set([...c, ...c], 6 * k);
  });
  setLines(extraLines, segs, col);
}

async function showSkeleton(i) {
  setStatus('загружаю форму нейрона…');
  try {
    const buf = await (await fetch(`/api/skeleton/${i}`)).arrayBuffer();
    const m = new Uint32Array(buf, 0, 1)[0];
    const segs = new Float32Array(buf, 4, m * 6);
    const col = new Float32Array(m * 6).fill(0.9);
    setLines(skeletonLines, segs, col);
    setStatus('');
  } catch (e) { setStatus('форма недоступна (нет сети или архива)'); }
}

// --------------------------------------------------------- websocket
function send(obj) { if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj)); }
function stimOf(i) { return state.stimMap && state.stimMap.get(i); }

function connect() {
  const ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) { onFrame(ev.data); return; }
    const msg = JSON.parse(ev.data);
    if (msg.type === 'status') onStatus(msg);
    else if (msg.type === 'info') onInfo(msg);
    else if (msg.type === 'error') setStatus('ошибка: ' + msg.message);
  };
  ws.onclose = () => { setStatus('нет связи с симуляцией, переподключаюсь…'); setTimeout(connect, 1500); };
  ws.onopen = () => setStatus('');
  state.ws = ws;
}

function onFrame(buf) {
  const head = new DataView(buf);
  const n = head.getUint32(8, true);
  const idx = new Uint32Array(buf, 12, n);
  const act = state.act;
  if (!act) return;
  for (let k = 0; k < n; k++) { const i = idx[k]; act[i] = Math.min(1.5, act[i] + 0.7); }
  state.lastFrameWall = performance.now();
  state.tModel = head.getFloat32(0, true);
}

function onStatus(s) {
  state.status = s;
  state.stimMap = new Map(s.stim);
  state.silSet = new Set(s.silenced);
  const flag = state.flag;
  if (flag) {
    flag.fill(0);
    for (const [i] of s.stim) flag[i] = 1;
    for (const i of s.silenced) flag[i] = 2;
    points.geometry.attributes.flag.needsUpdate = true;
  }
  drawExtra(s.extra);
  $('#n-stim').textContent = s.n_stim; $('#n-sil').textContent = s.n_silenced; $('#n-extra').textContent = s.n_extra;
  $('#rate').textContent = Math.round(s.spikes_per_s); $('#active').textContent = s.active;
  $('#speed').textContent = s.paused ? 'пауза' : `${s.realtime.toFixed(2)}× реального`;
  $('#pause').textContent = s.paused ? 'Пуск' : 'Пауза';
  for (const b of document.querySelectorAll('#presets button')) {
    const p = state.meta.presets[+b.dataset.k];
    b.classList.toggle('stim', p.rate > 0 && p.idx.some((i) => state.stimMap.has(i)));
  }
}

function onInfo(info) {
  state.info = info;
  $('#info').style.display = 'block';
  $('#info-label').textContent = info.cell_type || 'без типа';
  $('#info-id').textContent = `FlyWire ${info.id} · #${info.idx}`;
  const st = [];
  st.push(['класс', CLASS_RU[info.super_class] || info.super_class || '—']);
  st.push(['подкласс', info.cell_class || '—']);
  st.push(['сторона', info.side === 'left' ? 'левая' : info.side === 'right' ? 'правая' : info.side || '—']);
  st.push(['медиатор', info.nt || '—']);
  st.push(['входов', `${info.n_in} нейронов · ${info.syn_in} синапсов`]);
  st.push(['выходов', `${info.n_out} нейронов · ${info.syn_out} синапсов`]);
  st.push(['состояние', info.silenced ? 'выключен' : info.stim > 0 ? `возбуждён ${info.stim} Гц` : 'обычное']);
  $('#info-stats').innerHTML = st.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join('');
  $('#info-activate').textContent = info.stim > 0 ? 'Снять возбуждение' : 'Возбудить тип';
  $('#info-silence').textContent = info.silenced ? 'Включить тип' : 'Выключить тип';
}

// ----------------------------------------------------------------- UI
function setStatus(t) { $('#status').textContent = t; }
for (const b of document.querySelectorAll('[data-mode]')) b.onclick = () => {
  state.mode = b.dataset.mode; state.pendingPre = null;
  for (const o of document.querySelectorAll('[data-mode]')) o.classList.toggle('on', o === b);
  $('#hint').textContent = { look: 'щёлкните по нейрону, чтобы узнать, кто он', activate: 'щёлкните по нейрону: все клетки его типа получат стимул', silence: 'щёлкните по нейрону: все клетки его типа замолчат', connect: 'щёлкните по источнику, затем по цели: появится новая связь' }[state.mode];
};
$('#rate-slider').oninput = (e) => $('#rate-out').textContent = `${e.target.value} Гц`;
$('#w-slider').oninput = (e) => $('#w-out').textContent = e.target.value;
$('#sign').onclick = () => { const b = $('#sign'); const on = !b.classList.contains('on'); b.classList.toggle('on', on); b.textContent = on ? '+ возбуждающая' : '− тормозная'; };
$('#pause').onclick = () => send({ cmd: state.status && state.status.paused ? 'play' : 'pause' });
$('#reset').onclick = () => send({ cmd: 'reset' });
$('#clear').onclick = () => { send({ cmd: 'clear' }); setLines(synLines, new Float32Array(0), new Float32Array(0)); setLines(skeletonLines, new Float32Array(0), new Float32Array(0)); };
$('#toggle-scope').onclick = () => { state.scope = state.scope === 'type' ? 'type_side' : 'type'; $('#toggle-scope').textContent = state.scope === 'type' ? 'Тип клеток: обе стороны' : 'Тип клеток: одна сторона'; };
$('#info-close').onclick = () => { $('#info').style.display = 'none'; selectMarker.visible = false; };
$('#info-activate').onclick = () => { const i = state.info; send({ cmd: i.stim > 0 ? 'deactivate' : 'activate', idx: [i.idx], rate: +$('#rate-slider').value, expand: state.scope }); setTimeout(() => send({ cmd: 'info', idx: i.idx }), 300); };
$('#info-silence').onclick = () => { const i = state.info; send({ cmd: i.silenced ? 'unsilence' : 'silence', idx: [i.idx], expand: state.scope }); setTimeout(() => send({ cmd: 'info', idx: i.idx }), 300); };
$('#info-synapses').onclick = () => showSynapses(state.info);
$('#info-skeleton').onclick = () => showSkeleton(state.info.idx);

function buildPresets() {
  const box = $('#presets');
  state.meta.presets.forEach((p, k) => {
    const b = document.createElement('button');
    b.textContent = p.label; b.title = p.hint; b.dataset.k = k;
    b.onclick = () => {
      if (p.rate > 0) send({ cmd: b.classList.contains('stim') ? 'deactivate' : 'activate', idx: p.idx, rate: p.rate });
      select(p.idx[0]);
    };
    box.appendChild(b);
  });
  const lg = $('#legend');
  lg.innerHTML = state.meta.super_classes.map((c) => `<i style="background:${CLASS_COLORS[c] || '#777'}"></i><span>${CLASS_RU[c] || c}</span>`).join('');
}

// ---------------------------------------------------------------- main
async function main() {
  setStatus('загружаю нейроны…');
  state.meta = await (await fetch('/api/meta.json')).json();
  const buf = await (await fetch('/api/neurons.bin')).arrayBuffer();
  state.n = state.meta.n;
  state.pos = new Float32Array(buf, 0, state.n * 3);
  state.cls = new Uint8Array(buf, state.n * 12, state.n);
  $('#n-neurons').textContent = state.n.toLocaleString('ru');
  $('#n-syn').textContent = state.meta.n_synapses.toLocaleString('ru');
  buildPoints(); frameCamera(); buildPresets();
  loadMesh('volume', { color: 0x8fa3c7, opacity: 0.035, wire: 0.05 });
  connect();
  setStatus('');
  let last = performance.now();
  renderer.setAnimationLoop(() => {
    const now = performance.now(), dt = (now - last) / 1000; last = now;
    const decay = Math.pow(0.03, dt / 0.9);
    const act = state.act;
    for (let i = 0; i < act.length; i++) if (act[i] > 0.002) act[i] *= decay; else act[i] = 0;
    points.geometry.attributes.act.needsUpdate = true;
    if (state.tModel !== undefined) $('#t').textContent = `${(state.tModel / 1000).toFixed(2)} с`;
    if (!controls.autoRotate && now - idleSince > 45000) controls.autoRotate = true;
    pointMat.uniforms.uScale.value = innerHeight / 2;
    controls.update();
    renderer.render(scene, camera);
  });
}
main().catch((e) => setStatus('ошибка: ' + e.message));
