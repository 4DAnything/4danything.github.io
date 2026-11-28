// Embeddable 4D Gaussian-splat viewer: one or more .4dgs objects (v2 records) + optional scene point cloud.
// Adapted from docs/viewer4d (same shaders / formats); one Viewer4D per <canvas>.
"use strict";
(function () {
const MAGIC = 0x34444753, SCENE_MAGIC = 0x43534756, REC = 32;

const VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 corner;
layout(location=1) in vec3 aPos;
layout(location=2) in vec3 aScale;
layout(location=3) in vec4 aColor;
layout(location=4) in vec4 aRot;
layout(location=5) in vec3 bPos;
layout(location=6) in vec3 bScale;
layout(location=7) in vec4 bColor;
layout(location=8) in vec4 bRot;
uniform mat4 uView, uProj;
uniform vec2 uFocal, uViewport;
uniform float uMix;
out vec4 vColor; out vec2 vPos;
vec4 interpolateRotation(vec4 a, vec4 b, float t) {
  float d = dot(a, b); if (d < 0.0) { b = -b; d = -d; }
  if (d > 0.9995) return normalize(mix(a, b, t));
  float angle = acos(clamp(d, 0.0, 1.0));
  return (sin((1.0-t)*angle)*a + sin(t*angle)*b) / sin(angle);
}
void main() {
  vec4 cam = uView * vec4(mix(aPos, bPos, uMix), 1.0);
  if (cam.z < 0.05) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 q = normalize(aRot * (255.0/128.0) - 1.0);
  if (uMix > 0.0) q = interpolateRotation(q, normalize(bRot * (255.0/128.0) - 1.0), uMix);
  float w=q.x, x=q.y, y=q.z, z=q.w;
  mat3 R = mat3(1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y),
                2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x),
                2.*(x*z+w*y), 2.*(y*z-w*x), 1.-2.*(x*x+y*y));
  vec3 scale = uMix > 0.0 ? exp(mix(log(aScale), log(bScale), uMix)) : aScale;
  mat3 S = mat3(scale.x,0.,0., 0.,scale.y,0., 0.,0.,scale.z);
  mat3 M = S * R; mat3 cov3 = transpose(M) * M; mat3 W = mat3(uView);
  float iz = 1.0 / cam.z;
  mat3 J = mat3(uFocal.x*iz, 0., 0., 0., uFocal.y*iz, 0., -uFocal.x*cam.x*iz*iz, -uFocal.y*cam.y*iz*iz, 0.);
  mat3 T = W * cov3 * transpose(W); mat3 cov2 = J * T * transpose(J);
  float a = cov2[0][0]+0.3, b = cov2[0][1], d = cov2[1][1]+0.3;
  float mid = 0.5*(a+d), rad = length(vec2(0.5*(a-d), b));
  float l1 = mid + rad, l2 = max(mid - rad, 0.02);
  vec2 e1 = normalize(vec2(b, l1 - a)); if (b == 0.0) e1 = (a >= d) ? vec2(1,0) : vec2(0,1);
  vec2 major = min(sqrt(2.0*l1), 1024.0) * e1, minor = min(sqrt(2.0*l2), 1024.0) * vec2(-e1.y, e1.x);
  vec4 clip = uProj * cam;
  vec2 off = (corner.x*major + corner.y*minor) / uViewport * 2.0 * clip.w;
  gl_Position = vec4(clip.xy + off, clip.z, clip.w); vColor = mix(aColor, bColor, uMix); vPos = corner;
}`;
const FS = `#version 300 es
precision highp float;
in vec4 vColor; in vec2 vPos; out vec4 frag;
void main() { float A = -dot(vPos, vPos); if (A < -4.0) discard; frag = vec4(vColor.rgb, exp(A) * vColor.a); }`;
const SVS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aP; layout(location=1) in vec3 aC;
uniform mat4 uView, uProj; uniform vec3 uMin, uExt; uniform float uPt; out vec3 vC;
void main() { gl_Position = uProj * (uView * vec4(uMin + aP * uExt, 1.0)); gl_PointSize = uPt; vC = aC; }`;
const LVS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aP; uniform mat4 uView, uProj;
void main() { gl_Position = uProj * (uView * vec4(aP, 1.0)); }`;
const LFS = `#version 300 es
precision highp float;
uniform vec4 uCol; out vec4 frag; void main() { frag = uCol; }`;
const QVS = `#version 300 es
precision highp float;
layout(location=0) in vec4 aP; void main() { gl_Position = aP; }`;
const SFS = `#version 300 es
precision highp float;
in vec3 vC; out vec4 frag; void main() { frag = vec4(vC, 1.0); }`;

const TVS = `#version 300 es
precision highp float;
layout(location=0) in vec4 aP; layout(location=1) in vec4 aC;
uniform float uPointSize; out vec4 vC;
void main() { vC = aC; gl_Position = aP; gl_PointSize = uPointSize; }`;
const TFS = `#version 300 es
precision highp float;
in vec4 vC; uniform bool uRound; out vec4 frag;
void main() {
  float coverage = 1.0;
  if (uRound) { vec2 p = gl_PointCoord * 2.0 - 1.0; coverage = 1.0 - smoothstep(0.7, 1.0, dot(p, p)); }
  frag = vec4(vC.rgb, vC.a * coverage);
}`;
const GZ_MAGIC = 0x5A443456;

function heightColor(height) {       // teaser palette: blue at low initial height, red at high initial height
  const h = (1 - height) * 0.667;
  return [0, 2/3, 1/3].map(offset => Math.max(0, Math.min(1, Math.abs(((h + offset) % 1) * 6 - 3) - 1)));
}
function mm3(A, B) { const C = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i*3+j] = A[i*3]*B[j] + A[i*3+1]*B[3+j] + A[i*3+2]*B[6+j]; return C; }
function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [1,0,0, 0,c,-s, 0,s,c]; }
function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [c,0,s, 0,1,0, -s,0,c]; }

// An object is {F, fps, tracked, nAt(f), posAt(f), rotAt(f), sclAt(f), sclMul(f), rgbaAt(f), tracks}: per-frame typed arrays,
// built on demand; "tracked" = splat i is the same Gaussian in every frame (so one depth order serves nearby frames).
function lru(keep, build) {
  const c = new Map();
  return f => { let v = c.get(f); if (!v) { v = build(f); c.set(f, v); if (c.size > keep) c.delete(c.keys().next().value); } return v; };
}

function tracksOf(F, idx, posAt) {
  const K = idx.length, tr = new Float32Array(F * K * 3);
  for (let f = 0; f < F; f++) { const P = posAt(f); for (let j = 0; j < K; j++) { const i = idx[j]; tr.set(P.subarray(i * 3, i * 3 + 3), (f * K + j) * 3); } }
  return { F, K, pos: tr };
}

function expScl(n, get, l0, ls) {     // log-quantized u8 scales -> linear
  const out = new Float32Array(n * 3); for (let i = 0; i < n * 3; i++) out[i] = Math.exp(l0 + get(i) * ls); return out;
}

async function parse4dgz(ab) {        // gzip; static sca/rgba (planar), 2^bits positions + rotations delta-coded in time
  const u = new Uint8Array(await new Response(new Blob([ab]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()), b = u.buffer;
  const dv = new DataView(b), F = dv.getUint32(4, true), fps = dv.getFloat32(8, true), n = dv.getUint32(12, true), M = 2 ** dv.getUint32(16, true) - 1;
  const bb = new Float32Array(b.slice(20, 44)), rg = new Float32Array(b.slice(44, 44 + 8 * F)); let o = 44 + 8 * F;
  const sp = u.subarray(o, o + 3 * n); o += 3 * n; const cp = u.subarray(o, o + 4 * n); o += 4 * n;
  const scl = expScl(n, i => sp[(i % 3) * n + ((i / 3) | 0)], rg[0], (rg[1] - rg[0]) / 255);
  const rgba = new Uint8Array(n * 4); for (let i = 0; i < n; i++) for (let k = 0; k < 4; k++) rgba[i*4+k] = cp[k*n+i];
  const q = new Uint16Array(F * n * 3), rot = new Uint8Array(F * n * 4);
  for (let k = 0; k < 3; k++) for (let i = 0; i < n; i++) q[i*3+k] = u[o + k*n + i] | (u[o + (3+k)*n + i] << 8);
  o += 6 * n; for (let k = 0; k < 4; k++) for (let i = 0; i < n; i++) rot[i*4+k] = u[o + k*n + i]; o += 4 * n;
  const P = (F - 1) * n;
  for (let f = 1; f < F; f++) for (let k = 0; k < 3; k++) { const lo = o + k*P + (f-1)*n, hi = o + (3+k)*P + (f-1)*n, d0 = (f-1)*n*3 + k, d1 = f*n*3 + k;
    for (let i = 0, a = 0; i < n; i++, a += 3) q[d1+a] = q[d0+a] + ((u[lo+i] | (u[hi+i] << 8)) << 16 >> 16); }
  o += 6 * P; const r8 = new Int8Array(b, o, 4 * P);
  for (let f = 1; f < F; f++) for (let k = 0; k < 4; k++) { const s0 = (f-1)*4*n + k*n, d0 = (f-1)*n*4 + k, d1 = f*n*4 + k;
    for (let i = 0, a = 0; i < n; i++, a += 4) rot[d1+a] = rot[d0+a] + r8[s0+i]; }
  const ex = [0, 1, 2].map(k => (bb[3+k] - bb[k]) / M);
  const posAt = lru(3, f => { const out = new Float32Array(n * 3), base = f * n * 3;
    for (let i = 0; i < n * 3; i += 3) { out[i] = bb[0] + q[base+i] * ex[0]; out[i+1] = bb[1] + q[base+i+1] * ex[1]; out[i+2] = bb[2] + q[base+i+2] * ex[2]; }
    return out; });
  const obj = { F, fps, tracked: true, nAt: () => n, posAt, rotAt: f => rot.subarray(f * n * 4, (f + 1) * n * 4),
                sclAt: () => scl, sclMul: f => Math.exp(rg[2*f] - rg[0]), rgbaAt: () => rgba };
  return obj;
}

function parse4dgs(ab) {             // v2 (per-frame 18 B records) or v3 (static sca/rgba, per-frame pos + rot)
  const dv = new DataView(ab), u8 = new Uint8Array(ab);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error("bad 4dgs");
  const F = dv.getUint32(4, true), fps = dv.getFloat32(8, true) || 8;
  if (dv.getUint32(12, true) === 3) {
    const n = dv.getUint32(16, true), H = new Float32Array(ab.slice(20, 20 + 32 * F)), s0 = 20 + 32 * F, c0 = s0 + 3 * n, f0 = c0 + 4 * n;
    const scl = expScl(n, i => u8[s0+i], H[6], (H[7] - H[6]) / 255), rgba = u8.slice(c0, c0 + 4 * n);
    const posAt = lru(3, f => { const b = f0 + f * 10 * n, h = H.subarray(f*8, f*8+8), out = new Float32Array(n * 3), e = [0, 1, 2].map(k => (h[3+k]-h[k]) / 65535);
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) out[i*3+k] = h[k] + (u8[b + i*6 + 2*k] | (u8[b + i*6 + 2*k + 1] << 8)) * e[k];
      return out; });
    const obj = { F, fps, tracked: true, nAt: () => n, posAt, rotAt: f => u8.subarray(f0 + f * 10 * n + 6 * n, f0 + (f + 1) * 10 * n),
                  sclAt: () => scl, sclMul: f => Math.exp(H[f*8+6] - H[6]), rgbaAt: () => rgba };
    return obj;
  }
  const counts = new Uint32Array(ab, 16, F), H = new Float32Array(ab.slice(16 + 4*F, 16 + 36*F)), offs = [];
  let off = 16 + 36*F; for (let f = 0; f < F; f++) { offs.push(off); off += counts[f] * 18; }
  const field = (f, w, at) => { const n = counts[f], b = offs[f], out = new Uint8Array(n * w); for (let i = 0; i < n; i++) for (let k = 0; k < w; k++) out[i*w+k] = u8[b + i*18 + at + k]; return out; };
  const posAt = lru(3, f => { const n = counts[f], b = offs[f], h = H.subarray(f*8, f*8+8), out = new Float32Array(n * 3), e = [0, 1, 2].map(k => (h[3+k]-h[k]) / 65535);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) out[i*3+k] = h[k] + (u8[b + i*18 + 2*k] | (u8[b + i*18 + 2*k + 1] << 8)) * e[k];
    return out; });
  const sclAt = lru(3, f => { const s = field(f, 3, 6); return expScl(counts[f], i => s[i], H[f*8+6], (H[f*8+7] - H[f*8+6]) / 255); });
  const obj = { F, fps, tracked: [...counts].every(c => c === counts[0]), nAt: f => counts[f], posAt, sclAt, sclMul: () => 1,
                rgbaAt: lru(3, f => field(f, 4, 9)), rotAt: lru(3, f => field(f, 4, 13)) };
  return obj;
}

function loadJs(src) {               // <script> loading works from file:// where fetch() does not
  return new Promise((ok, no) => { const s = document.createElement("script"); s.src = src; s.onload = ok;
    s.onerror = () => no(new Error("missing " + src)); document.head.appendChild(s); });
}
function b64buf(t) { const bin = atob(t), out = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i); return out.buffer; }

async function getBin(u) {           // raw binary, base64 text (*.b64.txt), or packed <u>.js when window.V4D_PACKED
  if (window.V4D_PACKED) {
    const k = u.split("/").pop(), B = window.V4D_BLOBS = window.V4D_BLOBS || {};
    if (!(k in B)) await loadJs(u + ".js");
    return b64buf(B[k]);
  }
  const r = await fetch(u); if (!r.ok) throw new Error(r.status + " " + u);
  if (!u.endsWith(".b64.txt")) return r.arrayBuffer();
  const t = await r.text(), bin = atob(t.trim()), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

function mergeObjects(objs) {        // several objects as one stream (per-frame concatenation, scale multipliers baked in)
  const F = Math.min(...objs.map(o => o.F)), nAt = f => objs.reduce((s, o) => s + o.nAt(f), 0);
  const cat = (get, w, T) => lru(3, f => { const out = new T(nAt(f) * w); let k = 0; for (const o of objs) { const a = get(o, f); out.set(a, k); k += a.length; } return out; });
  const obj = { F, fps: objs[0].fps, tracked: objs.every(o => o.tracked), nAt, sclMul: () => 1,
    posAt: cat((o, f) => o.posAt(f), 3, Float32Array), rotAt: cat((o, f) => o.rotAt(f), 4, Uint8Array), rgbaAt: cat((o, f) => o.rgbaAt(f), 4, Uint8Array),
    sclAt: cat((o, f) => { const s = o.sclAt(f), m = o.sclMul(f); if (m === 1) return s; const t = new Float32Array(s.length); for (let i = 0; i < s.length; i++) t[i] = s[i] * m; return t; }, 3, Float32Array) };
  const tk = objs.filter(o => o.tracks), K = tk.reduce((s, o) => s + o.tracks.K, 0), tr = new Float32Array(F * K * 3);
  for (let f = 0; f < F; f++) { let j = 0; for (const o of tk) { tr.set(o.tracks.pos.subarray(f * o.tracks.K * 3, (f + 1) * o.tracks.K * 3), (f * K + j) * 3); j += o.tracks.K; } }
  const colors = new Float32Array(K * 3); let colorOffset = 0;
  for (const o of tk) { colors.set(o.tracks.colors, colorOffset); colorOffset += o.tracks.K * 3; }
  obj.tracks = K ? { F, K, pos: tr, colors } : null; return obj;
}

async function parseAny(ab, u) {
  if (/\.4dgz(\.js)?$/.test(u)) return parse4dgz(ab);
  if (/\.4dgs(?:\.delta)?\.gz$/.test(u)) ab = await new Response(new Blob([ab]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  if (/\.4dgs\.delta\.gz$/.test(u)) {
    const d = new DataView(ab), F = d.getUint32(4, true), n = d.getUint32(16, true), h = 16 + 36 * F;
    const planes = new Uint8Array(ab), records = new Uint8Array(ab.byteLength); records.set(planes.subarray(0, h));
    for (let f = 0; f < F; f++) for (let k = 0; k < 18; k++) for (let i = 0; i < n; i++) {
      const dst = h + (f*n+i)*18+k;
      records[dst] = planes[h+(f*18+k)*n+i] + (f ? records[dst-18*n] : 0);
    }
    ab = records.buffer;
  }
  return parse4dgs(ab);
}

function side(obj) { return { obj, ord: null, ordEpoch: -1, ordFrame: -1, stamp: 0, buf: null, bufKey: "" }; }

class Viewer4D {
  constructor(root, opts = {}) {
    this.root = root; this.opts = opts;
    this.canvas = root.querySelector("canvas");
    this.ui = { play: root.querySelector("[data-v=play]"), frame: root.querySelector("[data-v=frame]"),
                fno: root.querySelector("[data-v=fno]"), scene: root.querySelector("[data-v=scene]"),
                reset: root.querySelector("[data-v=reset]"), camchk: root.querySelector("[data-v=cam]"), trk: root.querySelector("[data-v=tracks]"), load: root.querySelector("[data-v=load]"),
                inset: root.querySelector("video[data-v=inset]"), continuous: root.querySelector("[data-v=continuous]") };
    const gl = this.gl = this.canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) { this.ui.load.textContent = "WebGL2 is not available in this browser."; return; }
    const mk = (vs, fs) => { const p = gl.createProgram();
      for (const [t, s] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) { const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh); gl.attachShader(p, sh); }
      gl.linkProgram(p); return p; };
    this.prog = mk(VS, FS); this.sprog = mk(SVS, SFS); this.lprog = mk(LVS, LFS); this.qprog = mk(QVS, LFS); this.lBuf = gl.createBuffer(); this.cam = null; this.showCam = true;
    this.tprog = mk(TVS, TFS); this.tBuf = gl.createBuffer(); this.tracks = null; this.showTracks = false;
    this.quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2,-2, 2,-2, -2,2, 2,2]), gl.STATIC_DRAW);
    this.sPos = gl.createBuffer(); this.sCol = gl.createBuffer();
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    this.A = null; this.nF = 0; this.epoch = 0; this.cur = 0; this.playing = true; this.fps = 8;
    this.playhead = 0; this.continuous = opts.continuous ?? !!(this.ui.continuous && this.ui.continuous.checked);
    this.sceneN = 0; this.showScene = true; this.visible = true; this.token = 0;
    this.bg = opts.bg || [1, 1, 1]; this.speed = opts.speed || 1.0;
    this.bindControls(); this.loop();
    new IntersectionObserver(es => { this.visible = es[0].isIntersecting; }, { rootMargin: "200px" }).observe(root);
  }

  async load(def) {
    const tok = ++this.token;
    this.ui.load.hidden = false; this.ui.load.textContent = "Loading " + (def.label || "scene") + "…";
    try {
      const bufs = await Promise.all(def.objects.map(getBin));
      const sceneAb = def.scene ? await getBin(def.scene).catch(() => null) : null;
      if (tok !== this.token) return;
      const objs = await Promise.all(bufs.map((b, i) => parseAny(b, def.objects[i])));
      if (tok !== this.token) return;
      const C = def.cams && def.cams.c2w && def.cams.c2w[0];
      this.trackUp = def.cam_up || (C ? [-C[1], -C[5], -C[9]] : [0, -1, 0]);
      this.buildTracks(objs);
      const m = objs.length > 1 ? mergeObjects(objs) : objs[0];
      if (this.A) for (const buffer of [this.A.glBuf, this.A.glBufNext]) if (buffer) this.gl.deleteBuffer(buffer);
      this.A = side(m); this.nF = m.F; this.fps = m.fps; this.tracks = m.tracks;
      this.sceneN = 0; if (sceneAb) this.loadScene(sceneAb);
      this.fitView(def.view); this.buildCams(def.cams); this.cur = 0; this.playhead = 0; this.epoch++;
      this.ui.frame.max = this.nF - 1;
      this.setContinuous(this.continuous);
      const v = this.ui.inset;
      if (v) { if (def.input) { v.src = def.input; v.hidden = false; if (this.playing) v.play().catch(() => {}); } else { v.removeAttribute("src"); v.hidden = true; } }
      this.ui.load.hidden = true;
    } catch (e) { if (tok === this.token) this.ui.load.textContent = "Could not load scene (" + e.message + ")."; }
  }

  buildCams(c) {                      // per-frame input-camera frustum lines + camera path, in the export frame
    this.cam = null; if (!c || !c.c2w || !c.c2w.length) return;
    const C0 = [c.c2w[0][3], c.c2w[0][7], c.c2w[0][11]], s = 0.08 * Math.hypot(this.target[0]-C0[0], this.target[1]-C0[1], this.target[2]-C0[2]);
    const uv = [[0, 0], [c.W, 0], [c.W, c.H], [0, c.H]], fr = [], path = new Float32Array(c.c2w.length * 3);
    c.c2w.forEach((M, f) => {
      const o = [M[3], M[7], M[11]], w = p => [0, 1, 2].map(i => M[i*4]*p[0] + M[i*4+1]*p[1] + M[i*4+2]*p[2] + o[i]);
      const k = uv.map(([u, v]) => w([(u - c.W/2) / c.fx * s, (v - c.H/2) / c.fy * s, s])), L = [];
      for (let i = 0; i < 4; i++) L.push(...o, ...k[i], ...k[i], ...k[(i+1) % 4]);   // 4 rays + image-plane rectangle
      fr.push(new Float32Array(L));
      path.set(o, f * 3);
    });
    this.cam = { fr, path };
  }

  thickLines(pts, M, W, H, px) {      // segment list (xyz pairs) -> clip-space quads px wide
    const out = [], cl = p => [0, 1, 2, 3].map(r => M[r]*p[0] + M[4+r]*p[1] + M[8+r]*p[2] + M[12+r]);
    for (let i = 0; i + 5 < pts.length; i += 6) {
      const a = cl([pts[i], pts[i+1], pts[i+2]]), b = cl([pts[i+3], pts[i+4], pts[i+5]]);
      if (a[3] < 0.02 || b[3] < 0.02) continue;
      let dx = (b[0]/b[3] - a[0]/a[3]) * W, dy = (b[1]/b[3] - a[1]/a[3]) * H; const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
      const nx = -dy * px / W, ny = dx * px / H, q = (c, s) => [c[0] + s*nx*c[3], c[1] + s*ny*c[3], c[2], c[3]];
      out.push(...q(a, 1), ...q(a, -1), ...q(b, 1), ...q(b, 1), ...q(a, -1), ...q(b, -1));
    }
    return new Float32Array(out);
  }

  drawCams(view, proj) {
    const gl = this.gl, p = this.qprog, U = n => gl.getUniformLocation(p, n), c = this.cam, W = gl.drawingBufferWidth, H = gl.drawingBufferHeight;
    const t = Math.min(this.cur, c.fr.length - 1), f = Math.floor(t), u = t - f;
    const fr = u ? c.fr[f].map((v, k) => v + u * (c.fr[f + 1][k] - v)) : c.fr[f], M = new Float32Array(16);
    for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) { let v = 0; for (let m = 0; m < 4; m++) v += proj[m*4+r] * view[k*4+m]; M[k*4+r] = v; }
    const path = []; for (let i = 0; i + 5 < c.path.length; i += 3) path.push(...c.path.subarray(i, i + 6));
    gl.useProgram(p); gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
    for (let i = 0; i < 9; i++) { gl.disableVertexAttribArray(i); gl.vertexAttribDivisor(i, 0); }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    for (const [seg, px, col] of [[path, 1.2, [0.2, 0.2, 0.2, 0.55]], [fr, 2.2, [0.255, 0.514, 0.769, 1.0]]]) {
      const q = this.thickLines(seg, M, W, H, px * devicePixelRatio); if (!q.length) continue;
      gl.bufferData(gl.ARRAY_BUFFER, q, gl.DYNAMIC_DRAW); gl.uniform4f(U("uCol"), ...col); gl.drawArrays(gl.TRIANGLES, 0, q.length / 4);
    }
    gl.disableVertexAttribArray(0);
  }

  buildTracks(objects) {             // same 2,000-point pool and stride 12 as the teaser, shared across objects
    const up = this.trackUp || [0, -1, 0], length = Math.hypot(...up) || 1;
    const perObject = Math.max(1, Math.floor(2000 / objects.length)), parts = [];
    let offset = 0, lo = Infinity, hi = -Infinity;
    for (const o of objects) {
      if (!o.tracked) { o.tracks = null; continue; }
      const n = o.nAt(0), count = Math.min(perObject, n), ids = [];
      for (let j = 0; j < count; j++) if ((j + offset) % 12 === 0) ids.push(Math.floor(j * n / count));
      offset += count;
      const tr = o.tracks = tracksOf(o.F, ids, o.posAt), heights = new Float32Array(tr.K);
      for (let j = 0; j < tr.K; j++) {
        const p = j * 3, h = (tr.pos[p]*up[0] + tr.pos[p+1]*up[1] + tr.pos[p+2]*up[2]) / length;
        heights[j] = h; lo = Math.min(lo, h); hi = Math.max(hi, h);
      }
      parts.push([tr, heights]);
    }
    for (const [tr, heights] of parts) {
      tr.colors = new Float32Array(tr.K * 3);
      for (let j = 0; j < tr.K; j++) tr.colors.set(heightColor((heights[j] - lo) / Math.max(hi - lo, 1e-8)), j * 3);
    }
  }

  drawTracks(V, tracks, cur) {        // Time-faded trajectories with current endpoints.
    if (!tracks || !tracks.K) return;
    const gl = this.gl, p = this.tprog, U = n => gl.getUniformLocation(p, n), K = tracks.K;
    cur = Math.max(0, Math.min(cur, tracks.F - 1));
    const start = Math.max(0, cur - 12), times = [start], matrix = new Float32Array(16), lines = [], tips = [];
    for (let t = (Math.floor(start * 4) + 1) / 4; t < cur; t += .25) times.push(t);
    if (cur > start) times.push(cur);
    for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) {
      let value = 0; for (let m = 0; m < 4; m++) value += V.proj[m*4+r] * V.view[k*4+m];
      matrix[k*4+r] = value;
    }
    const clips = times.map((time, step) => {
      const points = new Float32Array(K * 4), f = Math.floor(time), next = Math.min(f + 1, tracks.F - 1), u = time - f;
      for (let j = 0; j < K; j++) {
        const pos = [];
        for (let k = 0; k < 3; k++) {
          const at = frame => tracks.pos[(frame*K+j)*3+k], a = at(f), b = at(next);
          // Historical curves use the teaser's cubic interpolation. The tip follows the displayed GS exactly.
          pos[k] = u && step < times.length - 1
            ? (2*u**3-3*u**2+1)*a + (u**3-2*u**2+u)*(b-at(Math.max(0, f-1)))*.5
              + (-2*u**3+3*u**2)*b + (u**3-u**2)*(at(Math.min(tracks.F-1, next+1))-a)*.5
            : a + (b-a)*u;
        }
        for (let k = 0; k < 4; k++) points[j*4+k] = matrix[k]*pos[0] + matrix[4+k]*pos[1] + matrix[8+k]*pos[2] + matrix[12+k];
      }
      return points;
    });
    const dpr = typeof devicePixelRatio === 'number' ? devicePixelRatio : 1;
    const vertex = (point, j, time) => [...point, ...tracks.colors.subarray(j*3, j*3+3), Math.max(0, 1-(cur-time)/12)];
    for (let step = 0; step + 1 < times.length; step++) for (let j = 0; j < K; j++) {
      const a = clips[step].subarray(j*4, j*4+4), b = clips[step+1].subarray(j*4, j*4+4);
      if (a[3] < .02 || b[3] < .02) continue;
      const dx = (b[0]/b[3]-a[0]/a[3])*V.w, dy = (b[1]/b[3]-a[1]/a[3])*V.h, length = Math.hypot(dx, dy);
      if (length < 1e-5) continue;
      const nx = -dy/length*.45*dpr/V.w, ny = dx/length*.45*dpr/V.h;
      const corner = (point, sign, time) => vertex([point[0]+sign*nx*point[3], point[1]+sign*ny*point[3], point[2], point[3]], j, time);
      const a0 = corner(a, 1, times[step]), a1 = corner(a, -1, times[step]);
      const b0 = corner(b, 1, times[step+1]), b1 = corner(b, -1, times[step+1]);
      lines.push(...a0, ...a1, ...b0, ...b0, ...a1, ...b1);
    }
    for (let j = 0; j < K; j++) {
      const point = clips[clips.length-1].subarray(j*4, j*4+4);
      if (point[3] >= .02) tips.push(...vertex(point, j, cur));
    }
    for (let i = 0; i < 9; i++) { gl.disableVertexAttribArray(i); gl.vertexAttribDivisor(i, 0); }
    gl.useProgram(p); gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 16);
    gl.uniform1f(U("uPointSize"), 3*dpr);
    for (const [data, round, mode] of [[lines, 0, gl.TRIANGLES], [tips, 1, gl.POINTS]]) if (data.length) {
      gl.uniform1i(U("uRound"), round); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
      gl.drawArrays(mode, 0, data.length / 8);
    }
    gl.disableVertexAttribArray(0); gl.disableVertexAttribArray(1);
  }

  loadScene(ab) {
    const gl = this.gl, dv = new DataView(ab);
    if (dv.getUint32(0, true) !== SCENE_MAGIC) return;
    this.sceneN = dv.getUint32(4, true); this.sceneBox = new Float32Array(ab.slice(8, 32));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sPos); gl.bufferData(gl.ARRAY_BUFFER, new Uint16Array(ab, 32, this.sceneN * 3), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sCol); gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(ab, 32 + this.sceneN * 6, this.sceneN * 3), gl.STATIC_DRAW);
  }

  fitView(view) {                     // orbit centre = object centroid over (sampled) frames
    const o = this.A.obj, st = Math.max(1, Math.ceil(o.F / 6)), FR = [];
    for (let f = 0; f < o.F; f += st) FR.push({ pos: o.posAt(f), n: o.nAt(f) });
    let c = [0, 0, 0], n = 0;
    for (const fr of FR) for (let i = 0, p = fr.pos; i < fr.n; i += 7) { c[0] += p[i*3]; c[1] += p[i*3+1]; c[2] += p[i*3+2]; n++; }
    c = c.map(v => v / Math.max(n, 1)); let ext = 0;
    for (const fr of FR) for (let i = 0, p = fr.pos; i < fr.n; i += 7) { const dx = p[i*3]-c[0], dy = p[i*3+1]-c[1], dz = p[i*3+2]-c[2]; ext = Math.max(ext, dx*dx+dy*dy+dz*dz); }
    this.target = c;
    if (view && view.legacy) {
      this.radius0 = Math.sqrt(ext) * (view.zoom || 1.6) + 0.2; this.V0 = mm3(rotX(view.pitch || 0.12), rotY(view.yaw || -0.35));
    } else {                          // start at the input camera (origin, OpenCV axes) looking at the centroid, then turn slightly
      const L = Math.hypot(c[0], c[1], c[2]) || 1, f = c.map(v => v / L);
      let r = [f[2], 0, -f[0]]; const rl = Math.hypot(r[0], r[2]) || 1; r = r.map(v => v / rl);   // down (0,1,0) x f
      const d = [f[1]*r[2] - f[2]*r[1], f[2]*r[0] - f[0]*r[2], f[0]*r[1] - f[1]*r[0]];               // f x r
      this.V0 = mm3(mm3(rotX((view && view.pitch) || 0.38), rotY((view && view.yaw) || -0.62)), [...r, ...d, ...f]);
      let objR = 0;                   // mean per-frame half-diagonal of the objects' box
      for (const fr of FR) { const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9], p = fr.pos;
        for (let i = 0; i < fr.n; i += 5) for (let k = 0; k < 3; k++) { const v = p[i*3+k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
        objR += Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]) / 2 / FR.length; }
      this.radius0 = Math.min(Math.max(1.6 * L, 3 * objR), 12 * objR) * ((view && view.zoom) || 1.0);   // back far enough to see the input camera
    }
    this.radius = this.radius0; this.V3 = this.V0.slice();
  }

  viewMatrix() {
    const V = this.V3, r = this.radius, t = this.target;
    const ex = t[0] - V[6]*r, ey = t[1] - V[7]*r, ez = t[2] - V[8]*r;
    return new Float32Array([V[0], V[3], V[6], 0, V[1], V[4], V[7], 0, V[2], V[5], V[8], 0,
      -(V[0]*ex + V[1]*ey + V[2]*ez), -(V[3]*ex + V[4]*ey + V[5]*ez), -(V[6]*ex + V[7]*ey + V[8]*ez), 1]);
  }

  prepare(S, cur, view) {              // depth order (reused across nearby frames when tracked) + GPU records in that order
    const o = S.obj, time = Math.max(0, Math.min(cur, o.F - 1)); cur = Math.floor(time);
    S.nextFrame = this.continuous && o.tracked ? Math.min(cur + 1, o.F - 1) : cur;
    S.mix = S.nextFrame === cur ? 0 : time - cur;
    if (!S.ord || S.ordEpoch !== this.epoch || (!o.tracked && S.ordFrame !== cur) || (o.tracked && Math.abs(cur - S.ordFrame) >= 6)) {
      const P = o.posAt(cur), n = o.nAt(cur), vz0 = view[2], vz1 = view[6], vz2 = view[10];
      if (!S.d || S.d.length < n) { S.d = new Float32Array(n); S.key = new Uint32Array(n); }
      const d = S.d, key = S.key; let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < n; i++) { const v = vz0*P[i*3] + vz1*P[i*3+1] + vz2*P[i*3+2]; d[i] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
      const B = 65536, sc = (B - 1) / Math.max(mx - mn, 1e-9), cnt = new Uint32Array(B + 1);
      for (let i = 0; i < n; i++) { const k = B - 1 - (((d[i] - mn) * sc) | 0); key[i] = k; cnt[k + 1]++; }
      for (let b = 0; b < B; b++) cnt[b + 1] += cnt[b];
      const idx = new Uint32Array(n); for (let i = 0; i < n; i++) idx[cnt[key[i]]++] = i;
      S.ord = idx; S.ordEpoch = this.epoch; S.ordFrame = cur; S.stamp++;
    }
    const n = S.ord.length, ord = S.ord;
    const pack = (f, buf) => {
      const P = o.posAt(f), Sc = o.sclAt(f), m = o.sclMul(f), C = o.rgbaAt(f), R = o.rotAt(f);
      if (!buf || buf.byteLength < n * REC) buf = new ArrayBuffer(n * REC);
      const fo = new Float32Array(buf, 0, n * 8), bo = new Uint8Array(buf, 0, n * REC);
      for (let i = 0, o8 = 0, ob = 24; i < n; i++, o8 += 8, ob += 32) { const j = ord[i], j3 = j * 3, j4 = j * 4;
        fo[o8] = P[j3]; fo[o8+1] = P[j3+1]; fo[o8+2] = P[j3+2]; fo[o8+3] = Sc[j3] * m; fo[o8+4] = Sc[j3+1] * m; fo[o8+5] = Sc[j3+2] * m;
        bo[ob] = C[j4]; bo[ob+1] = C[j4+1]; bo[ob+2] = C[j4+2]; bo[ob+3] = C[j4+3]; bo[ob+4] = R[j4]; bo[ob+5] = R[j4+1]; bo[ob+6] = R[j4+2]; bo[ob+7] = R[j4+3]; }
      return buf;
    };
    const k = cur + ":" + S.stamp;
    if (S.bufNextKey === k) {       // last tick's second endpoint is this tick's first; keep its GPU upload
      [S.buf, S.bufNext] = [S.bufNext, S.buf]; [S.bufKey, S.bufNextKey] = [S.bufNextKey, S.bufKey];
      [S.glBuf, S.glBufNext] = [S.glBufNext, S.glBuf]; [S.up, S.upNext] = [S.upNext, S.up];
    }
    if (S.bufKey !== k) { S.buf = pack(cur, S.buf); S.bufKey = k; }
    if (S.nextFrame !== cur) {
      const nextKey = S.nextFrame + ":" + S.stamp;
      if (S.bufNextKey !== nextKey) { S.bufNext = pack(S.nextFrame, S.bufNext); S.bufNextKey = nextKey; }
    }
    return n;
  }

  drawBegin() {
    const gl = this.gl, cv = this.canvas, w = Math.round(cv.clientWidth * devicePixelRatio), h = Math.round(cv.clientHeight * devicePixelRatio);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    gl.viewport(0, 0, w, h); gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
    gl.depthMask(true); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.V3) return { w, h };
    const view = this.viewMatrix(), f = 1 / Math.tan(25 * Math.PI / 180), nr = 0.02, fa = 100;
    return { w, h, f, view, proj: new Float32Array([f*h/w,0,0,0, 0,-f,0,0, 0,0,(fa+nr)/(fa-nr),1, 0,0,-2*fa*nr/(fa-nr),0]) };
  }

  drawScene(V) {
    const gl = this.gl, b = this.sceneBox, p = this.sprog, U = n => gl.getUniformLocation(p, n);
    for (let i = 0; i < 9; i++) { gl.disableVertexAttribArray(i); gl.vertexAttribDivisor(i, 0); }
    gl.useProgram(p); gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.depthMask(true);
    gl.uniformMatrix4fv(U("uView"), false, V.view); gl.uniformMatrix4fv(U("uProj"), false, V.proj);
    gl.uniform3f(U("uMin"), b[0], b[1], b[2]); gl.uniform3f(U("uExt"), b[3]-b[0], b[4]-b[1], b[5]-b[2]);   // aP is normalized to 0..1
    gl.uniform1f(U("uPt"), 2.0 * devicePixelRatio);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sPos); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.UNSIGNED_SHORT, true, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sCol); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.drawArrays(gl.POINTS, 0, this.sceneN); gl.disableVertexAttribArray(0); gl.disableVertexAttribArray(1);
  }

  drawObjs(V, S, cur, scene) {
    if (!S) return;
    const gl = this.gl, n = this.prepare(S, cur, V.view);
    const p = this.prog, U = n => gl.getUniformLocation(p, n);
    for (let i = 0; i < 9; i++) { gl.disableVertexAttribArray(i); gl.vertexAttribDivisor(i, 0); }
    gl.useProgram(p); gl.enable(gl.BLEND); gl.depthMask(false); if (scene) gl.enable(gl.DEPTH_TEST); else gl.disable(gl.DEPTH_TEST);
    gl.uniformMatrix4fv(U("uView"), false, V.view); gl.uniformMatrix4fv(U("uProj"), false, V.proj);
    gl.uniform2f(U("uFocal"), V.f*V.h/2, V.f*V.h/2); gl.uniform2f(U("uViewport"), V.w, V.h);
    gl.uniform1f(U("uMix"), S.mix);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    if (!S.glBuf) S.glBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, S.glBuf); if (S.up !== S.bufKey) { gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(S.buf, 0, n * REC), gl.DYNAMIC_DRAW); S.up = S.bufKey; }
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, REC, 0); gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, REC, 12); gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, REC, 24); gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4); gl.vertexAttribPointer(4, 4, gl.UNSIGNED_BYTE, true, REC, 28); gl.vertexAttribDivisor(4, 1);
    if (S.nextFrame !== Math.floor(Math.max(0, Math.min(cur, S.obj.F - 1)))) {
      if (!S.glBufNext) S.glBufNext = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, S.glBufNext);
      if (S.upNext !== S.bufNextKey) { gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(S.bufNext, 0, n * REC), gl.DYNAMIC_DRAW); S.upNext = S.bufNextKey; }
    }
    for (const [loc, width, type, normalized, offset] of [[5, 3, gl.FLOAT, false, 0], [6, 3, gl.FLOAT, false, 12], [7, 4, gl.UNSIGNED_BYTE, true, 24], [8, 4, gl.UNSIGNED_BYTE, true, 28]]) {
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, width, type, normalized, REC, offset); gl.vertexAttribDivisor(loc, 1);
    }
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
  }

  draw() {
    const V = this.drawBegin(); if (!this.A || !V.view) return;
    const scene = this.sceneN && this.showScene; if (scene) this.drawScene(V);
    this.drawObjs(V, this.A, this.cur, scene);
    if (this.showTracks) this.drawTracks(V, this.tracks, this.cur);
    if (this.cam && this.showCam) this.drawCams(V.view, V.proj);
    this.updateTimeUI();
  }

  updateTimeUI() {
    this.ui.fno.textContent = (this.continuous ? (this.cur + 1).toFixed(2) : this.cur + 1) + " / " + this.nF;
    this.ui.frame.value = this.cur;
  }

  setContinuous(enabled) {
    this.continuous = enabled && (!this.A || this.A.obj.tracked);
    this.ui.frame.step = this.continuous ? "0.01" : "1";
    if (this.ui.continuous) { this.ui.continuous.checked = this.continuous; this.ui.continuous.disabled = !!this.A && !this.A.obj.tracked; }
    this.cur = Math.max(0, Math.min(this.nF - 1, this.continuous ? this.playhead : Math.floor(this.playhead)));
  }

  loop() {
    let last = performance.now();
    const tick = now => {
      const v = this.ui.inset;
      if (this.playing && this.nF && v && !v.hidden && v.readyState >= 2 && !v.paused) {   // lock to the input video's clock
        this.playhead = Math.min(this.nF - 1, v.currentTime * this.fps);
      } else if (this.playing && this.nF) {
        this.playhead = (this.playhead + (now - last) / 1000 * this.fps * this.speed) % this.nF;
      }
      if (this.playing && this.nF) this.cur = Math.min(this.nF - 1, this.continuous ? this.playhead : Math.floor(this.playhead + 1e-3));
      last = now;
      if (this.visible) this.draw();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  setPlaying(p) {
    this.playing = p; this.ui.play.textContent = p ? "Pause" : "Play"; this.ui.play.setAttribute("aria-pressed", String(!p));
    this.playhead = this.cur;
    const v = this.ui.inset; if (!v || v.hidden) return;
    if (p) { v.currentTime = (this.cur / this.fps) % (v.duration || 1); v.play().catch(() => {}); } else v.pause();
  }

  bindControls() {
    const cv = this.canvas, ui = this.ui; let drag = null;
    if (ui.inset) ui.inset.playbackRate = this.speed;
    cv.addEventListener("contextmenu", e => e.preventDefault());
    cv.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointerup", () => drag = null);
    cv.addEventListener("pointermove", e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) { const s = this.radius * 0.0015, V = this.V3;
        for (let k = 0; k < 3; k++) this.target[k] -= (V[k]*dx + V[3+k]*dy) * s;
      } else this.V3 = mm3(mm3(rotX(dy * 0.006), rotY(-dx * 0.006)), this.V3);
      this.epoch++;
    });
    cv.addEventListener("wheel", e => { e.preventDefault(); this.radius = Math.max(0.05, Math.min(50, this.radius * Math.exp(e.deltaY * 0.001))); this.epoch++; }, { passive: false });
    ui.play.addEventListener("click", () => this.setPlaying(!this.playing));
    ui.frame.addEventListener("input", e => { this.setPlaying(false); this.cur = this.playhead = +e.target.value; if (ui.inset && !ui.inset.hidden) ui.inset.currentTime = this.cur / this.fps; });
    if (ui.continuous) ui.continuous.addEventListener("change", e => this.setContinuous(e.target.checked));
    ui.reset.addEventListener("click", () => { this.V3 = this.V0.slice(); this.radius = this.radius0; this.epoch++; });
    if (ui.scene) ui.scene.addEventListener("change", e => { this.showScene = e.target.checked; });
    if (ui.camchk) ui.camchk.addEventListener("change", e => { this.showCam = e.target.checked; });
    if (ui.trk) { this.showTracks = ui.trk.checked; ui.trk.addEventListener("change", e => { this.showTracks = e.target.checked; }); }
  }
}
class Viewer4DCmp extends Viewer4D {
  constructor(root, opts = {}) { super(root, opts); this.B = null; this.split = 0.5; this.tokB = 0; }

  async loadBaseline(objects, label) {        // swap the right-hand method without touching camera, scene or frame
    const tok = ++this.tokB; this.ui.load.hidden = false; this.ui.load.textContent = "Loading " + (label || "baseline") + "…";
    try {
      const bufs = await Promise.all(objects.map(getBin)), objs = await Promise.all(bufs.map((b, i) => parseAny(b, objects[i]))); if (tok !== this.tokB) return;
      this.buildTracks(objs);
      const m = objs.length > 1 ? mergeObjects(objs) : objs[0]; this.B = side(m); this.tracksB = m.tracks;
      this.ui.load.hidden = true;
    } catch (e) { if (tok === this.tokB) this.ui.load.textContent = "Could not load " + (label || "baseline") + " (" + e.message + ")."; }
  }

  draw() {
    const V = this.drawBegin(), gl = this.gl; if (!this.A || !V.view) return;
    const scene = this.sceneN && this.showScene, x = Math.round(V.w * this.split);
    gl.enable(gl.SCISSOR_TEST);
    for (const [x0, w, S, tr] of [[0, x, this.A, this.tracks], [x, V.w - x, this.B, this.tracksB]]) {
      if (w <= 0) continue; gl.scissor(x0, 0, w, V.h);
      if (scene) this.drawScene(V); if (S) this.drawObjs(V, S, this.cur, scene);
      if (this.showTracks && S) this.drawTracks(V, tr, Math.min(this.cur, S.obj.F - 1));
    }
    gl.disable(gl.SCISSOR_TEST);
    if (this.cam && this.showCam) this.drawCams(V.view, V.proj);
    this.updateTimeUI();
  }
}
window.Viewer4D = Viewer4D; window.Viewer4DCmp = Viewer4DCmp;
})();
