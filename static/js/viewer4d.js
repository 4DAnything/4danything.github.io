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
uniform mat4 uView, uProj;
uniform vec2 uFocal, uViewport;
out vec4 vColor; out vec2 vPos;
void main() {
  vec4 cam = uView * vec4(aPos, 1.0);
  if (cam.z < 0.05) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 q = normalize(aRot * (255.0/128.0) - 1.0);
  float w=q.x, x=q.y, y=q.z, z=q.w;
  mat3 R = mat3(1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y),
                2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x),
                2.*(x*z+w*y), 2.*(y*z-w*x), 1.-2.*(x*x+y*y));
  mat3 S = mat3(aScale.x,0.,0., 0.,aScale.y,0., 0.,0.,aScale.z);
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
  gl_Position = vec4(clip.xy + off, clip.z, clip.w); vColor = aColor; vPos = corner;
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
const SFS = `#version 300 es
precision highp float;
in vec3 vC; out vec4 frag; void main() { frag = vec4(vC, 1.0); }`;

function mm3(A, B) { const C = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i*3+j] = A[i*3]*B[j] + A[i*3+1]*B[3+j] + A[i*3+2]*B[6+j]; return C; }
function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [1,0,0, 0,c,-s, 0,s,c]; }
function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [c,0,s, 0,1,0, -s,0,c]; }

function parse4dgs(ab) {             // -> {fps, frames:[{raw (32B recs), n, pos}]}
  const dv = new DataView(ab);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error("bad 4dgs");
  const F = dv.getUint32(4, true), fps = dv.getFloat32(8, true) || 8, counts = new Uint32Array(ab, 16, F);
  const hdrs = new Float32Array(ab, 16 + 4*F, 8*F); let off = 16 + 36*F; const frames = [];
  for (let f = 0; f < F; f++) {
    const n = counts[f], u8 = new Uint8Array(ab, off, n * 18), out = new ArrayBuffer(n * REC);
    const fo = new Float32Array(out), bo = new Uint8Array(out), pos = new Float32Array(n * 3);
    const h = hdrs.subarray(f*8, f*8+8), ex = (h[3]-h[0])/65535, ey = (h[4]-h[1])/65535, ez = (h[5]-h[2])/65535, ls = (h[7]-h[6])/255;
    for (let i = 0; i < n; i++) {
      const s = i*18, o = i*8;
      fo[o]   = pos[i*3]   = h[0] + (u8[s]   | (u8[s+1]<<8)) * ex;
      fo[o+1] = pos[i*3+1] = h[1] + (u8[s+2] | (u8[s+3]<<8)) * ey;
      fo[o+2] = pos[i*3+2] = h[2] + (u8[s+4] | (u8[s+5]<<8)) * ez;
      fo[o+3] = Math.exp(h[6] + u8[s+6]*ls); fo[o+4] = Math.exp(h[6] + u8[s+7]*ls); fo[o+5] = Math.exp(h[6] + u8[s+8]*ls);
      for (let k = 0; k < 8; k++) bo[i*32+24+k] = u8[s+9+k];
    }
    frames.push({ raw: out, n, pos }); off += n * 18;
  }
  return { fps, frames };
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

function mergeObjects(objs) {        // concatenate per-frame splats of several objects into one stream
  const F = Math.min(...objs.map(o => o.frames.length)), frames = [];
  for (let f = 0; f < F; f++) {
    const parts = objs.map(o => o.frames[f]), n = parts.reduce((s, p) => s + p.n, 0);
    const raw = new Uint8Array(n * REC), pos = new Float32Array(n * 3); let k = 0;
    for (const p of parts) { raw.set(new Uint8Array(p.raw), k * REC); pos.set(p.pos, k * 3); k += p.n; }
    frames.push({ raw: raw.buffer, n, pos });
  }
  return { fps: objs[0].fps, frames };
}

class Viewer4D {
  constructor(root, opts = {}) {
    this.root = root; this.opts = opts;
    this.canvas = root.querySelector("canvas");
    this.ui = { play: root.querySelector("[data-v=play]"), frame: root.querySelector("[data-v=frame]"),
                fno: root.querySelector("[data-v=fno]"), scene: root.querySelector("[data-v=scene]"),
                reset: root.querySelector("[data-v=reset]"), load: root.querySelector("[data-v=load]"),
                inset: root.querySelector("video[data-v=inset]") };
    const gl = this.gl = this.canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) { this.ui.load.textContent = "WebGL2 is not available in this browser."; return; }
    const mk = (vs, fs) => { const p = gl.createProgram();
      for (const [t, s] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) { const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh); gl.attachShader(p, sh); }
      gl.linkProgram(p); return p; };
    this.prog = mk(VS, FS); this.sprog = mk(SVS, SFS);
    this.quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2,-2, 2,-2, -2,2, 2,2]), gl.STATIC_DRAW);
    this.dataBuf = gl.createBuffer(); this.sPos = gl.createBuffer(); this.sCol = gl.createBuffer();
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    this.frames = []; this.sorted = []; this.epoch = 0; this.cur = 0; this.playing = true; this.fps = 8;
    this.sceneN = 0; this.showScene = true; this.visible = true; this.token = 0;
    this.bg = opts.bg || [1, 1, 1]; this.speed = opts.speed || 0.5;
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
      const objs = bufs.map(parse4dgs), m = objs.length > 1 ? mergeObjects(objs) : objs[0];
      this.frames = m.frames; this.fps = m.fps; this.sorted = this.frames.map(() => null);
      this.sceneN = 0; if (sceneAb) this.loadScene(sceneAb);
      this.fitView(def.view); this.cur = 0; this.acc = 0; this.epoch++;
      this.ui.frame.max = this.frames.length - 1;
      const v = this.ui.inset;
      if (v) { if (def.input) { v.src = def.input; v.hidden = false; v.play().catch(() => {}); } else { v.removeAttribute("src"); v.hidden = true; } }
      this.ui.load.hidden = true;
    } catch (e) { if (tok === this.token) this.ui.load.textContent = "Could not load scene (" + e.message + ")."; }
  }

  loadScene(ab) {
    const gl = this.gl, dv = new DataView(ab);
    if (dv.getUint32(0, true) !== SCENE_MAGIC) return;
    this.sceneN = dv.getUint32(4, true); this.sceneBox = new Float32Array(ab.slice(8, 32));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sPos); gl.bufferData(gl.ARRAY_BUFFER, new Uint16Array(ab, 32, this.sceneN * 3), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sCol); gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(ab, 32 + this.sceneN * 6, this.sceneN * 3), gl.STATIC_DRAW);
  }

  fitView(view) {                     // orbit centre = object centroid over all frames
    let c = [0, 0, 0], n = 0;
    for (const fr of this.frames) for (let i = 0; i < fr.n; i += 7) { c[0] += fr.pos[i*3]; c[1] += fr.pos[i*3+1]; c[2] += fr.pos[i*3+2]; n++; }
    c = c.map(v => v / Math.max(n, 1)); let ext = 0;
    for (const fr of this.frames) for (let i = 0; i < fr.n; i += 7) { const dx = fr.pos[i*3]-c[0], dy = fr.pos[i*3+1]-c[1], dz = fr.pos[i*3+2]-c[2]; ext = Math.max(ext, dx*dx+dy*dy+dz*dz); }
    this.target = c;
    if (view && view.legacy) {
      this.radius0 = Math.sqrt(ext) * (view.zoom || 1.6) + 0.2; this.V0 = mm3(rotX(view.pitch || 0.12), rotY(view.yaw || -0.35));
    } else {                          // start at the input camera (origin, OpenCV axes) looking at the centroid, then turn slightly
      const L = Math.hypot(c[0], c[1], c[2]) || 1, f = c.map(v => v / L);
      let r = [f[2], 0, -f[0]]; const rl = Math.hypot(r[0], r[2]) || 1; r = r.map(v => v / rl);   // down (0,1,0) x f
      const d = [f[1]*r[2] - f[2]*r[1], f[2]*r[0] - f[0]*r[2], f[0]*r[1] - f[1]*r[0]];               // f x r
      this.V0 = mm3(mm3(rotX((view && view.pitch) || 0.15), rotY((view && view.yaw) || -0.3)), [...r, ...d, ...f]);
      let objR = 0;                   // mean per-frame half-diagonal of the objects' box
      for (const fr of this.frames) { const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
        for (let i = 0; i < fr.n; i += 5) for (let k = 0; k < 3; k++) { const v = fr.pos[i*3+k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
        objR += Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]) / 2 / this.frames.length; }
      this.radius0 = Math.min(Math.max(L, 2.5 * objR), 6 * objR) * ((view && view.zoom) || 1.0);
    }
    this.radius = this.radius0; this.V3 = this.V0.slice();
  }

  viewMatrix() {
    const V = this.V3, r = this.radius, t = this.target;
    const ex = t[0] - V[6]*r, ey = t[1] - V[7]*r, ez = t[2] - V[8]*r;
    return new Float32Array([V[0], V[3], V[6], 0, V[1], V[4], V[7], 0, V[2], V[5], V[8], 0,
      -(V[0]*ex + V[1]*ey + V[2]*ez), -(V[3]*ex + V[4]*ey + V[5]*ez), -(V[6]*ex + V[7]*ey + V[8]*ez), 1]);
  }

  sortFrame(f, view) {
    const fr = this.frames[f], n = fr.n, d = new Float32Array(n), vz = [view[2], view[6], view[10]];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) { const v = vz[0]*fr.pos[i*3] + vz[1]*fr.pos[i*3+1] + vz[2]*fr.pos[i*3+2]; d[i] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
    const B = 65536, sc = (B - 1) / Math.max(mx - mn, 1e-9), key = new Uint32Array(n), cnt = new Uint32Array(B + 1);
    for (let i = 0; i < n; i++) { const k = B - 1 - (((d[i] - mn) * sc) | 0); key[i] = k; cnt[k + 1]++; }
    for (let b = 0; b < B; b++) cnt[b + 1] += cnt[b];
    const idx = new Uint32Array(n); for (let i = 0; i < n; i++) idx[cnt[key[i]]++] = i;
    const src = new Uint32Array(fr.raw), out = new Uint32Array(n * 8);
    for (let i = 0; i < n; i++) { const s = idx[i] * 8, o = i * 8; for (let k = 0; k < 8; k++) out[o+k] = src[s+k]; }
    this.sorted[f] = { buf: out, epoch: this.epoch };
  }

  draw() {
    const gl = this.gl, cv = this.canvas, w = Math.round(cv.clientWidth * devicePixelRatio), h = Math.round(cv.clientHeight * devicePixelRatio);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    gl.viewport(0, 0, w, h); gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
    gl.depthMask(true); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.frames.length) return;
    const view = this.viewMatrix(), f = 1 / Math.tan(25 * Math.PI / 180), nr = 0.02, fa = 100;
    const proj = new Float32Array([f*h/w,0,0,0, 0,-f,0,0, 0,0,(fa+nr)/(fa-nr),1, 0,0,-2*fa*nr/(fa-nr),0]);
    const scene = this.sceneN && this.showScene;
    for (let i = 0; i < 5; i++) { gl.disableVertexAttribArray(i); gl.vertexAttribDivisor(i, 0); }
    if (scene) {
      const b = this.sceneBox, p = this.sprog, U = n => gl.getUniformLocation(p, n);
      gl.useProgram(p); gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
      gl.uniformMatrix4fv(U("uView"), false, view); gl.uniformMatrix4fv(U("uProj"), false, proj);
      gl.uniform3f(U("uMin"), b[0], b[1], b[2]); gl.uniform3f(U("uExt"), b[3]-b[0], b[4]-b[1], b[5]-b[2]);   // aP is normalized to 0..1
      gl.uniform1f(U("uPt"), 2.0 * devicePixelRatio);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.sPos); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.UNSIGNED_SHORT, true, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.sCol); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.drawArrays(gl.POINTS, 0, this.sceneN); gl.disableVertexAttribArray(0); gl.disableVertexAttribArray(1);
    }
    const s = this.sorted[this.cur];
    if (!s || s.epoch !== this.epoch) this.sortFrame(this.cur, view);
    const p = this.prog, U = n => gl.getUniformLocation(p, n), fr = this.frames[this.cur];
    gl.useProgram(p); gl.enable(gl.BLEND); gl.depthMask(false); if (scene) gl.enable(gl.DEPTH_TEST); else gl.disable(gl.DEPTH_TEST);
    gl.uniformMatrix4fv(U("uView"), false, view); gl.uniformMatrix4fv(U("uProj"), false, proj);
    gl.uniform2f(U("uFocal"), f*h/2, f*h/2); gl.uniform2f(U("uViewport"), w, h);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dataBuf); gl.bufferData(gl.ARRAY_BUFFER, this.sorted[this.cur].buf, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, REC, 0); gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, REC, 12); gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, REC, 24); gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4); gl.vertexAttribPointer(4, 4, gl.UNSIGNED_BYTE, true, REC, 28); gl.vertexAttribDivisor(4, 1);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, fr.n);
    this.ui.fno.textContent = (this.cur + 1) + " / " + this.frames.length; this.ui.frame.value = this.cur;
  }

  loop() {
    let last = performance.now(); this.acc = 0;
    const tick = now => {
      if (this.playing && this.frames.length) {
        this.acc += (now - last) / 1000 * this.fps * this.speed;
        if (this.acc >= 1) {
          const nf = (this.cur + Math.floor(this.acc)) % this.frames.length;
          if (nf < this.cur && this.ui.inset && this.ui.inset.duration) this.ui.inset.currentTime = 0;
          this.cur = nf; this.acc %= 1;
        }
      }
      last = now;
      if (this.visible) this.draw();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  setPlaying(p) {
    this.playing = p; this.ui.play.textContent = p ? "Pause" : "Play"; this.ui.play.setAttribute("aria-pressed", String(!p));
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
    ui.frame.addEventListener("input", e => { this.setPlaying(false); this.cur = +e.target.value; if (ui.inset && !ui.inset.hidden) ui.inset.currentTime = this.cur / this.fps; });
    ui.reset.addEventListener("click", () => { this.V3 = this.V0.slice(); this.radius = this.radius0; this.epoch++; });
    if (ui.scene) ui.scene.addEventListener("change", e => { this.showScene = e.target.checked; });
  }
}
window.Viewer4D = Viewer4D;
})();
