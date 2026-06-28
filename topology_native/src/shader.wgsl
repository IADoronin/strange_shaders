// Raymarcher топологических пространств + оптика «заторенной вселенной».
// Режимы (mode):
//   0 T³ · 1 зеркальный орбифолд · 2 скрученный T³ · 3 звезда+планета(T³) ·
//   4 обычное R³ · 5 ЗАТОРЕННАЯ ВСЕЛЕННАЯ (звёзды = эмиссионные точки с PSF, HDR).

struct U {
    a: vec4<f32>,   // res.x, res.y, time, cell(период заворота)
    b: vec4<f32>,   // mouse.x, mouse.y, exposure, star_lum
    c: vec4<f32>,   // mode, pad, pad, pad
};
@group(0) @binding(0) var<uniform> u: U;

const FOCAL: f32 = 1.2;
const PI: f32 = 3.14159265;
// сцена «заторенной вселенной»
const STAR_OFF   = vec3<f32>(0.0, 0.0, 0.0);     // центр звезды в ячейке
const PLANET_OFF = vec3<f32>(2.6, 0.0, 0.0);     // центр планеты в ячейке
const R_PLANET: f32 = 0.5;
const R_STARCORE: f32 = 0.22;                    // мин. размер ядра PSF (мировые ед.)
const STAR_COL = vec3<f32>(1.0, 0.85, 0.55);

// ---- helpers ----
fn modf3(x: vec3<f32>, y: f32) -> vec3<f32> { return x - y * floor(x / y); }

fn sdSphere(p: vec3<f32>, r: f32) -> f32 { return length(p) - r; }
fn sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}
fn sdSeg(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a; let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
// асимметричная хиральная фигура «F»
fn glyphF(q: vec3<f32>) -> f32 {
    let p = q - vec3<f32>(0.0, -1.0, 0.9);
    let r = 0.085;
    let spine = sdSeg(p, vec3<f32>(0.0, -0.55, 0.0), vec3<f32>(0.0, 0.55, 0.0), r);
    let top   = sdSeg(p, vec3<f32>(0.0, 0.55, 0.0), vec3<f32>(0.5, 0.55, 0.0), r);
    let mid   = sdSeg(p, vec3<f32>(0.0, 0.10, 0.0), vec3<f32>(0.34, 0.10, 0.0), r);
    return min(spine, min(top, mid));
}

fn opU(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { if (a.x < b.x) { return a; } return b; }

// fold: R³ -> фундаментальная область (топология = выбор группы Γ)
fn foldP(p: vec3<f32>, mode: i32, cell: f32) -> vec3<f32> {
    let half = 0.5 * cell;
    if (mode == 4) { return p; }                       // обычное R³ — без склейки
    if (mode == 1) { return abs(modf3(p, 2.0 * cell) - cell) - half; } // зеркальный орбифолд
    if (mode == 2) {                                   // скрученный T³
        var q = modf3(p + half, cell) - half;
        let ix = floor((p.x + half) / cell);
        if (modf3(vec3<f32>(ix, 0.0, 0.0), 2.0).x > 0.5) { q = vec3<f32>(q.x, -q.y, -q.z); }
        return q;
    }
    return modf3(p + half, cell) - half;               // T³ (0,3,5)
}

// содержимое ячейки -> (расстояние, материал)
fn cellScene(q: vec3<f32>, mode: i32) -> vec2<f32> {
    if (mode == 4) {
        let star   = sdSphere(q - vec3<f32>(0.0, 0.0, 0.0), 1.2);
        let planet = sdSphere(q - vec3<f32>(3.2, 0.6, 0.0), 0.55);
        if (star < planet) { return vec2<f32>(star, 2.0); }
        return vec2<f32>(planet, 3.0);
    }
    if (mode == 3) {
        let star   = sdSphere(q - vec3<f32>(-0.9, 0.0, 0.0), 0.55);
        let planet = sdSphere(q - vec3<f32>(1.0, 0.6, 0.3), 0.30);
        if (star < planet) { return vec2<f32>(star, 2.0); }
        return vec2<f32>(planet, 3.0);
    }
    let bx = length(q.yz) - 0.10;
    let by = length(q.xz) - 0.10;
    let bz = length(q.xy) - 0.10;
    let bars = min(bx, min(by, bz));
    let node = sdSphere(q, 0.30);
    let tor  = sdTorus(q, vec2<f32>(1.15, 0.16));
    var res = vec2<f32>(min(bars, min(node, tor)), 1.0);
    res = opU(res, vec2<f32>(glyphF(q), 4.0));
    return res;
}

fn map(p: vec3<f32>, mode: i32, cell: f32) -> vec2<f32> {
    return cellScene(foldP(p, mode, cell), mode);
}
fn calcNormal(p: vec3<f32>, mode: i32, cell: f32) -> vec3<f32> {
    let e = vec2<f32>(0.0015, 0.0);
    return normalize(vec3<f32>(
        map(p + e.xyy, mode, cell).x - map(p - e.xyy, mode, cell).x,
        map(p + e.yxy, mode, cell).x - map(p - e.yxy, mode, cell).x,
        map(p + e.yyx, mode, cell).x - map(p - e.yyx, mode, cell).x));
}

// ---- «заторенная вселенная»: планета (T³) и центры звёзд ----
fn planetDist(p: vec3<f32>, cell: f32) -> f32 {
    let q = modf3(p + 0.5 * cell, cell) - 0.5 * cell;
    return sdSphere(q - PLANET_OFF, R_PLANET);
}
fn nearestStarCenter(p: vec3<f32>, cell: f32) -> vec3<f32> {
    let q = modf3(p + 0.5 * cell, cell) - 0.5 * cell;
    return p - (q - STAR_OFF);
}
fn starCenterDist(p: vec3<f32>, cell: f32) -> f32 {
    let q = modf3(p + 0.5 * cell, cell) - 0.5 * cell;
    return length(q - STAR_OFF);
}
fn planetNormal(p: vec3<f32>, cell: f32) -> vec3<f32> {
    let e = vec2<f32>(0.0015, 0.0);
    return normalize(vec3<f32>(
        planetDist(p + e.xyy, cell) - planetDist(p - e.xyy, cell),
        planetDist(p + e.yxy, cell) - planetDist(p - e.yxy, cell),
        planetDist(p + e.yyx, cell) - planetDist(p - e.yyx, cell)));
}

// положение ближайшей звезды СЦЕНЫ для освещения планеты (режимы 3 и 4)
fn sceneStarCenter(p: vec3<f32>, mode: i32, cell: f32) -> vec3<f32> {
    if (mode == 4) { return vec3<f32>(0.0, 0.0, 0.0); }   // R³: звезда в начале координат
    let q = modf3(p + 0.5 * cell, cell) - 0.5 * cell;     // T³: звезда (-0.9,0,0) в каждой ячейке
    return p - (q - vec3<f32>(-0.9, 0.0, 0.0));
}

struct Ray { o: vec3<f32>, d: vec3<f32> };
fn make_ray(uv: vec2<f32>, mode: i32, cell: f32, t: f32, mouse: vec2<f32>) -> Ray {
    var ro: vec3<f32>;
    var ta: vec3<f32>;
    if (mode == 4) {
        let a = t * 0.35;
        ro = vec3<f32>(9.0 * sin(a), 2.5 + 2.0 * sin(t * 0.2) + mouse.y * 4.0, 9.0 * cos(a));
        ta = vec3<f32>(2.0, 0.4, 0.0);
    } else {
        ro = vec3<f32>(0.15 * cell * sin(t * 0.3), 0.15 * cell * cos(t * 0.23), t * 0.3 * cell);
        let yaw = mouse.x * PI;
        let pitch = mouse.y * 1.2;
        ta = ro + vec3<f32>(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch));
    }
    let f = normalize(ta - ro);
    let r = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), f));
    let up = cross(f, r);
    let rd = normalize(mat3x3<f32>(r, up, f) * vec3<f32>(uv, FOCAL));
    return Ray(ro, rd);
}

// ---- режимы 0..4: порт WebGL ----
fn render_classic(uv: vec2<f32>, mode: i32, cell: f32, t: f32, mouse: vec2<f32>) -> vec3<f32> {
    let half = 0.5 * cell;
    let ray = make_ray(uv, mode, cell, t, mouse);
    let ro = ray.o; let rd = ray.d;

    var d = 0.0; let tmax = max(220.0, 6.0 * cell); var glow = 0.0;
    var hit = false; var hitSteps = 0; var p = ro;
    for (var i = 0; i < 192; i = i + 1) {
        p = ro + rd * d;
        let s = map(p, mode, cell);
        if (mode == 3 || mode == 4) {
            let q = modf3(p + half, cell) - half;
            let ds3 = length(q - vec3<f32>(-0.9, 0.0, 0.0)) - 0.55;   // звезда в ячейке T³
            let ds4 = sdSphere(p, 1.2);                               // одна звезда в R³
            let dsx = select(ds4, ds3, mode == 3);
            glow = glow + 0.018 / (1.0 + 10.0 * dsx * dsx);
        }
        let eps = 0.001 + 0.0006 * d;
        if (s.x < eps) { hit = true; hitSteps = i; break; }
        d = d + s.x;
        if (d > tmax) { hitSteps = i; break; }
        hitSteps = i;
    }

    var col = vec3<f32>(0.0);
    if (hit) {
        let n = calcNormal(p, mode, cell);
        let lig = normalize(vec3<f32>(0.5, 0.8, -0.3));
        let dif = clamp(dot(n, lig), 0.0, 1.0);
        let amb = 0.35 + 0.65 * n.y;
        let m = map(p, mode, cell);
        if (m.y > 3.5) {
            col = vec3<f32>(1.0, 0.55, 0.12) * (0.3 * amb + dif) + 0.05;
        } else if (m.y > 2.5) {                       // планета — звезда И её ОБРАЗЫ
            if (mode == 3) {
                // T³: свет приходит от звезды и её 26 ближайших образов (3×3×3), ~1/d²
                let center = p - (modf3(p + 0.5 * cell, cell) - 0.5 * cell);
                var diff = 0.0;
                for (var ix = -1; ix <= 1; ix = ix + 1) {
                    for (var iy = -1; iy <= 1; iy = iy + 1) {
                        for (var iz = -1; iz <= 1; iz = iz + 1) {
                            let img = center + cell * vec3<f32>(f32(ix), f32(iy), f32(iz)) + vec3<f32>(-0.9, 0.0, 0.0);
                            let dv = img - p;
                            diff = diff + max(dot(n, normalize(dv)), 0.0) / (0.4 + 0.2 * dot(dv, dv));
                        }
                    }
                }
                col = vec3<f32>(0.30, 0.55, 0.95) * (0.04 + diff);
            } else {
                // R³: одна звезда, образов нет
                let ld = normalize(sceneStarCenter(p, mode, cell) - p);
                col = vec3<f32>(0.30, 0.55, 0.95) * (0.05 + max(dot(n, ld), 0.0));
            }
        } else if (m.y > 1.5) {
            col = vec3<f32>(1.0, 0.85, 0.45) * 1.5;
        } else {
            let q = modf3(p + half, cell) - half;
            let base = 0.5 + 0.5 * cos(vec3<f32>(0.0, 2.0, 4.0) + length(q) * 2.0 + f32(mode));
            col = base * (0.25 * amb + dif);
        }
        col = col * (1.0 - 0.5 * f32(hitSteps) / 192.0);
    }
    let fogc = vec3<f32>(0.02, 0.03, 0.06);
    col = mix(fogc, col, exp(-d / max(200.0, 2.0 * cell)));   // характерная дистанция тумана ~ период
    col = col + glow * vec3<f32>(1.0, 0.7, 0.35);
    return pow(col, vec3<f32>(0.4545));
}

// ---- режим 5: заторенная вселенная (PSF + HDR) ----
fn render_universe(uv: vec2<f32>, cell: f32, t: f32, mouse: vec2<f32>, exposure: f32, starLum: f32, H: f32) -> vec3<f32> {
    let ray = make_ray(uv, 5, cell, t, mouse);
    let ro = ray.o; let rd = ray.d;

    let kAng = 1.6 / H;        // угловой размер ~ пикселя -> ядро не схлопывается
    let tmax = max(160.0, 6.0 * cell);
    var tt = 0.0;
    var emis = vec3<f32>(0.0);
    var hit = false; var hitP = ro;
    for (var i = 0; i < 256; i = i + 1) {
        let p = ro + rd * tt;
        let dSolid = planetDist(p, cell);
        let dc = starCenterDist(p, cell);
        // PSF-вклад звезды: яркость ~ L/d², ядро core>=kAng*t (>= ~пиксель)
        let core = max(R_STARCORE, kAng * tt);
        let w = (core * core) / (core * core + dc * dc);
        let dt = max(0.01, min(dSolid, 0.3 * dc + 0.04));
        // Интеграл по лучу с ядром ширины core (∝ t -> фиксированный угловой размер).
        // Делитель t^3 даёт суммарный вклад образа ∝ 1/t^2 (закон обратных квадратов):
        // звезда ТУСКНЕЕТ как 1/d², но не исчезает (пятно всегда >= ~пикселя).
        emis = emis + STAR_COL * starLum * w * dt / (tt * tt * tt + 0.5);
        if (dSolid < 0.001 + 0.0006 * tt) { hit = true; hitP = p; break; }
        tt = tt + dt;
        if (tt > tmax) { break; }
    }

    var surf = vec3<f32>(0.0);
    if (hit) {
        let n = planetNormal(hitP, cell);
        let sc = nearestStarCenter(hitP, cell);
        let ld = normalize(sc - hitP);
        let dif = max(dot(n, ld), 0.0);
        let dStar2 = dot(sc - hitP, sc - hitP);
        let recv = starLum / (0.3 + 0.15 * dStar2);          // освещённость планеты её солнцем
        surf = vec3<f32>(0.35, 0.45, 0.55) * (0.04 + dif * recv) * STAR_COL;
        // лёгкий туман по дистанции до планеты
        surf = surf * exp(-tt / max(100.0, 2.0 * cell));
    }

    let hdr = surf + emis;
    // тон-маппинг (экспозиция) + гамма
    let mapped = vec3<f32>(1.0) - exp(-hdr * exposure);
    return pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.4545));
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let res = u.a.xy;
    let time = u.a.z;
    let cell = u.a.w;
    let mouse = u.b.xy;
    let exposure = u.b.z;
    let starLum = u.b.w;
    let mode = i32(u.c.x + 0.5);

    // координата пикселя (origin внизу, как в GLSL)
    let px = vec2<f32>(frag.x, res.y - frag.y);

    var col: vec3<f32>;
    if (mode == 5) {
        // SSAA 2x2 — чтобы далёкие планеты не мерцали, звёзды и так гладкие (PSF)
        var acc = vec3<f32>(0.0);
        var off = array<vec2<f32>, 4>(
            vec2<f32>(-0.25, -0.25), vec2<f32>(0.25, -0.25),
            vec2<f32>(-0.25, 0.25), vec2<f32>(0.25, 0.25));
        for (var k = 0; k < 4; k = k + 1) {
            let uv = (px + off[k] - 0.5 * res) / res.y;
            acc = acc + render_universe(uv, cell, time, mouse, exposure, starLum, res.y);
        }
        col = acc * 0.25;
    } else {
        let uv = (px - 0.5 * res) / res.y;
        col = render_classic(uv, mode, cell, time, mouse);
    }
    return vec4<f32>(col, 1.0);
}
