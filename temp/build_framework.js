const pptxgen = require("pptxgenjs");

// ---------- Palette (underground robotics / SLAM + RL) ----------
const NAVY   = "0E2A47"; // primary dark
const NAVY2  = "16395E";
const TEAL   = "1C7293"; // SLAM base
const TEAL_D = "0F4C5C";
const CYAN   = "2A9D8F"; // degradation layer
const AMBER  = "E9A23B"; // decision / RL accent
const AMBER_D= "C77F1A";
const ORANGE = "E76F51"; // navigation
const LIGHT  = "F4F7FA";
const CARD   = "FFFFFF";
const INK    = "1A2230";
const MUTE   = "5B6B7C";
const HFONT  = "Microsoft YaHei";
const BFONT  = "Microsoft YaHei";

let pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
pres.author = "Claude";
pres.title = "面向地下退化环境的具身主动感知与自适应导航机理";

const W = 13.333, H = 7.5;
const mkShadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 90, opacity: 0.18 });

// small helper: rounded card
function card(s, x, y, w, h, fill, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: opts.r || 0.08,
    fill: { color: fill },
    line: opts.line ? { color: opts.line, width: opts.lw || 1 } : { type: "none" },
    shadow: opts.shadow ? mkShadow() : undefined,
  });
}
function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, {
    x, y, w, h, margin: o.margin != null ? o.margin : 2,
    fontFace: o.font || BFONT, fontSize: o.fs || 12, color: o.color || INK,
    bold: !!o.bold, italic: !!o.italic, align: o.align || "center",
    valign: o.valign || "middle", lineSpacingMultiple: o.lsm || 1.0,
    breakLine: o.breakLine,
  });
}
function downArrow(s, x, y, h, color = NAVY) {
  s.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color, width: 2.25, endArrowType: "triangle" } });
}
function rightArrow(s, x, y, w, color = MUTE) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2, endArrowType: "triangle" } });
}

/* =======================================================================
 * SLIDE 1 — Title
 * ===================================================================== */
let s1 = pres.addSlide();
s1.background = { color: NAVY };
// motif: thin amber bar on left
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: H, fill: { color: AMBER } });
// faint geometric grid (tunnel feel)
for (let i = 0; i < 6; i++) {
  s1.addShape(pres.shapes.RECTANGLE, { x: 9.4 + i * 0.0, y: 0.0, w: 0.012, h: H, fill: { color: "FFFFFF", transparency: 92 } });
}
s1.addShape(pres.shapes.OVAL, { x: 9.0, y: -2.2, w: 6.5, h: 6.5, fill: { color: TEAL, transparency: 80 }, line: { type: "none" } });
s1.addShape(pres.shapes.OVAL, { x: 10.2, y: -1.0, w: 4.5, h: 4.5, fill: { color: AMBER, transparency: 86 }, line: { type: "none" } });

txt(s1, "研究内容（三）", 0.9, 1.35, 9, 0.5, { fs: 18, color: AMBER, bold: true, align: "left" });
txt(s1, "面向地下退化环境的\n具身主动感知与自适应导航机理", 0.85, 1.9, 11.4, 1.9,
  { fs: 40, color: "FFFFFF", bold: true, align: "left", font: HFONT, lsm: 1.08 });
txt(s1,
  "LiDAR–惯性紧耦合建图底座（FAST-LIO2 / Faster-LIO）  +  可定位性在线量化（X-ICP）  +  Flow Q-Learning 主动探索决策",
  0.9, 4.05, 11.2, 0.8, { fs: 15, color: "CADCFC", align: "left", lsm: 1.2 });

// three pillars
const pills = [
  ["退化感知机理", "几何自相似环境下感知可靠性的可计算表征", TEAL],
  ["主动决策机理", "探索–利用–可定位性三元耦合的最优决策", AMBER],
  ["自适应导航机理", "感知-决策-导航闭环的在线自适应", ORANGE],
];
pills.forEach(([t, d, c], i) => {
  const x = 0.9 + i * 3.95;
  card(s1, x, 5.35, 3.7, 1.5, NAVY2, { line: c, lw: 1.5, r: 0.1 });
  s1.addShape(pres.shapes.RECTANGLE, { x, y: 5.35, w: 0.1, h: 1.5, fill: { color: c } });
  txt(s1, t, x + 0.28, 5.55, 3.3, 0.5, { fs: 16, color: "FFFFFF", bold: true, align: "left" });
  txt(s1, d, x + 0.28, 6.05, 3.25, 0.7, { fs: 11.5, color: "AEC2D6", align: "left", lsm: 1.15 });
});

/* =======================================================================
 * SLIDE 2 — Method Framework (方法框图)
 * ===================================================================== */
let s2 = pres.addSlide();
s2.background = { color: LIGHT };
txt(s2, "总体方法框图：退化感知驱动的具身主动 SLAM 闭环", 0.4, 0.18, 12.5, 0.55,
  { fs: 24, color: NAVY, bold: true, align: "left", font: HFONT });

const LBL_X = 0.4, LBL_W = 1.55;          // layer label column
const C_X = 2.15, C_R = 11.55;            // content area right edge
const C_W = C_R - C_X;                    // content width
const FB_X = 12.05;                       // feedback arrow column

function layerLabel(s, y, h, text, color) {
  card(s, LBL_X, y, LBL_W, h, color, { r: 0.07 });
  txt(s, text, LBL_X + 0.04, y, LBL_W - 0.08, h, { fs: 13, color: "FFFFFF", bold: true, lsm: 1.05 });
}

// ---- Row 0: Sensor input ----
let y0 = 0.92, h0 = 0.66;
layerLabel(s2, y0, h0, "传感器\n输入", NAVY);
const sensors = ["3D LiDAR\n(机械 / 固态)", "IMU\n(角速度·加速度)", "轮式里程计\n(里程·编码器)"];
const sw = (C_W - 0.6) / 3;
sensors.forEach((t, i) => {
  const x = C_X + i * (sw + 0.3);
  card(s2, x, y0, sw, h0, "DDE6F0", { line: NAVY, lw: 1, r: 0.07 });
  txt(s2, t, x, y0, sw, h0, { fs: 11.5, color: NAVY, bold: true, lsm: 1.0 });
});

// ---- Row 1: Passive LiDAR-Inertial SLAM base ----
let y1 = 1.86, h1 = 1.28;
layerLabel(s2, y1, h1, "被动\nLiDAR-惯性\nSLAM 底座", TEAL);
card(s2, C_X, y1, C_W, h1, CARD, { line: TEAL, lw: 1.25, r: 0.07, shadow: true });
const base = [
  ["点云预处理", "运动去畸变\n体素降采样"],
  ["紧耦合前端里程计", "iEKF·流形误差状态\nFAST-LIO2 直接配准"],
  ["增量地图管理", "ikd-Tree\nO(log n) 增删/kNN"],
  ["回环 & 位姿图优化", "回环检测\n全局 BA / PGO"],
  ["全局地图输出", "稠密点云地图\n+ 实时位姿 T∈SE(3)"],
];
const bw = (C_W - 0.5 - base.length * 0.0) / base.length - 0.18;
const bgap = 0.22;
const bStep = bw + bgap;
let bx0 = C_X + 0.18;
base.forEach((b, i) => {
  const x = bx0 + i * bStep;
  card(s2, x, y1 + 0.16, bw, h1 - 0.32, "E8F1F4", { line: TEAL, lw: 1, r: 0.06 });
  txt(s2, b[0], x, y1 + 0.24, bw, 0.42, { fs: 11.5, color: TEAL_D, bold: true, lsm: 1.0 });
  txt(s2, b[1], x, y1 + 0.62, bw, 0.5, { fs: 9.5, color: MUTE, lsm: 1.0 });
  if (i < base.length - 1) rightArrow(s2, x + bw + 0.015, y1 + h1 / 2, bgap - 0.05, TEAL);
});

// ---- Row 2: Degradation / localizability quantification ----
let y2 = 3.36, h2 = 1.06;
layerLabel(s2, y2, h2, "退化感知\n量化层", CYAN);
card(s2, C_X, y2, C_W, h2, CARD, { line: CYAN, lw: 1.25, r: 0.07, shadow: true });
const deg = [
  ["Hessian 旋转/平移分块", "A′ 分块独立 SVD\n消除尺度差异"],
  ["可定位性特征空间分析", "力矩类比·信息对贡献\n投影到特征空间"],
  ["6-DoF 可定位性状态 η", "η∈{none, partial, full}\n三级细粒度判决"],
  ["不确定性度量输出", "地图熵 H(m) · 位姿协方差 Σ\nD-opt |Σ| · 局部几何"],
];
const dw = (C_W - 0.36) / 4 - 0.2;
const dgap = 0.24, dStep = dw + dgap;
let dx0 = C_X + 0.18;
deg.forEach((d, i) => {
  const x = dx0 + i * dStep;
  card(s2, x, y2 + 0.14, dw, h2 - 0.28, "E3F4F1", { line: CYAN, lw: 1, r: 0.06 });
  txt(s2, d[0], x, y2 + 0.2, dw, 0.42, { fs: 11, color: "0E5C52", bold: true, lsm: 1.0 });
  txt(s2, d[1], x, y2 + 0.58, dw, 0.42, { fs: 9, color: MUTE, lsm: 1.0 });
  if (i < deg.length - 1) rightArrow(s2, x + dw + 0.02, y2 + h2 / 2, dgap - 0.06, CYAN);
});

// ---- Row 3: Embodied active decision (FQL) ----
let y3 = 4.64, h3 = 1.42;
layerLabel(s2, y3, h3, "具身主动\n决策层\n(FQL)", AMBER_D);
card(s2, C_X, y3, C_W, h3, "FFF6E8", { line: AMBER, lw: 1.5, r: 0.07, shadow: true });
// left: candidate + state
const colA_w = 2.5;
card(s2, C_X + 0.18, y3 + 0.16, colA_w, h3 - 0.32, CARD, { line: AMBER, lw: 1, r: 0.06 });
txt(s2, "候选生成 & 状态编码", C_X + 0.18, y3 + 0.24, colA_w, 0.36, { fs: 11, color: AMBER_D, bold: true });
txt(s2, "• 候选：前沿点 + 潜在回环点\n• 状态 sₜ = [H(m), Σ, η, 局部几何]",
  C_X + 0.30, y3 + 0.6, colA_w - 0.3, 0.7, { fs: 9.5, color: INK, align: "left", lsm: 1.15 });
// middle: FQL twin-policy
const colB_x = C_X + 0.18 + colA_w + 0.28;
const colB_w = 4.6;
card(s2, colB_x, y3 + 0.16, colB_w, h3 - 0.32, CARD, { line: AMBER, lw: 1, r: 0.06 });
txt(s2, "FQL 双策略架构", colB_x, y3 + 0.24, colB_w, 0.34, { fs: 11, color: AMBER_D, bold: true });
// two sub boxes
card(s2, colB_x + 0.2, y3 + 0.62, (colB_w - 0.6) / 2, 0.62, "FBE7C6", { line: AMBER, lw: 0.75, r: 0.05 });
txt(s2, "BC Flow 表达性策略\nμᵝ(s,z) 多模态行为先验", colB_x + 0.2, y3 + 0.62, (colB_w - 0.6) / 2, 0.62, { fs: 9, color: INK, lsm: 1.05 });
card(s2, colB_x + 0.2 + (colB_w - 0.6) / 2 + 0.2, y3 + 0.62, (colB_w - 0.6) / 2, 0.62, "FBE7C6", { line: AMBER, lw: 0.75, r: 0.05 });
txt(s2, "一步策略 μ(s,z)\n最大化 Q + 蒸馏正则 (无 BPTT)", colB_x + 0.2 + (colB_w - 0.6) / 2 + 0.2, y3 + 0.62, (colB_w - 0.6) / 2, 0.62, { fs: 9, color: INK, lsm: 1.05 });
// right: reward
const colC_x = colB_x + colB_w + 0.28;
const colC_w = C_R - colC_x - 0.18;
card(s2, colC_x, y3 + 0.16, colC_w, h3 - 0.32, CARD, { line: AMBER, lw: 1, r: 0.06 });
txt(s2, "退化感知奖励 r", colC_x, y3 + 0.24, colC_w, 0.34, { fs: 11, color: AMBER_D, bold: true });
txt(s2, "r = 信息增益 (D-opt/MI)\n  − λ₁·可定位性风险(η)\n  − λ₂·运动代价",
  colC_x + 0.18, y3 + 0.6, colC_w - 0.3, 0.72, { fs: 9.5, color: INK, align: "left", lsm: 1.2 });
// inner arrows
rightArrow(s2, C_X + 0.18 + colA_w + 0.02, y3 + h3 / 2, 0.22, AMBER_D);
rightArrow(s2, colB_x + colB_w + 0.02, y3 + h3 / 2, 0.22, AMBER_D);

// ---- Row 4: Localizability-constrained adaptive navigation ----
let y4 = 6.28, h4 = 0.92;
layerLabel(s2, y4, h4, "可定位性\n约束导航", ORANGE);
card(s2, C_X, y4, C_W, h4, CARD, { line: ORANGE, lw: 1.25, r: 0.07, shadow: true });
const nav = [
  ["退化方向约束局部规划", "Lagrangian 约束 / 退化方向锁定"],
  ["主动回环触发", "趋向强可定位性区重访降漂移"],
  ["自适应行为切换", "η=none→寻几何特征 / full→自由探索"],
  ["运动控制 → 机器人执行", "MPC / 轨迹跟踪 → 平台运动"],
];
const nw = (C_W - 0.36) / 4 - 0.2;
const ngap = 0.24, nStep = nw + ngap;
let nx0 = C_X + 0.18;
nav.forEach((n, i) => {
  const x = nx0 + i * nStep;
  card(s2, x, y4 + 0.12, nw, h4 - 0.24, "FCEAE3", { line: ORANGE, lw: 1, r: 0.06 });
  txt(s2, n[0], x, y4 + 0.16, nw, 0.36, { fs: 10.5, color: "9C3B23", bold: true, lsm: 1.0 });
  txt(s2, n[1], x, y4 + 0.5, nw, 0.34, { fs: 8.5, color: MUTE, lsm: 1.0 });
  if (i < nav.length - 1) rightArrow(s2, x + nw + 0.02, y4 + h4 / 2, ngap - 0.06, ORANGE);
});

// ---- vertical down arrows between layers (centered) ----
const midX = C_X + C_W / 2;
downArrow(s2, midX, y0 + h0 + 0.0, y1 - (y0 + h0), NAVY);
downArrow(s2, midX, y1 + h1 + 0.0, y2 - (y1 + h1), TEAL);
downArrow(s2, midX, y2 + h2 + 0.0, y3 - (y2 + h2), CYAN);
downArrow(s2, midX, y3 + h3 + 0.0, y4 - (y3 + h3), AMBER_D);

// ---- feedback loop (right side, bottom -> top) ----
s2.addShape(pres.shapes.LINE, { x: FB_X, y: y4 + h4 / 2, w: 0.9, h: 0, line: { color: ORANGE, width: 2.5 } });
s2.addShape(pres.shapes.LINE, { x: FB_X + 0.9, y: y0 + h0 / 2, w: 0, h: (y4 + h4 / 2) - (y0 + h0 / 2), line: { color: ORANGE, width: 2.5 } });
s2.addShape(pres.shapes.LINE, { x: FB_X, y: y0 + h0 / 2, w: 0.9, h: 0, line: { color: ORANGE, width: 2.5, endArrowType: "triangle" } });
txt(s2, "闭环反馈\n执行→环境→新观测", FB_X - 0.15, (y2 + y3) / 2 - 0.2, 1.25, 0.7, { fs: 9, color: "9C3B23", bold: true, lsm: 1.05 });

/* =======================================================================
 * SLIDE 3 — Research-content logic chain
 * ===================================================================== */
let s3 = pres.addSlide();
s3.background = { color: LIGHT };
txt(s3, "研究内容逻辑链：科学问题 → 研究内容 → 关键技术 → 创新点", 0.4, 0.18, 12.5, 0.55,
  { fs: 23, color: NAVY, bold: true, align: "left", font: HFONT });

// top: overarching scientific question
card(s3, 0.4, 0.9, 12.5, 0.72, NAVY, { r: 0.08 });
txt(s3, "核心科学问题：在几何退化、GPS 拒止的地下空间中，机器人如何 “主动运动” 以同时保障感知可靠性、最大化地图信息增益并实现自适应导航？",
  0.6, 0.9, 12.1, 0.72, { fs: 13.5, color: "FFFFFF", bold: true, align: "left", lsm: 1.1 });

const cols = [
  {
    c: TEAL, t: "研究内容一",
    h: "退化环境建图底座与可定位性在线量化",
    q: "科学问题：自相似几何下感知可靠性如何被实时、无需调参地表征？",
    k: ["FAST-LIO2/Faster-LIO 紧耦合里程计 (iEKF + ikd-Tree)", "X-ICP 式 Hessian 特征空间分解", "6-DoF 三级可定位性状态 η 实时估计"],
    n: "创新：把“可定位性”从离线诊断升级为在线、可微的决策状态量",
  },
  {
    c: AMBER_D, t: "研究内容二",
    h: "退化感知驱动的具身主动探索决策机理",
    q: "科学问题：探索–利用–可定位性三元权衡下的最优决策机理是什么？",
    k: ["ρ-POMDP 主动感知建模", "FQL 双策略：BC Flow 多模态先验 + 一步 Q 最大化", "退化感知奖励 r=信息增益−可定位性风险−代价"],
    n: "创新：用表达性离线 RL 策略统一刻画前沿/回环/避退化的多峰决策",
  },
  {
    c: ORANGE, t: "研究内容三",
    h: "可定位性约束下的自适应导航与闭环验证",
    q: "科学问题：感知-决策-导航闭环如何随退化状态在线自适应？",
    k: ["退化方向约束局部规划 + 主动回环触发", "η 驱动的行为切换 (offline→online 适配)", "矿井/隧道仿真 + 实物闭环验证"],
    n: "创新：可定位性约束贯通“估计-规划-控制”全链路的自适应闭环",
  },
];
const cw = 4.0, gap = 0.27, x0 = 0.4, cy = 1.78, ch = 5.4;
cols.forEach((col, i) => {
  const x = x0 + i * (cw + gap);
  card(s3, x, cy, cw, ch, CARD, { line: col.c, lw: 1.25, r: 0.09, shadow: true });
  s3.addShape(pres.shapes.RECTANGLE, { x, y: cy, w: cw, h: 0.6, fill: { color: col.c } });
  txt(s3, col.t, x + 0.2, cy, cw - 0.4, 0.6, { fs: 15, color: "FFFFFF", bold: true, align: "left" });
  txt(s3, col.h, x + 0.22, cy + 0.7, cw - 0.44, 0.7, { fs: 13, color: INK, bold: true, align: "left", lsm: 1.08 });
  txt(s3, col.q, x + 0.22, cy + 1.42, cw - 0.44, 0.78, { fs: 10.5, color: col.c, italic: true, align: "left", lsm: 1.12 });
  // key tech header
  txt(s3, "关键技术", x + 0.22, cy + 2.28, cw - 0.44, 0.3, { fs: 11, color: MUTE, bold: true, align: "left" });
  s3.addText(col.k.map((t, j) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 6 } })),
    { x: x + 0.24, y: cy + 2.6, w: cw - 0.46, h: 1.7, fontFace: BFONT, fontSize: 10, color: INK, align: "left", valign: "top", lineSpacingMultiple: 1.05 });
  // innovation footer
  card(s3, x + 0.18, cy + ch - 0.92, cw - 0.36, 0.78, "F0F4F8", { line: col.c, lw: 0.75, r: 0.06 });
  txt(s3, col.n, x + 0.3, cy + ch - 0.92, cw - 0.58, 0.78, { fs: 9.5, color: col.c, bold: true, align: "left", lsm: 1.12 });
});

pres.writeFile({ fileName: "D:/Desktop/Obsidian文件/个人笔记/temp/研究内容三_方法框图.pptx" })
  .then(f => console.log("WROTE: " + f))
  .catch(e => { console.error(e); process.exit(1); });
