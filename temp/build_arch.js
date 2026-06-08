const pptxgen = require("pptxgenjs");

// ---------- Palette ----------
const NAVY = "0E2A47", TEAL = "1C7293", TEAL_D = "0F4C5C", CYAN = "2A9D8F",
  AMBER = "E9A23B", AMBER_D = "B9791A", ORANGE = "E76F51", ORANGE_D = "B5462C",
  LIGHT = "F3F6FA", CARD = "FFFFFF", INK = "1A2230", MUTE = "5B6B7C",
  HFONT = "Microsoft YaHei", BFONT = "Microsoft YaHei";

let pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
pres.author = "Claude";
pres.title = "面向地下退化环境的具身主动感知与自适应导航机理 — 总体框架图";

const W = 13.333;
const mkShadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 2, angle: 90, opacity: 0.14 });

function rrect(s, x, y, w, h, fill, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: o.r ?? 0.06,
    fill: fill === null ? { type: "none" } : { color: fill },
    line: o.line ? { color: o.line, width: o.lw || 1 } : { type: "none" },
    shadow: o.shadow ? mkShadow() : undefined,
  });
}
function rect(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { type: "none" } });
}
function txt(s, t, x, y, w, h, o = {}) {
  s.addText(t, {
    x, y, w, h, margin: o.margin ?? 2, fontFace: o.font || BFONT, fontSize: o.fs || 11,
    color: o.color || INK, bold: !!o.bold, italic: !!o.italic, align: o.align || "center",
    valign: o.valign || "middle", lineSpacingMultiple: o.lsm || 1.0, breakLine: o.breakLine,
  });
}
function dArrow(s, x, y, h, c) { s.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color: c, width: 2.25, endArrowType: "triangle" } }); }
function rArrow(s, x, y, w, c) { s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: c, width: 1.75, endArrowType: "triangle" } }); }

let s = pres.addSlide();
s.background = { color: LIGHT };

// ===== Title =====
rect(s, 0, 0, W, 0.78, NAVY);
rect(s, 0, 0, 0.16, 0.78, AMBER);
txt(s, "面向地下退化环境的具身主动感知与自适应导航机理 · 总体研究框架图",
  0.35, 0.04, 9.6, 0.7, { fs: 19, color: "FFFFFF", bold: true, align: "left", font: HFONT });
// legend (top-right): data flow vs closed loop
s.addShape(pres.shapes.LINE, { x: 10.15, y: 0.27, w: 0.5, h: 0, line: { color: "CCD6E0", width: 2, endArrowType: "triangle" } });
txt(s, "数据流 Data flow", 10.7, 0.13, 2.4, 0.3, { fs: 9.5, color: "DCE5EF", align: "left" });
s.addShape(pres.shapes.LINE, { x: 10.15, y: 0.57, w: 0.5, h: 0, line: { color: ORANGE, width: 2, endArrowType: "triangle" } });
txt(s, "闭环反馈 Closed-loop", 10.7, 0.43, 2.4, 0.3, { fs: 9.5, color: "F6CDBF", align: "left" });

// ===== Geometry =====
const BX = 0.4, BR = 11.95;             // band area (leave right gutter for loop)
const BW = BR - BX;
const FBX = 12.15;                      // feedback column
const HH = 0.34;                        // band header height
let cy = 0.98;                          // running y

// band header bar (bilingual) + content frame; returns content rect {x,y,w,h}
function band(cn, en, accent, contentH, badge) {
  // header
  rrect(s, BX, cy, BW, HH, accent, { r: 0.05 });
  rect(s, BX, cy, 0.14, HH, NAVY);
  txt(s, badge, BX + 0.02, cy, 0.5, HH, { fs: 13, color: "FFFFFF", bold: true });
  s.addText([
    { text: cn, options: { bold: true, fontSize: 12.5, color: "FFFFFF" } },
    { text: "   ·  " + en, options: { fontSize: 9.5, color: "FFFFFF", italic: true } },
  ], { x: BX + 0.5, y: cy, w: BW - 0.6, h: HH, align: "left", valign: "middle", margin: 1, fontFace: HFONT });
  const c = { x: BX, y: cy + HH + 0.06, w: BW, h: contentH };
  cy = c.y + contentH;       // advance
  return c;
}
function gap(g, accent) { dArrow(s, BX + BW / 2, cy + 0.005, g - 0.01, accent || MUTE); cy += g; }

// boxes-in-a-row helper
function rowBoxes(c, items, accent, tint, opts = {}) {
  const n = items.length, pad = 0.16, ga = opts.gap ?? 0.22;
  const bw = (c.w - pad * 2 - ga * (n - 1)) / n;
  items.forEach((it, i) => {
    const x = c.x + pad + i * (bw + ga);
    rrect(s, x, c.y, bw, c.h, tint, { line: accent, lw: 1, r: 0.05 });
    txt(s, it[0], x, c.y + 0.05, bw, opts.th ?? 0.34, { fs: opts.tfs ?? 11, color: opts.tcolor || accent, bold: true, lsm: 1.0 });
    if (it[1]) txt(s, it[1], x, c.y + (opts.th ?? 0.34) + 0.02, bw, c.h - (opts.th ?? 0.34) - 0.06, { fs: opts.sfs ?? 9, color: MUTE, lsm: 1.0 });
    if (opts.arrow && i < n - 1) rArrow(s, x + bw + 0.015, c.y + c.h / 2, ga - 0.04, accent);
  });
  return bw;
}

// ---- Band 1: Sensor Input ----
let c1 = band("感知输入", "Sensor Input", NAVY, 0.62, "1");
rowBoxes(c1, [
  ["3D LiDAR", "机械 / 固态 · 高密度点云"],
  ["IMU", "角速度 · 加速度 · 高频"],
  ["轮式里程计", "里程 · 编码器先验"],
], NAVY, "E2E9F2", { tcolor: NAVY, tfs: 12, sfs: 9.5 });
gap(0.14, NAVY);

// ---- Band 2: Passive SLAM backbone ----
let c2 = band("被动 LiDAR-惯性 SLAM 建图底座", "Passive LiDAR-Inertial SLAM Backbone", TEAL, 0.82, "2");
rowBoxes(c2, [
  ["点云预处理", "运动去畸变 · 体素降采样"],
  ["紧耦合前端里程计", "iEKF·流形误差状态\nFAST-LIO2 直接配准"],
  ["增量地图管理", "ikd-Tree · O(log n)\n增删 / kNN 查询"],
  ["回环 & 位姿图优化", "回环检测 · 全局 BA/PGO"],
  ["全局地图与位姿", "稠密点云地图\n+ 实时位姿 T∈SE(3)"],
], TEAL, "E6F1F3", { arrow: true, tcolor: TEAL_D, tfs: 11.5, sfs: 9 });
gap(0.14, TEAL);

// ---- Band 3: Localizability quantification ----
let c3 = band("退化感知量化", "Localizability Quantification", CYAN, 0.74, "3");
rowBoxes(c3, [
  ["Hessian 旋转/平移分块", "A′ 分块独立 SVD · 消除尺度差异"],
  ["可定位性特征空间分析", "力矩类比 · 信息对贡献投影"],
  ["6-DoF 可定位性状态 η", "η∈{none, partial, full} 三级判决"],
  ["不确定性度量输出", "地图熵 H(m) · 协方差 Σ · D-opt"],
], CYAN, "E1F4F0", { arrow: true, tcolor: "0E5C52", tfs: 11, sfs: 8.7 });
gap(0.14, CYAN);

// ---- Band 4: Active decision (FQL) ----
let c4 = band("具身主动探索决策 (核心)", "Embodied Active Exploration · Flow Q-Learning", AMBER_D, 1.0, "4");
{
  const pad = 0.16, ga = 0.24;
  const wA = 2.7, wC = 3.3;
  const wB = c4.w - pad * 2 - ga * 2 - wA - wC;
  // A candidate + state
  let xA = c4.x + pad;
  rrect(s, xA, c4.y, wA, c4.h, "FFF3E0", { line: AMBER, lw: 1, r: 0.05 });
  txt(s, "候选生成 & 状态编码", xA, c4.y + 0.06, wA, 0.3, { fs: 10.5, color: AMBER_D, bold: true });
  txt(s, "• 候选：前沿点 + 潜在回环点\n• 状态 sₜ = [ H(m), Σ, η, 局部几何 ]", xA + 0.14, c4.y + 0.36, wA - 0.26, c4.h - 0.4, { fs: 9.3, color: INK, align: "left", lsm: 1.18 });
  rArrow(s, xA + wA + 0.02, c4.y + c4.h / 2, ga - 0.05, AMBER_D);
  // B FQL twin policy
  let xB = xA + wA + ga;
  rrect(s, xB, c4.y, wB, c4.h, "FFF3E0", { line: AMBER, lw: 1.25, r: 0.05 });
  txt(s, "FQL 双策略架构（无 BPTT · 单步推理）", xB, c4.y + 0.06, wB, 0.3, { fs: 10.5, color: AMBER_D, bold: true });
  const sbw = (wB - 0.5) / 2;
  rrect(s, xB + 0.16, c4.y + 0.38, sbw, c4.h - 0.52, "FBE3BE", { line: AMBER, lw: 0.75, r: 0.05 });
  txt(s, "BC Flow 表达性策略 μᵝ(s,z)\n多模态行为先验（前沿/回环/避退化）", xB + 0.16, c4.y + 0.38, sbw, c4.h - 0.52, { fs: 8.8, color: INK, lsm: 1.12 });
  rrect(s, xB + 0.16 + sbw + 0.18, c4.y + 0.38, sbw, c4.h - 0.52, "FBE3BE", { line: AMBER, lw: 0.75, r: 0.05 });
  txt(s, "一步策略 μ(s,z)\n最大化 Q + 蒸馏正则", xB + 0.16 + sbw + 0.18, c4.y + 0.38, sbw, c4.h - 0.52, { fs: 8.8, color: INK, lsm: 1.12 });
  rArrow(s, xB + wB + 0.02, c4.y + c4.h / 2, ga - 0.05, AMBER_D);
  // C reward
  let xC = xB + wB + ga;
  rrect(s, xC, c4.y, wC, c4.h, "FFF3E0", { line: AMBER, lw: 1, r: 0.05 });
  txt(s, "退化感知奖励 r", xC, c4.y + 0.06, wC, 0.3, { fs: 10.5, color: AMBER_D, bold: true });
  s.addText([
    { text: "r = 信息增益 (D-opt / MI)", options: { breakLine: true, color: INK } },
    { text: "      − λ₁ · 可定位性风险(η)", options: { breakLine: true, color: ORANGE_D, bold: true } },
    { text: "      − λ₂ · 运动 / 能耗代价", options: { color: INK } },
  ], { x: xC + 0.16, y: c4.y + 0.36, w: wC - 0.28, h: c4.h - 0.42, align: "left", valign: "middle", fontFace: BFONT, fontSize: 9.6, lineSpacingMultiple: 1.18, margin: 1 });
}
gap(0.14, AMBER_D);

// ---- Band 5: Adaptive navigation ----
let c5 = band("可定位性约束自适应导航", "Localizability-Constrained Adaptive Navigation", ORANGE, 0.72, "5");
rowBoxes(c5, [
  ["退化方向约束局部规划", "Lagrangian 约束 · 退化方向锁定"],
  ["主动回环触发", "趋向强可定位性区重访降漂移"],
  ["自适应行为切换", "η=none→寻几何特征 / full→自由探索"],
  ["运动控制 → 平台执行", "MPC / 轨迹跟踪 → 机器人运动"],
], ORANGE, "FCEAE2", { arrow: true, tcolor: ORANGE_D, tfs: 10.5, sfs: 8.5 });

// ===== Feedback loop (right gutter, bottom -> top) =====
const yTop = 0.98 + HH / 2;                 // mid of band1 header
const yBot = c5.y + c5.h / 2;               // mid of band5 content
s.addShape(pres.shapes.LINE, { x: FBX, y: yBot, w: 0.85, h: 0, line: { color: ORANGE, width: 2.5 } });
s.addShape(pres.shapes.LINE, { x: FBX + 0.85, y: yTop, w: 0, h: yBot - yTop, line: { color: ORANGE, width: 2.5 } });
s.addShape(pres.shapes.LINE, { x: FBX, y: yTop, w: 0.85, h: 0, line: { color: ORANGE, width: 2.5, endArrowType: "triangle" } });
txt(s, "闭环反馈\n执行 → 环境\n→ 新观测", FBX - 0.05, (yTop + yBot) / 2 - 0.35, 1.15, 0.8, { fs: 9, color: ORANGE_D, bold: true, lsm: 1.12 });

pres.writeFile({ fileName: "D:/Desktop/Obsidian文件/个人笔记/temp/研究内容三_总体框架图.pptx" })
  .then(f => console.log("WROTE: " + f))
  .catch(e => { console.error(e); process.exit(1); });
