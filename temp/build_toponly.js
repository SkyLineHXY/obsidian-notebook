// 研究内容一 架构图（修订版）: 干净双层 + 4点修正
//  下层 被动LIO建图(L→R) → 右侧上行 → 上层 主动决策(R→L) → 左侧下行 闭环
//  FIX1 上层补"运动规划/轨迹生成→执行"模块
//  FIX2 状态表示统一为"图注意力+LSTM 编码 Gˢ"，η为节点特征
//  FIX3 标注"双向A*稀疏化→边缘算力可承受""单步推理→机载低功耗实时"
//  FIX4 显式画出"重访锚点(主动触发回环)"决策行为；输出补后验协方差 P̄
const pptxgen = require("pptxgenjs");

const NAVY="0E2A47", TEAL="1C7293", TEAL_D="0F4C5C", CYAN="2A9D8F",
  AMBER="E9A23B", AMBER_D="B9791A", ORANGE="E76F51", ORANGE_D="B5462C",
  GREEN="2E7D52", LIGHT="F3F6FA", INK="1A2230", MUTE="5B6B7C",
  HF="Microsoft YaHei", BF="Microsoft YaHei";
const PHOTO_DIR="D:/Desktop/Obsidian文件/个人笔记/temp/";  // downsized copies
const OUT="D:/Desktop/研究生/国自然_地下空间多机器人SLAM/rc1_toponly.pptx";

let pres=new pptxgen();
pres.defineLayout({name:"W",width:13.333,height:7.5}); pres.layout="W";
pres.author="Claude"; pres.title="研究内容一 总体框架图(修订版)";
const W=13.333;
const shadow=()=>({type:"outer",color:"000000",blur:5,offset:2,angle:90,opacity:0.13});

function rr(s,x,y,w,h,fill,o={}){s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h,rectRadius:o.r??0.05,
  fill:fill===null?{type:"none"}:{color:fill},line:o.line?{color:o.line,width:o.lw||1,dashType:o.dash?"dash":"solid"}:{type:"none"},
  shadow:o.shadow?shadow():undefined});}
function rect(s,x,y,w,h,fill){s.addShape(pres.shapes.RECTANGLE,{x,y,w,h,fill:{color:fill},line:{type:"none"}});}
function txt(s,t,x,y,w,h,o={}){s.addText(t,{x,y,w,h,margin:o.margin??2,fontFace:o.font||BF,fontSize:o.fs||11,
  color:o.color||INK,bold:!!o.bold,italic:!!o.italic,align:o.align||"center",valign:o.valign||"middle",
  lineSpacingMultiple:o.lsm||1.0});}
function aLeft(s,x,y,w,c){s.addShape(pres.shapes.LINE,{x,y,w,h:0,line:{color:c,width:1.9,beginArrowType:"triangle"}});}
function aRight(s,x,y,w,c){s.addShape(pres.shapes.LINE,{x,y,w,h:0,line:{color:c,width:1.9,endArrowType:"triangle"}});}

let s=pres.addSlide(); s.background={color:LIGHT};

// ===== Title =====
rect(s,0,0,W,0.72,NAVY); rect(s,0,0,0.16,0.72,AMBER);
txt(s,"面向地下退化环境的具身主动感知与自适应导航机理 · 总体研究框架图",0.34,0.04,9.3,0.64,
  {fs:18,color:"FFFFFF",bold:true,align:"left",font:HF});
s.addShape(pres.shapes.LINE,{x:10.05,y:0.26,w:0.5,h:0,line:{color:"CCD6E0",width:2,endArrowType:"triangle"}});
txt(s,"数据流 Data flow",10.6,0.12,2.5,0.28,{fs:9,color:"DCE5EF",align:"left"});
s.addShape(pres.shapes.LINE,{x:10.05,y:0.55,w:0.5,h:0,line:{color:ORANGE,width:2,endArrowType:"triangle"}});
txt(s,"闭环演化反馈 Closed-loop",10.6,0.41,2.6,0.28,{fs:9,color:"F6CDBF",align:"left"});

// ===== Geometry =====
const BX=0.62, BW=12.10;            // band x/width
const LG=0.30, RG=12.90;            // left/right gutter x-centers for loop wires
const HH=0.36;

// ---------------- TOP band header (主动决策) ----------------
const tHY=0.82;
rr(s,BX,tHY,BW,HH,AMBER_D,{r:0.04});
rect(s,BX,tHY,0.5,HH,NAVY); txt(s,"上层",BX,tHY,0.5,HH,{fs:11,color:"FFFFFF",bold:true});
s.addText([{text:"具身主动探索决策（核心）",options:{bold:true,fontSize:12.5,color:"FFFFFF"}},
  {text:"   ·  Embodied Active Exploration · Flow Q-Learning",options:{fontSize:9.5,color:"FFFFFF",italic:true}}],
  {x:BX+0.55,y:tHY,w:BW-0.65,h:HH,align:"left",valign:"middle",margin:1,fontFace:HF});

// TOP content boxes — placed canvas L→R, logical flow R→L
const tCY=1.28, tCH=2.06;
const pad=0.16, ga=0.10;
// widths (L→R): 执行 / 运动规划 / FQL核心 / 状态编码 / 稀疏信息图
const wEx=1.55, wPl=2.05, wFq=3.55, wEn=2.18, wGr=2.07;
let x=BX+pad;
const xEx=x; x+=wEx+ga;
const xPl=x; x+=wPl+ga;
const xFq=x; x+=wFq+ga;
const xEn=x; x+=wEn+ga;
const xGr=x;

// (rightmost) 稀疏信息图构建 𝒢ˢ  — contains 双向A*稀疏化 + FIX3a
rr(s,xGr,tCY,wGr,tCH,"FFF3E0",{line:AMBER,lw:1.1,r:0.05,shadow:true});
txt(s,"稀疏信息图构建 𝒢ˢ",xGr,tCY+0.05,wGr,0.3,{fs:10.5,color:AMBER_D,bold:true});
txt(s,"互补孔洞栅格 → 2D 占据图\n→ 双向 A* 稀疏化\n节点特征注入 η · 候选探索点",xGr+0.12,tCY+0.36,wGr-0.22,tCH-0.78,
  {fs:9,color:INK,align:"left",lsm:1.12});
rr(s,xGr+0.12,tCY+tCH-0.34,wGr-0.24,0.28,"DBEAD8",{line:GREEN,lw:0.75,r:0.06}); // FIX3a
txt(s,"降维：节点数↓ ⇒ 边缘算力可承受",xGr+0.12,tCY+tCH-0.34,wGr-0.24,0.28,{fs:8,color:GREEN,bold:true});

// 图注意力+LSTM 状态编码 (FIX2)
rr(s,xEn,tCY,wEn,tCH,"FFF3E0",{line:AMBER,lw:1.1,r:0.05,shadow:true});
txt(s,"图注意力+LSTM 状态编码",xEn,tCY+0.05,wEn,0.3,{fs:10.5,color:AMBER_D,bold:true});
txt(s,"zₜ = Enc(𝒢ˢ)\n节点特征(node feat.):\nH(m) · Σ · η · 局部几何",xEn+0.12,tCY+0.36,wEn-0.22,tCH-0.5,
  {fs:9,color:INK,align:"left",lsm:1.16});

// FQL 双策略核心 + 奖励 (FIX3b)
rr(s,xFq,tCY,wFq,tCH,"FFF3E0",{line:AMBER,lw:1.3,r:0.05,shadow:true});
txt(s,"FQL 双策略架构（无 BPTT · 单步推理）",xFq,tCY+0.05,wFq,0.28,{fs:10.5,color:AMBER_D,bold:true});
const sbw=(wFq-0.46)/2, sby=tCY+0.36, sbh=0.74;
rr(s,xFq+0.15,sby,sbw,sbh,"FBE3BE",{line:AMBER,lw:0.75,r:0.05});
txt(s,"BC Flow 表达性策略 μᵝ(z,w)\n多模态行为先验",xFq+0.15,sby,sbw,sbh,{fs:8.6,color:INK,lsm:1.12});
rr(s,xFq+0.15+sbw+0.16,sby,sbw,sbh,"FBE3BE",{line:AMBER,lw:0.75,r:0.05});
txt(s,"一步策略 μ(z,w)\n最大化 Q + 蒸馏正则",xFq+0.15+sbw+0.16,sby,sbw,sbh,{fs:8.6,color:INK,lsm:1.12});
// FIX3b strip
rr(s,xFq+0.15,sby+sbh+0.05,wFq-0.3,0.26,"DBEAD8",{line:GREEN,lw:0.75,r:0.06});
txt(s,"单步推理 ⇒ 机载低功耗实时 · 支持 offline→online",xFq+0.15,sby+sbh+0.05,wFq-0.3,0.26,{fs:8,color:GREEN,bold:true});
// reward strip
rr(s,xFq+0.15,sby+sbh+0.36,wFq-0.3,0.34,"FCEAE2",{line:ORANGE,lw:0.9,r:0.06});
s.addText([{text:"奖励 r = 信息增益(D-opt/MI) ",options:{color:INK}},
  {text:"− λ₁·可定位性风险(η)",options:{color:ORANGE_D,bold:true}},
  {text:" − λ₂·运动/能耗代价",options:{color:INK}}],
  {x:xFq+0.15,y:sby+sbh+0.36,w:wFq-0.3,h:0.34,align:"center",valign:"middle",fontFace:BF,fontSize:8.4,margin:1});

// 运动规划/轨迹生成 (FIX1) + FIX4 behavior tag
rr(s,xPl,tCY,wPl,tCH,"FFF3E0",{line:AMBER,lw:1.1,r:0.05,shadow:true});
txt(s,"运动规划 / 轨迹生成",xPl,tCY+0.05,wPl,0.3,{fs:10.5,color:AMBER_D,bold:true});
txt(s,"退化方向等式约束\n锁定不可观自由度\n生成主动寻优轨迹",xPl+0.12,tCY+0.36,wPl-0.22,tCH-0.78,
  {fs:9,color:INK,align:"left",lsm:1.12});
rr(s,xPl+0.12,tCY+tCH-0.34,wPl-0.24,0.28,"E7DDF2",{line:"6B4FA3",lw:0.75,r:0.06}); // FIX4
txt(s,"行为:探索前沿/重访锚点/规避退化",xPl+0.12,tCY+tCH-0.34,wPl-0.24,0.28,{fs:7.6,color:"5A3E96",bold:true});

// 执行 (leftmost)
rr(s,xEx,tCY,wEx,tCH,"FFE7D8",{line:ORANGE,lw:1.1,r:0.05,shadow:true});
txt(s,"执行\n(MPC/轨迹跟踪)",xEx,tCY+0.05,wEx,0.7,{fs:10.5,color:ORANGE_D,bold:true,lsm:1.05});
txt(s,"目标点 / 速度指令\n→ 平台运动",xEx+0.1,tCY+0.86,wEx-0.2,tCH-0.95,{fs:8.6,color:INK,lsm:1.12});

// TOP arrows: flow R→L (left-pointing)
const tMid=tCY+tCH/2;
aLeft(s,xEx+wEx+0.01,tMid,ga-0.02,AMBER_D);
aLeft(s,xPl+wPl+0.01,tMid,ga-0.02,AMBER_D);
aLeft(s,xFq+wFq+0.01,tMid,ga-0.02,AMBER_D);
aLeft(s,xEn+wEn+0.01,tMid,ga-0.02,AMBER_D);


pres.writeFile({fileName:OUT}).then(f=>console.log("WROTE:"+f)).catch(e=>{console.error(e);process.exit(1);});
