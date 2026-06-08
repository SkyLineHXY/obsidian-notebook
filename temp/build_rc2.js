const pptxgen = require("pptxgenjs");

const NAVY="0E2A47", NAVY2="20486E", TEAL="1C7293", TEAL_D="0F4C5C", CYAN="2A9D8F",
  AMBER="E9A23B", AMBER_D="B9791A", AMBER_DD="8F5E10", ORANGE="E76F51", ORANGE_D="B5462C",
  SLATE="55667A", LIGHT="F4F7FA", CARD="FFFFFF", INK="1A2230", MUTE="5E6E7E",
  HFONT="Microsoft YaHei", BFONT="Microsoft YaHei";

let pres=new pptxgen();
pres.defineLayout({name:"W",width:13.333,height:7.5});
pres.layout="W";
pres.title="研究内容2 FQL 细节框架图";
const W=13.333;
const mkShadow=()=>({type:"outer",color:"000000",blur:5,offset:2,angle:90,opacity:0.13});

function rrect(s,x,y,w,h,fill,o={}){s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h,rectRadius:o.r??0.06,fill:fill===null?{type:"none"}:{color:fill},line:o.line?{color:o.line,width:o.lw||1,dashType:o.dash||"solid"}:{type:"none"},shadow:o.shadow?mkShadow():undefined});}
function rect(s,x,y,w,h,fill){s.addShape(pres.shapes.RECTANGLE,{x,y,w,h,fill:{color:fill},line:{type:"none"}});}
function txt(s,t,x,y,w,h,o={}){s.addText(t,{x,y,w,h,margin:o.margin??2,fontFace:o.font||BFONT,fontSize:o.fs||10,color:o.color||INK,bold:!!o.bold,italic:!!o.italic,align:o.align||"center",valign:o.valign||"middle",lineSpacingMultiple:o.lsm||1.0});}
function dArrow(s,x,y,h,c,dash){s.addShape(pres.shapes.LINE,{x,y,w:0,h,line:{color:c,width:2,endArrowType:"triangle",dashType:dash||"solid"}});}
function rArrow(s,x,y,w,c,dash){s.addShape(pres.shapes.LINE,{x,y,w,h:0,line:{color:c,width:2,endArrowType:"triangle",dashType:dash||"solid"}});}
function lArrow(s,x,y,w,c,dash){s.addShape(pres.shapes.LINE,{x,y,w,h:0,line:{color:c,width:2,beginArrowType:"triangle",dashType:dash||"solid"}});}
function upArrow(s,x,yTop,h,c,dash){s.addShape(pres.shapes.LINE,{x,y:yTop,w:0,h,line:{color:c,width:2,beginArrowType:"triangle",dashType:dash||"solid"}});}
// dashed group box with centered colored title
function group(s,x,y,w,h,color,title,o={}){
  rrect(s,x,y,w,h,o.fill||null,{line:color,lw:o.lw||1.25,dash:o.dash||"dash",r:0.08});
  txt(s,title,x+0.1,y+0.05,w-0.2,0.3,{fs:o.tfs||11,color:color,bold:true});
}

let s=pres.addSlide();
s.background={color:LIGHT};

// ===== Title =====
rect(s,0,0,W,0.62,NAVY); rect(s,0,0,0.16,0.62,AMBER);
txt(s,"研究内容 2 · 退化感知驱动的具身主动探索决策（Flow Q-Learning）细节框架",0.34,0.02,12.6,0.58,{fs:17,color:"FFFFFF",bold:true,align:"left",font:HFONT});

// ===== columns geometry =====
const GY=0.78, GH=4.5;                  // main group top / height
const c1={x:0.3,w:2.12}, c2={x:2.72,w:2.42}, c3={x:5.42,w:4.62}, c4={x:10.32,w:2.71};

// ---- Col1: perception outputs (from RC1) ----
group(s,c1.x,GY,c1.w,GH,TEAL,"感知量化输出 (研究内容1)",{fill:"EAF3F5"});
{
  const items=[
    ["全局地图 & 位姿","T ∈ SE(3) · ikd-Tree"],
    ["地图熵 H(m)","位姿协方差 Σ"],
    ["6-DoF 可定位性 η","{none, partial, full}"],
    ["局部几何特征","退化方向 / 法向分布"],
  ];
  const bx=c1.x+0.14, bw=c1.w-0.28, top=GY+0.38, bh=0.88, gap=0.16;
  items.forEach((it,i)=>{const yy=top+i*(bh+gap);rrect(s,bx,yy,bw,bh,CARD,{line:TEAL,lw:1,r:0.05});txt(s,it[0],bx,yy+0.08,bw,0.4,{fs:10.5,color:TEAL_D,bold:true,lsm:1.0});txt(s,it[1],bx,yy+0.46,bw,0.36,{fs:8.5,color:MUTE,lsm:1.0});});
}
rArrow(s,c1.x+c1.w+0.02,GY+GH/2,c2.x-(c1.x+c1.w)-0.04,TEAL);

// ---- Col2: candidate generation + state encoder ----
group(s,c2.x,GY,c2.w,GH,NAVY2,"候选生成 & 状态编码",{fill:"EAEFF6"});
{
  const bx=c2.x+0.16, bw=c2.w-0.32;
  // two candidate boxes
  rrect(s,bx,GY+0.42,bw,0.78,CARD,{line:NAVY2,lw:1,r:0.05});
  txt(s,"前沿点检测",bx,GY+0.5,bw,0.36,{fs:10.5,color:NAVY2,bold:true});
  txt(s,"探索候选 (未知边界)",bx,GY+0.86,bw,0.3,{fs:8.5,color:MUTE});
  rrect(s,bx,GY+1.34,bw,0.78,CARD,{line:NAVY2,lw:1,r:0.05});
  txt(s,"潜在回环点",bx,GY+1.42,bw,0.36,{fs:10.5,color:NAVY2,bold:true});
  txt(s,"利用候选 (重访降漂移)",bx,GY+1.78,bw,0.3,{fs:8.5,color:MUTE});
  // merge -> action set
  dArrow(s,c2.x+c2.w/2,GY+2.14,0.18,NAVY2);
  rrect(s,bx,GY+2.36,bw,0.62,"E3E9F3",{line:NAVY2,lw:1,r:0.05});
  txt(s,"候选目标集 = 动作空间 𝒜ₜ",bx,GY+2.36,bw,0.62,{fs:9.5,color:NAVY,bold:true,lsm:1.0});
  // state encoder
  rrect(s,bx,GY+3.16,bw,1.12,CARD,{line:NAVY2,lw:1.25,r:0.05});
  txt(s,"状态编码器  (MLP / Transformer)",bx,GY+3.24,bw,0.34,{fs:9.5,color:NAVY2,bold:true});
  txt(s,"状态嵌入\nsₜ = [ H(m), Σ, η, 局部几何 ]",bx,GY+3.6,bw,0.62,{fs:9.5,color:INK,lsm:1.12});
}
// arrow col2 -> col3 (state)
rArrow(s,c2.x+c2.w+0.02,GY+GH/2,c3.x-(c2.x+c2.w)-0.04,NAVY2);
txt(s,"sₜ",c2.x+c2.w+0.0,GY+GH/2-0.34,c3.x-(c2.x+c2.w),0.26,{fs:9,color:NAVY,bold:true});

// ---- Col3: FQL twin-policy core (solid, highlighted) ----
rrect(s,c3.x,GY,c3.w,GH,"FFF7E9",{line:AMBER,lw:2,r:0.08,shadow:true});
txt(s,"FQL 双策略决策核心   (无 BPTT · 部署单步推理)",c3.x+0.1,GY+0.05,c3.w-0.2,0.32,{fs:12,color:AMBER_DD,bold:true});
{
  const ix=c3.x+0.18, iw=c3.w-0.36;
  // Row A: BC Flow expressive policy (offline)
  const ay=GY+0.42, ah=1.04;
  rrect(s,ix,ay,iw,ah,"FFFBF2",{line:AMBER,lw:1,dash:"dash",r:0.05});
  txt(s,"① BC Flow 表达性策略 μᵝ(s,z)   [仅离线行为克隆，刻画多模态行为分布]",ix+0.06,ay+0.04,iw-0.12,0.26,{fs:9.5,color:AMBER_D,bold:true,align:"left"});
  const aby=ay+0.36, abh=0.56;
  const fw=[0.98,1.66,0.98]; let fx=ix+0.16;
  const fb=[["z ~ 𝒩(0, I)","噪声采样"],["Flow ODE 积分 (T 步)","a⁽⁰⁾→a⁽¹⁾→…→a⁽ᵀ⁾"],["μᵝ 多模态\n行为先验"]];
  fb.forEach((b,i)=>{rrect(s,fx,aby,fw[i],abh,CARD,{line:AMBER,lw:1,r:0.05});txt(s,b[0],fx,aby+(b[1]?0.04:0),fw[i],b[1]?0.32:abh,{fs:8.5,color:INK,bold:i===1,lsm:1.0});if(b[1])txt(s,b[1],fx,aby+0.3,fw[i],0.24,{fs:7.6,color:MUTE});if(i<2)rArrow(s,fx+fw[i]+0.01,aby+abh/2,0.15,AMBER_D);fx+=fw[i]+0.16;});

  // Row B: one-step policy
  const by=ay+ah+0.12, bh=0.86;
  rrect(s,ix,by,iw,bh,"FFFBF2",{line:AMBER,lw:1,r:0.05});
  txt(s,"② 一步策略 μ(s,z)   [承担 Q 最大化，部署时单步生成动作]",ix+0.06,by+0.03,iw-0.12,0.24,{fs:9.5,color:AMBER_D,bold:true,align:"left"});
  const bby=by+0.32, bbh=0.46; let bx2=ix+0.16;
  const bbw=[1.2,1.3,1.12];
  const bb=[["sₜ ⊕ z","状态+噪声"],["单步策略网络","one-step"],["候选动作 a","目标选择 ∈ 𝒜ₜ"]];
  bb.forEach((b,i)=>{rrect(s,bx2,bby,bbw[i],bbh,CARD,{line:AMBER,lw:1,r:0.05});txt(s,b[0],bx2,bby+0.02,bbw[i],0.26,{fs:8.5,color:INK,bold:true,lsm:1.0});txt(s,b[1],bx2,bby+0.26,bbw[i],0.18,{fs:7.4,color:MUTE});if(i<2)rArrow(s,bx2+bbw[i]+0.01,bby+bbh/2,0.15,AMBER_D);bx2+=bbw[i]+0.16;});
  // distillation arrow from Row A (mu^beta) down into Row B action
  s.addShape(pres.shapes.LINE,{x:ix+iw-0.55,y:aby+abh,w:0,h:by-(aby+abh),line:{color:CYAN,width:2,endArrowType:"triangle",dashType:"dash"}});
  txt(s,"蒸馏 α‖μ−μᵝ‖²",ix+iw-1.9,aby+abh+0.0,1.35,0.22,{fs:8,color:"0E5C52",bold:true,align:"right"});

  // Row C: critic
  const cy2=by+bh+0.12, ch=0.78;
  rrect(s,ix,cy2,iw,ch,"FFFBF2",{line:AMBER,lw:1,r:0.05});
  txt(s,"③ 评论家 Critic  Q_φ(s, a)   [TD 学习，提供价值信号]",ix+0.06,cy2+0.03,iw-0.12,0.24,{fs:9.5,color:AMBER_D,bold:true,align:"left"});
  rrect(s,ix+0.18,cy2+0.3,2.6,0.4,CARD,{line:AMBER,lw:1,r:0.05});
  txt(s,"L_Q = ( Q − r − γ Q̄′ )²",ix+0.18,cy2+0.3,2.6,0.4,{fs:9,color:INK,bold:true});
  // up arrow critic -> one-step (max Q)
  upArrow(s,ix+iw-1.0,bby+bbh,cy2-(bby+bbh),AMBER_DD);
  txt(s,"max Q",ix+iw-1.95,cy2-0.26,0.9,0.22,{fs:8.5,color:AMBER_DD,bold:true,align:"right"});

  // bottom: actor objective
  const oy=cy2+ch+0.1;
  rrect(s,ix,oy,iw,GY+GH-oy-0.12,"FBEFD6",{line:AMBER,lw:1,r:0.05});
  txt(s,"actor 目标:   min_μ  𝔼[ −Q(s, μ(s,z)) ]  +  α 𝔼[ ‖μ(s,z) − μᵝ(s,z)‖² ]",ix,oy,iw,GY+GH-oy-0.12,{fs:10,color:AMBER_DD,bold:true});
}

// ---- Col4: reward + output ----
group(s,c4.x,GY,c4.w,GH,ORANGE,"退化感知奖励 & 决策输出",{fill:"FCEDE6"});
{
  const bx=c4.x+0.14, bw=c4.w-0.28;
  // reward box
  rrect(s,bx,GY+0.42,bw,1.66,CARD,{line:ORANGE,lw:1.25,r:0.05});
  txt(s,"退化感知奖励 r",bx,GY+0.5,bw,0.3,{fs:10.5,color:ORANGE_D,bold:true});
  s.addText([
    {text:"信息增益 (D-opt / MI)",options:{breakLine:true,color:INK}},
    {text:"− λ₁ · 可定位性风险 R(η)",options:{breakLine:true,color:ORANGE_D,bold:true}},
    {text:"− λ₂ · 运动 / 能耗代价",options:{color:INK}},
  ],{x:bx+0.14,y:GY+0.82,w:bw-0.26,h:1.2,align:"left",valign:"top",fontFace:BFONT,fontSize:9.3,lineSpacingMultiple:1.25,margin:1});
  // arrow reward -> critic (leftward into col3)
  lArrow(s,c3.x+c3.w+0.02,GY+3.7,c4.x-(c3.x+c3.w)-0.04,ORANGE);
  txt(s,"r",c3.x+c3.w,GY+3.4,c4.x-(c3.x+c3.w),0.24,{fs:9,color:ORANGE_D,bold:true});
  // output box
  rrect(s,bx,GY+2.34,bw,1.0,"FCE0D6",{line:ORANGE,lw:1.5,r:0.05});
  txt(s,"探索决策输出 a*",bx,GY+2.42,bw,0.32,{fs:11,color:ORANGE_D,bold:true});
  txt(s,"目标点 / 速度指令\n→ 研究内容3 自适应导航",bx,GY+2.76,bw,0.56,{fs:8.8,color:INK,lsm:1.1});
  // action a (from col3) -> output : arrow into col4 from left at output height
  lArrow(s,c3.x+c3.w+0.02,GY+2.84,c4.x-(c3.x+c3.w)-0.04,AMBER_D);
  txt(s,"a",c3.x+c3.w,GY+2.54,c4.x-(c3.x+c3.w),0.24,{fs:9,color:AMBER_D,bold:true});
}

// ===== closed-loop feedback (dashed, bottom of main, right->left) =====
const fy=GY+GH+0.18;
s.addShape(pres.shapes.LINE,{x:c4.x+c4.w/2,y:GY+GH+0.02,w:0,h:fy-(GY+GH+0.02),line:{color:ORANGE,width:2,dashType:"dash"}});
s.addShape(pres.shapes.LINE,{x:c1.x+c1.w/2,y:fy,w:(c4.x+c4.w/2)-(c1.x+c1.w/2),h:0,line:{color:ORANGE,width:2,dashType:"dash"}});
upArrow(s,c1.x+c1.w/2,GY+GH+0.02,fy-(GY+GH+0.02),ORANGE,"dash");
txt(s,"闭环反馈：执行 a* → 研究内容3 自适应导航 → 环境交互 → 新观测 → 更新感知量化 (研究内容1)",c2.x,fy-0.0,c3.x+c3.w-c2.x,0.26,{fs:9,color:ORANGE_D,bold:true});

// ===== training paradigm band =====
const ty=fy+0.32, th=0.92;
group(s,0.3,ty,12.73,th,SLATE,"训练范式：离线预训练 + 在线适配 (offline → online)",{fill:"EDF0F4",tfs:10.5});
{
  const items=[
    ["演示缓冲 Demos buffer","专家轨迹 / 历史 SLAM 轨迹"],
    ["仿真退化场景","Gazebo 隧道·巷道·矿井"],
    ["离线预训练","BC Flow + 离线 RL (FQL)"],
    ["offline → online 在线微调","真实退化场景适配"],
    ["机载部署","单步推理 ~ ms 级"],
  ];
  const bx0=0.5, tw=12.33, n=items.length, ga=0.26, bw=(tw-ga*(n-1))/n, top=ty+0.34, bh=0.5;
  items.forEach((it,i)=>{const x=bx0+i*(bw+ga);rrect(s,x,top,bw,bh,CARD,{line:SLATE,lw:1,r:0.05});txt(s,it[0],x,top+0.02,bw,0.28,{fs:9,color:NAVY,bold:true,lsm:1.0});txt(s,it[1],x,top+0.28,bw,0.2,{fs:7.6,color:MUTE,lsm:1.0});if(i<n-1)rArrow(s,x+bw+0.02,top+bh/2,ga-0.06,SLATE);});
}

// caption
txt(s,"图 X　研究内容2：退化感知驱动的具身主动探索决策（FQL 双策略）细节框架图",0.3,7.16,12.73,0.3,{fs:10,color:MUTE,italic:true});

pres.writeFile({fileName:"D:/Desktop/Obsidian文件/个人笔记/temp/研究内容二_FQL细节框架图.pptx"})
  .then(f=>console.log("WROTE: "+f)).catch(e=>{console.error(e);process.exit(1);});
