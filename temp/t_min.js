const pptxgen=require("pptxgenjs");
const tests={
 a_textonly:s=>{s.addText("hello",{x:1,y:1,w:3,h:1});},
 b_shape:s=>{s.addShape("roundRect",{x:1,y:1,w:3,h:1,fill:{color:"E9A23B"}});},
 c_line:s=>{s.addShape("line",{x:1,y:1,w:0,h:2,line:{color:"E76F51",width:2}});},
 d_linearrow:s=>{s.addShape("line",{x:1,y:1,w:2,h:0,line:{color:"E76F51",width:2,beginArrowType:"triangle"}});},
 e_image:s=>{s.addImage({path:"D:/Desktop/研究生/国自然_地下空间多机器人SLAM/图片素材/IMU.png",x:1,y:1,w:1,h:1});},
};
(async()=>{for(const[k,fn]of Object.entries(tests)){const p=new pptxgen();const s=p.addSlide();fn(s);await p.writeFile({fileName:"D:/Desktop/Obsidian文件/个人笔记/temp/min_"+k+".pptx"});console.log("wrote",k);}})();
