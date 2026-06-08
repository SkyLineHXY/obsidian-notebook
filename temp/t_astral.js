const pptxgen=require("pptxgenjs");
(async()=>{
 const p=new pptxgen();const s=p.addSlide();
 s.addText("地图 𝓜 和 𝒢 graph",{x:1,y:1,w:5,h:1});
 await p.writeFile({fileName:"D:/Desktop/Obsidian文件/个人笔记/temp/min_astral.pptx"});
 console.log("wrote astral");
})();
