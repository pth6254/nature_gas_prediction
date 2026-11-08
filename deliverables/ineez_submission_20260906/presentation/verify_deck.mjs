// Native Chrome DevTools QA; uses Node's built-in WebSocket, no package install.
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const root=resolve(import.meta.dirname,'..');
const out=resolve(root,'tmp/slide_build');
const profile=resolve(out,'cdp-profile-'+Date.now());
await mkdir(profile,{recursive:true});
const child=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',[
 '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
 '--remote-debugging-port=0','--allow-file-access-from-files','--hide-scrollbars',
 '--force-device-scale-factor=1','--user-data-dir='+profile,'about:blank'
],{windowsHide:true,stdio:'ignore'});
let socket;
try {
 let port;
 for(let i=0;i<100;i++){try{port=(await readFile(resolve(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break}catch{await new Promise(r=>setTimeout(r,100))}}
 if(!port)throw Error('Chrome did not expose its task-owned debugging port');
 const pages=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
 socket=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
 await new Promise((res,rej)=>{socket.onopen=res;socket.onerror=rej});
 let id=0;const pending=new Map();const runtimeErrors=[];
 socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')runtimeErrors.push(m.params)};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});socket.send(JSON.stringify({id:n,method,params}))});
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 await call('Runtime.enable');await call('Page.enable');
 await call('Emulation.setDeviceMetricsOverride',{width:1600,height:900,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:pathToFileURL(resolve(root,'presentation/REPORT_SLIDES.html')).href+'?capture=1'});
 for(let i=0;i<100;i++){if(await evaluate("document.documentElement.dataset.ready==='true'"))break;await new Promise(r=>setTimeout(r,100))}
 const checks=[];
 for(let i=0;i<10;i++){
  await evaluate(`show(${i}); document.fonts.ready.then(()=>true)`);
  await new Promise(r=>setTimeout(r,80));
  const result=await evaluate(`(()=>{
   const s=slides[current], b=s.getBoundingClientRect(),footer=s.querySelector('.footer').getBoundingClientRect(),issues=[];
   const texts=[...s.querySelectorAll('h1,h2,h3,p,td,th,svg text,.eyebrow,.metric,.scope')].filter(e=>!e.closest('.source-notes'));
   for(const e of texts){const r=e.getBoundingClientRect();if(r.width&&r.height&&(r.left<b.left-1||r.right>b.right+1||r.top<b.top-1||r.bottom>footer.top-3))issues.push({text:e.textContent.slice(0,100),bounds:[r.x,r.y,r.width,r.height],reason:'outside slide/content bounds'})}
   const overlaps=[];
   for(const chart of s.querySelectorAll('svg')){const labels=[...chart.querySelectorAll('text')];for(let a=0;a<labels.length;a++)for(let z=a+1;z<labels.length;z++){const x=labels[a].getBoundingClientRect(),y=labels[z].getBoundingClientRect();const w=Math.min(x.right,y.right)-Math.max(x.left,y.left),h=Math.min(x.bottom,y.bottom)-Math.max(x.top,y.top);if(w>2&&h>3)overlaps.push([labels[a].textContent,labels[z].textContent])}}
   return{slide:current+1,issues,overlaps,svgCount:s.querySelectorAll('svg').length};
  })()`);
  checks.push(result);
  const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await writeFile(resolve(out,`slide_${String(i+1).padStart(2,'0')}.png`),Buffer.from(shot.data,'base64'));
 }
 // Exercise controls, notes, keyboard navigation and narrow-screen scaling.
 const interactions=await evaluate(`(()=>{show(0);el('next').click();const next=current===1;el('prev').click();const prev=current===0;notes();const opened=!el('notes').hidden;notes();const closed=el('notes').hidden;document.dispatchEvent(new KeyboardEvent('keydown',{key:'End'}));const end=current===9;document.dispatchEvent(new KeyboardEvent('keydown',{key:'Home'}));return{next,prev,opened,closed,end,home:current===0}})()`);
 await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});
 const mobile=await evaluate("(()=>{fit();return {overflow:document.documentElement.scrollWidth>390,stageWidth:el('stage').getBoundingClientRect().width}})()");
 await call('Emulation.setDeviceMetricsOverride',{width:1600,height:900,deviceScaleFactor:1,mobile:false});
 await call('Emulation.setEmulatedMedia',{media:'print'});
 const print=await evaluate(`(()=>{const visible=slides.filter(s=>getComputedStyle(s).display!=='none').length;return{visible,dimensions:slides.map(s=>({width:s.getBoundingClientRect().width,height:s.getBoundingClientRect().height,footerFits:s.querySelector('.footer').getBoundingClientRect().bottom<=s.getBoundingClientRect().bottom+1}))}})()`);
 const report={runtimeErrors,checks,interactions,mobile,print};
 await writeFile(resolve(root,'presentation/qa_report.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));
 await call('Browser.close');
} finally {if(socket)socket.close();if(child.exitCode===null)child.kill();}
