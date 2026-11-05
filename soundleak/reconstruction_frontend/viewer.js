export default function(component) {
  const root=component.parentElement.querySelector('.sl-cloud'),data=component.data;
  const canvas=root.querySelector('[data-cloud-canvas]'),caption=root.querySelector('[data-cloud-caption]');
  let yaw=.55,pitch=.25,zoom=1,mode='compare',drag=null,disposed=false;
  const all=[...data.predicted,...data.reference,...data.zero_input_control];
  const low=[0,1,2].map(i=>Math.min(...all.map(p=>p[i]))),high=[0,1,2].map(i=>Math.max(...all.map(p=>p[i])));
  const center=low.map((v,i)=>(v+high[i])/2),span=Math.max(...high.map((v,i)=>v-low[i]),.01);
  function rotate(p){
    const x=(p[0]-center[0])/span,y=(p[1]-center[1])/span,z=(p[2]-center[2])/span;
    const a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw);
    return[a,b*Math.sin(pitch)+z*Math.cos(pitch),b*Math.cos(pitch)-z*Math.sin(pitch)];
  }
  function draw(){
    if(disposed)return;
    const box=canvas.getBoundingClientRect();if(box.width<1)return;
    const dpr=Math.min(window.devicePixelRatio||1,2),w=box.width,h=box.height;
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);
    const side=mode==='compare',panels=side?2:1,pw=w/panels;
    function panel(index,layers,title){
      c.save();c.beginPath();c.rect(index*pw,0,pw,h);c.clip();
      const scale=Math.min(pw,h-70)*.88*zoom,cx=index*pw+pw/2,cy=h/2+15;
      const project=p=>{const v=rotate(p),f=3/(3+v[2]);return[cx+v[0]*scale*f,cy-v[1]*scale*f,v[2]];};
      c.fillStyle='#a8becf';c.font='12px system-ui';c.textAlign='center';c.fillText(title,cx,20);
      const origin=low.map((v,i)=>v-(high[i]-v)*.07);
      c.strokeStyle='#365267';c.lineWidth=1;c.font='11px ui-monospace,monospace';
      for(let axis=0;axis<3;axis++){
        const end=[...origin];end[axis]=high[axis];const a=project(origin),b=project(end);
        c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);c.stroke();
        c.fillStyle='#8eaabc';c.fillText('XYZ'[axis],b[0]+8,b[1]);
      }
      const marks=[];for(const layer of layers)for(const p of layer.points){const q=project(p);marks.push({q,color:layer.color});}
      marks.sort((a,b)=>b.q[2]-a.q[2]);
      for(const {q,color} of marks){c.globalAlpha=.72;c.fillStyle=color;c.beginPath();c.arc(q[0],q[1],Math.max(1.1,1.7*zoom),0,Math.PI*2);c.fill();}
      c.globalAlpha=1;c.restore();
    }
    if(side){
      panel(0,[{points:data.predicted,color:'#51e0d2'}],'Wi-Fi model prediction');
      panel(1,[{points:data.reference,color:'#ffc783'}],'Measured LiDAR reference');
      c.strokeStyle='#294052';c.beginPath();c.moveTo(pw,35);c.lineTo(pw,h-10);c.stroke();
    }else if(mode==='control'){
      panel(0,[{points:data.zero_input_control,color:'#c198ff'},{points:data.reference,color:'#ffc783'}],'Zero-input model control + measured reference');
    }else{
      panel(0,[{points:data.predicted,color:'#51e0d2'},{points:data.reference,color:'#ffc783'}],'Wi-Fi prediction + measured reference');
    }
    caption.textContent=mode==='control'
      ? 'Control: the same trained model receives all-zero input. It can still produce a scene-shaped point cloud from learned priors.'
      : `${data.id} · ${data.predicted.length.toLocaleString()} predicted points · ${data.reference.length.toLocaleString()} actual reference points. Views share the same scale and camera.`;
  }
  const buttons=[...root.querySelectorAll('[data-mode]')];
  buttons.forEach(b=>b.onclick=()=>{mode=b.dataset.mode;buttons.forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw();});
  root.querySelector('[data-reset]').onclick=()=>{yaw=.55;pitch=.25;zoom=1;draw();};
  canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw();};
  canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
  const wheel=e=>{e.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw();};
  canvas.addEventListener('wheel',wheel,{passive:false});
  const observer=new ResizeObserver(draw);observer.observe(canvas);draw();
  return()=>{disposed=true;observer.disconnect();canvas.removeEventListener('wheel',wheel);};
}
