export default function(component) {
  const {data, parentElement} = component;
  const root=parentElement.querySelector('.sl-live');
  const q=s=>root.querySelector(s), audio=q('[data-audio]'), seek=q('[data-seek]');
  const set=(s,v)=>{q(s).textContent=v;};
  const clock=s=>`${Math.floor(s/60).toString().padStart(2,'0')}:${Math.floor(s%60).toString().padStart(2,'0')}`;
  set('[data-title]',data.name); set('[data-scope]',data.scope);
  set('[data-resolution]',`${data.settings.grid_spacing_m.toFixed(0)} m grid`);
  set('[data-catalog]',`${data.candidate_segments} candidate segments`);
  // Community Cloud embeds the app below /~/+/. Resolve backend media paths
  // inside that document, preserving the hosting prefix used by its proxy.
  audio.src=data.audio_url.startsWith('/media/')
    ? new URL(`.${data.audio_url}`, document.baseURI).href : data.audio_url;
  seek.max=data.duration_s;
  const map=q('[data-map]'), spectrum=q('[data-spectrum]'), specImage=new Image();
  specImage.src=data.spectrogram_png;
  const cyan='#55e5cf', text='#9db2c4';
  let referenceVisible=false, stateIndex=-1, lastDraw=-1, raf, disposed=false;
  const east=data.east_m, north=data.north_m, duration=data.duration_s;
  const fitCanvas=canvas=>{const box=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);const w=Math.max(1,Math.round(box.width*dpr)),h=Math.max(1,Math.round(box.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return[c,box.width,box.height];};
  function pointAt(track,time){const ts=track.time_s;if(time<ts[0]||time>ts[ts.length-1])return null;let lo=0,hi=ts.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(ts[mid]<=time)lo=mid;else hi=mid;}const f=(time-ts[lo])/(ts[hi]-ts[lo]);return track.xyz_m[lo].map((v,i)=>v+(track.xyz_m[hi][i]-v)*f);}
  function frameAt(t){let i=0;while(i+1<data.states.length&&data.states[i+1].at_s<=t)i++;return i;}
  function mapDraw(t,state){
    const[c,w,h]=fitCanvas(map), pad={l:48,r:20,t:15,b:37};
    const side=Math.min(w-pad.l-pad.r,h-pad.t-pad.b),left=pad.l+(w-pad.l-pad.r-side)/2,top=pad.t;
    const x=v=>left+(v-east[0])/(east[east.length-1]-east[0])*side;
    const y=v=>top+side-(v-north[0])/(north[north.length-1]-north[0])*side;
    c.clearRect(0,0,w,h); c.fillStyle='#091620';c.fillRect(left,top,side,side);
    if(state.surface_hz){const values=state.surface_hz.flat(),min=Math.min(...values),max=Math.max(...values),dx=side/(east.length-1),dy=side/(north.length-1);c.save();c.beginPath();c.rect(left,top,side,side);c.clip();for(let iy=0;iy<north.length;iy++)for(let ix=0;ix<east.length;ix++){const v=(state.surface_hz[iy][ix]-min)/Math.max(max-min,.01);c.fillStyle=`hsl(${90+v*135},${55-v*25}%,${47-v*31}%)`;c.fillRect(x(east[ix])-dx/2,y(north[iy])-dy/2,dx+.5,dy+.5);}c.restore();}
    c.strokeStyle='#b9d4e118';c.lineWidth=1;c.font='9px ui-monospace,monospace';c.fillStyle=text;
    for(let i=0;i<5;i++){const v=i/4,px=left+side*v,py=top+side*v;c.beginPath();c.moveTo(px,top);c.lineTo(px,top+side);c.moveTo(left,py);c.lineTo(left+side,py);c.stroke();c.textAlign='center';c.fillText(((east[0]+v*(east[east.length-1]-east[0]))/1000).toFixed(1),px,top+side+16);c.textAlign='right';c.fillText(((north[north.length-1]-v*(north[north.length-1]-north[0]))/1000).toFixed(1),left-8,py+3);}
    c.textAlign='center';c.fillText('EAST OF SEARCH CENTER · km',left+side/2,top+side+31);
    c.save();c.translate(left-36,top+side/2);c.rotate(-Math.PI/2);c.fillText('NORTH · km',0,0);c.restore();
    const best=state.hypotheses[0],matched=new Set((best?.matches||[]).map(m=>m.candidate));
    c.save();c.beginPath();c.rect(left,top,side,side);c.clip();
    for(const track of data.tracks){const now=t+(best?.clock_offset_s||0),p=pointAt(track,now);if(!p)continue;const active=matched.has(track.name);c.strokeStyle=active?cyan:'#d4e7f055';c.lineWidth=active?1.8:.8;c.beginPath();let began=false;for(let i=0;i<track.time_s.length;i++){if(track.time_s[i]<now-180||track.time_s[i]>now)continue;const pos=track.xyz_m[i];if(!began){c.moveTo(x(pos[0]),y(pos[1]));began=true;}else c.lineTo(x(pos[0]),y(pos[1]));}if(began)c.lineTo(x(p[0]),y(p[1]));c.stroke();const before=pointAt(track,now-1);const angle=before?Math.atan2(-(p[1]-before[1]),p[0]-before[0]):0;c.save();c.translate(x(p[0]),y(p[1]));c.rotate(angle);c.beginPath();c.moveTo(7,0);c.lineTo(-5,-4);c.lineTo(-2,0);c.lineTo(-5,4);c.closePath();c.fillStyle=active?cyan:'#bdd4e0';c.fill();c.restore();}
    for(let i=state.hypotheses.length-1;i>=0;i--){const p=state.hypotheses[i];c.beginPath();c.arc(x(p.east_m),y(p.north_m),i===0?6:3,0,Math.PI*2);c.fillStyle=i===0?'#ffffff':cyan;c.fill();c.fillStyle='#dcecf3';c.font='10px ui-monospace,monospace';c.textAlign='left';c.fillText(`${i+1}`,x(p.east_m)+8,y(p.north_m)-6);}
    if(referenceVisible&&data.reference){const px=x(data.reference.east_m),py=y(data.reference.north_m);c.strokeStyle='#ffd18c';c.lineWidth=2;c.beginPath();c.moveTo(px-7,py);c.lineTo(px+7,py);c.moveTo(px,py-7);c.lineTo(px,py+7);c.stroke();}
    c.restore();
    if(!state.surface_hz){c.textAlign='center';c.fillStyle=text;c.font='12px system-ui';c.fillText('Waiting for a completed observation',left+side/2,top+side/2);}
    if(referenceVisible&&data.reference){const r=data.reference,d=best?Math.hypot(best.east_m-r.east_m,best.north_m-r.north_m):null;set('[data-reference]',`${r.label}${d===null?'':` · top-grid distance: ${d.toFixed(0)} m`}. ${r.note}`);}
  }
  function spectrumDraw(t,state){
    const[c,w,h]=fitCanvas(spectrum),l=44,r=16,top=8,b=24,pw=w-l-r,ph=h-top-b,lo=Math.max(0,t-90),hi=lo+90;
    c.clearRect(0,0,w,h);c.fillStyle='#060f18';c.fillRect(l,top,pw,ph);
    if(specImage.complete&&specImage.naturalWidth&&t>0){const sx=lo/duration*specImage.naturalWidth,sw=(t-lo)/duration*specImage.naturalWidth;if(sw>0)c.drawImage(specImage,sx,0,sw,specImage.naturalHeight,l,top,pw*(t-lo)/90,ph);}
    c.font='9px ui-monospace,monospace';c.fillStyle=text;c.textAlign='right';for(const f of [100,400,800])c.fillText(`${f}`,l-7,top+ph-(f-40)/760*ph+3);
    c.textAlign='center';for(let i=0;i<=3;i++)c.fillText(clock(lo+i*30),l+pw*i/3,h-7);
    const seen=new Set(state.seen_ids);c.save();c.beginPath();c.rect(l,top,pw,ph);c.clip();
    for(const p of data.proposals){if(!seen.has(p.id))continue;c.strokeStyle=cyan;c.lineWidth=1.6;c.beginPath();let started=false;for(let i=0;i<p.time_s.length;i++){if(p.time_s[i]<lo||p.time_s[i]>t)continue;const px=l+(p.time_s[i]-lo)/90*pw,py=top+ph-(p.frequency_hz[i]-40)/760*ph;if(!started){c.moveTo(px,py);started=true;}else c.lineTo(px,py);}c.stroke();}
    c.strokeStyle='#55e5cf88';c.beginPath();c.moveTo(l+(t-lo)/90*pw,top);c.lineTo(l+(t-lo)/90*pw,top+ph);c.stroke();c.restore();
  }
  function updateFeed(state){const feed=q('[data-feed]');feed.replaceChildren();const matches=new Map((state.hypotheses[0]?.matches||[]).map(m=>[m.event_id,m]));for(const id of [...state.seen_ids].reverse().slice(0,6)){const p=data.proposals.find(p=>p.id===id),match=matches.get(id),card=document.createElement('div');card.className='sl-event';const head=document.createElement('div');head.className='sl-event-top';const time=document.createElement('span');time.textContent=`${clock(p.start_s)}–${clock(p.end_s)}`;const tag=document.createElement('span');tag.textContent=match?'CANDIDATE FIT':'UNMATCHED';head.append(time,tag);const name=document.createElement('strong');name.textContent=match?.candidate||'Unconfirmed acoustic proposal';const detail=document.createElement('small');detail.textContent=match?`Tone ${match.source_tone_hz.toFixed(1)} Hz · residual ${match.rmse_hz.toFixed(2)} Hz`:id;card.append(head,name,detail);feed.append(card);}if(!state.seen_ids.length){const empty=document.createElement('p');empty.className='sl-empty';empty.textContent='Play the recording or jump to the next completed observation.';feed.append(empty);}set('[data-reason]',state.reasons.join(' ')||'Finite regional candidate fit; independent location accuracy unvalidated.');}
  function render(force=false){if(disposed)return;if(root.getBoundingClientRect().width===0){audio.pause();set('[data-play]','▶ Play recording');return;}const t=Math.min(audio.currentTime||0,duration),idx=frameAt(t),state=data.states[idx],best=state.hypotheses[0];if(force||idx!==stateIndex){stateIndex=idx;updateFeed(state);set('[data-count]',state.seen_ids.length);set('[data-matched]',best?best.matches.length:'—');set('[data-status]',state.status.replaceAll('_',' '));set('[data-location]',best?`Top grid: ${best.latitude.toFixed(5)}°, ${best.longitude.toFixed(5)}° · ${best.clock_offset_s>=0?'+':''}${best.clock_offset_s.toFixed(1)} s shared clock · ${best.capped_objective_hz.toFixed(2)} Hz objective`:'No location supported yet.');}seek.value=t;set('[data-clock]',`${clock(t)} / ${clock(duration)}`);set('[data-level]',(data.rms_db[Math.min(Math.floor(t),data.rms_db.length-1)]??-100).toFixed(1));mapDraw(t,state);spectrumDraw(t,state);q('[data-next]').disabled=audio.readyState<1||!data.states.some(s=>s.at_s>t+.05);}
  q('[data-play]').onclick=async()=>{if(audio.paused){try{await audio.play();set('[data-play]','Ⅱ Pause');}catch(e){set('[data-reason]',`Playback unavailable: ${e.message}`);}}else{audio.pause();set('[data-play]','▶ Play recording');}};
  q('[data-speed]').onchange=e=>{audio.playbackRate=Number(e.target.value);};
  q('[data-next]').onclick=()=>{const next=data.states.find(s=>s.at_s>audio.currentTime+.05);if(next){audio.currentTime=Math.min(next.at_s+.02,duration);render(true);}};
  q('[data-reset]').onclick=()=>{audio.pause();audio.currentTime=0;set('[data-play]','▶ Play recording');render(true);};
  seek.oninput=()=>{audio.currentTime=Number(seek.value);render(true);};
  q('[data-reveal]').disabled=!data.reference;
  q('[data-reveal]').onclick=()=>{referenceVisible=!referenceVisible;q('[data-reference]').hidden=!referenceVisible;set('[data-reveal]',referenceVisible?'Hide reference':'Reveal reference');render(true);};
  audio.onended=()=>set('[data-play]','▶ Play recording');audio.onloadedmetadata=()=>render(true);audio.onerror=()=>set('[data-reason]','The recording could not be loaded. Reload the app to refresh its local playback URL.');
  const observer=new ResizeObserver(()=>render(true));observer.observe(root);
  function loop(ms){if(disposed)return;if(ms-lastDraw>100){render();lastDraw=ms;}raf=requestAnimationFrame(loop);}raf=requestAnimationFrame(loop);render(true);
  return()=>{disposed=true;cancelAnimationFrame(raf);observer.disconnect();audio.pause();audio.removeAttribute('src');audio.load();};
}
