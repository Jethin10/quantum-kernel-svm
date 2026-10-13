/* Saved-evidence explorer. Controls never start a quantum job or retune a model. */
'use strict';
const $ = id => document.getElementById(id);
const BLUE = '#2565b2', CORAL = '#bf5b35';
const compilerNames = {baseline:'Baseline SABRE',fused_sabre:'Fused SABRE',fused_parity:'Native parity'};
const graphNames = {A:'A · Chain',A_tree:'A · Guide tree',A_cycle:'A · Cycle + tail',B:'B · Square lattice',B_reference:'B · Guide reference',B_tree:'B · Tree'};
let data, condition, selectedRow = 0, selectedColumn = 1;
const percent = x => `${(100*x).toFixed(1)}%`;
const number = x => Number(x).toLocaleString('en-US');
const svg = (body,viewBox='0 0 480 320',label='') => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" role="img" aria-label="${label}">${body}</svg>`;

function graph(config,active){
  const positions = Object.values(config.positions), xs = positions.map(p=>p[0]), ys = positions.map(p=>p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = 190/Math.max(xmax-xmin,1), sy = 112/Math.max(ymax-ymin,1), scale = Math.min(sx,sy);
  const xy = q => [120+(config.positions[q][0]-(xmin+xmax)/2)*scale,75-(config.positions[q][1]-(ymin+ymax)/2)*scale];
  let body = config.edges.map(([a,b])=>{const p=xy(a),q=xy(b);return `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#b1c3d0" stroke-width="2.5"/>`;}).join('');
  for(let q=0;q<config.num_qubits;q++){
    const [x,y]=xy(q), used=active.includes(q);
    body += `<circle cx="${x}" cy="${y}" r="12" fill="${used?BLUE:'#e0e8ed'}"/><text x="${x}" y="${y+3.5}" text-anchor="middle" font-size="10" fill="${used?'white':'#617384'}">${q}</text>`;
  }
  return svg(body,'0 0 240 150',`${graphNames[config.name]}, ${config.num_qubits} physical qubits, ${config.edges.length} edges. Logical placement ${active.join(', ')}.`);
}

function renderCondition(){
  condition = data.conditions.find(r=>r.processor===$('processor').value&&r.compiler===$('compiler').value);
  if(!condition) throw new Error('The selected measured condition is missing.');
  const baseline = data.conditions.find(r=>r.processor===condition.processor&&r.compiler==='baseline');
  const reduction = 100*(baseline.cx-condition.cx)/baseline.cx;
  $('metrics').innerHTML = [
    ['Held-out accuracy',percent(condition.accuracy),`${Math.round(condition.accuracy*20)} / 20 correct · PSD-corrected SVM`],
    ['CX gates / overlap',number(condition.cx),condition.compiler==='baseline'?`${condition.swaps} inserted routing SWAPs`:`${reduction.toFixed(1)}% fewer than ${condition.processor} baseline`],
    ['Compiled depth',number(condition.depth),`${condition.swaps} routing SWAPs · same ideal kernel`],
    ['Raw kernel error',condition.rmse.toFixed(4),'RMSE against exact training kernel'],
  ].map(([label,value,note])=>`<div class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
  $('condition-finding').textContent = `${compilerNames[condition.compiler]} on ${condition.processor} uses ${condition.cx} CX gates per pair. This seeded run classifies ${Math.round(condition.accuracy*20)} of 20 held-out points correctly; 20 conditional shot-resampling trials average ${percent(condition.resampling.mean)} (SD ${percent(condition.resampling.std)}).`;
  $('condition-finding').textContent += ` The 4,840-pair acquisition took ${condition.aerSeconds.toFixed(1)} simulator seconds before cache reuse. Frozen ring: r=${data.featureMap.reps}, scale=${data.featureMap.scale}, C=${data.C}.`;
  const config = data.architectures.find(r=>r.name===condition.processor);
  $('active-graph').innerHTML = graph(config,condition.layout);
  $('graph-title').textContent = graphNames[config.name];
  renderHeatmap(); renderMoons(); renderPair();
}

function renderHeatmap(){
  const canvas=$('heatmap'), ctx=canvas.getContext('2d'),size=canvas.width/80;
  for(let i=0;i<80;i++) for(let j=0;j<80;j++){
    const v=Math.max(0,Math.min(1,condition.rawTrain[i][j]));
    ctx.fillStyle=`rgb(${Math.round(241-223*v)},${Math.round(246-185*v)},${Math.round(251-130*v)})`;
    ctx.fillRect(j*size,i*size,size,size);
  }
  ctx.strokeStyle=CORAL;ctx.lineWidth=2;
  ctx.strokeRect(selectedColumn*size,selectedRow*size,size,size);
}

function renderPair(){
  const i=Math.min(selectedRow,selectedColumn),j=Math.max(selectedRow,selectedColumn);
  const index=data.pairs.findIndex(p=>p[0]===i&&p[1]===j);
  const successes=condition.zeroCounts[index],estimate=successes/data.shots;
  const se=Math.sqrt(estimate*(1-estimate)/data.shots);
  $('pair-detail').innerHTML=`<span>0000 outcomes<strong>${number(successes)} / ${number(data.shots)}</strong></span><span>Measured overlap<strong>${estimate.toFixed(5)}</strong></span><span>Exact reference<strong>${data.exactTrain[i][j].toFixed(5)}</strong></span><span>Shot SE estimate<strong>${se.toFixed(5)}</strong></span>`;
}

function renderMoons(){
  const X=data.dataset.X, y=data.dataset.y, train=new Set(data.dataset.trainIndices), test=data.dataset.testIndices;
  const xmin=-1.4,xmax=2.4,ymin=-.8,ymax=1.5;
  const xx=x=>42+(x-xmin)/(xmax-xmin)*410, yy=y=>278-(y-ymin)/(ymax-ymin)*246;
  let body='';
  for(const x of [-1,0,1,2]) body+=`<line x1="${xx(x)}" y1="25" x2="${xx(x)}" y2="278" stroke="#edf1f4"/><text x="${xx(x)}" y="296" text-anchor="middle" class="plot-label">${x}</text>`;
  for(const v of [-.5,0,.5,1]) body+=`<line x1="42" y1="${yy(v)}" x2="452" y2="${yy(v)}" stroke="#edf1f4"/><text x="30" y="${yy(v)+3}" text-anchor="end" class="plot-label">${v}</text>`;
  X.forEach(([a,b],i)=>{
    const color=y[i]===0?BLUE:CORAL, training=train.has(i), predicted=training?null:condition.predictions[test.indexOf(i)], wrong=!training&&predicted!==y[i];
    const cx=xx(a),cy=yy(b);
    if(wrong) body+=`<path d="M ${cx} ${cy-8} L ${cx+8} ${cy} L ${cx} ${cy+8} L ${cx-8} ${cy} Z" fill="none" stroke="#142a3b" stroke-width="1.5"><title>Sample ${i}: true ${y[i]}, predicted ${predicted}</title></path>`;
    body+=`<circle cx="${cx}" cy="${cy}" r="${training?3.3:4.6}" fill="${training?'white':color}" stroke="${color}" stroke-width="${training?1.2:1.6}" opacity="${training?.72:1}"><title>Sample ${i} · ${training?'training':`held-out, predicted ${predicted}`} · true class ${y[i]}</title></circle>`;
  });
  body+='<text x="250" y="318" text-anchor="middle" class="plot-label">Feature x₁</text><text x="12" y="160" transform="rotate(-90 12 160)" text-anchor="middle" class="plot-label">Feature x₂</text>';
  $('moons').innerHTML=svg(body,'0 0 480 330','Two-moons dataset. Hollow training points, filled held-out points; incorrect predictions marked by diamonds.');
}

function renderArchitectures(){
  const topology=$('interaction').value,router=$('router').value;
  const rows=data.topologyGrid.filter(r=>r.interaction===topology&&r.strategy===router);
  $('topology-atlas').innerHTML=data.architectures.map(config=>{
    const row=rows.find(r=>r.processor===config.name),layout=JSON.parse(row.initial_layout.replaceAll("'",'"'));
    return `<article class="topology-card"><h3>${graphNames[config.name]}</h3><small>${config.num_qubits} physical qubits · ${config.edges.length} couplings</small><div class="graph">${graph(config,layout)}</div><div class="graph-stats"><span>CX<strong>${row.cx_count}</strong></span><span>Depth<strong>${row.depth}</strong></span><span>SWAPs<strong>${row.routing_swaps}</strong></span></div></article>`;
  }).join('');
  const a=rows.find(r=>r.processor==='A'),b=rows.find(r=>r.processor==='B');
  $('topology-finding').textContent=`${topology==='full'?'Full K4':topology==='ring'?'Ring':'Linear'} with ${router==='sabre'?'SABRE':'BasicSwap'}: chain A uses ${a.cx_count} CX / depth ${a.depth}; square B uses ${b.cx_count} CX / depth ${b.depth}.`;
}

function renderRobustness(){
  const metric=$('robust-metric').value,title={accuracy:'Seeded held-out accuracy',kernel_rmse:'Raw kernel RMSE versus each split’s exact reference',cx_count:'CX gates per overlap'}[metric];
  $('robust-chart-title').textContent=title;
  const rows=data.noisySplits,all=rows.map(r=>r[metric]);
  const min=metric==='accuracy'?0:0,max=metric==='accuracy'?1.05:Math.max(...all)*1.12;
  const x=seed=>70+seed*190,y=value=>245-(value-min)/(max-min)*205;
  let body='';
  for(let n=0;n<=4;n++){
    const v=metric==='accuracy'?n*.25:max*n/4;
    const label=metric==='accuracy'?percent(v):metric==='cx_count'?v.toFixed(0):v.toFixed(2);
    body+=`<line x1="70" y1="${y(v)}" x2="830" y2="${y(v)}" stroke="#e4ebf0"/><text x="55" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#617384">${label}</text>`;
  }
  for(let seed=0;seed<5;seed++)body+=`<text x="${x(seed)}" y="270" text-anchor="middle" font-size="11" fill="#617384">Seed ${seed}</text>`;
  for(const processor of ['A','B'])for(const compiler of ['baseline','fused_parity']){
    const group=rows.filter(r=>r.processor===processor&&r.compiler===compiler).sort((a,b)=>a.split_seed-b.split_seed), color=processor==='A'?BLUE:CORAL;
    body+=`<polyline points="${group.map(r=>`${x(r.split_seed)},${y(r[metric])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" ${compiler==='baseline'?'stroke-dasharray="6 5"':''}/>`;
    for(const row of group)body+=`<circle cx="${x(row.split_seed)}" cy="${y(row[metric])}" r="${compiler==='baseline'?4:5}" fill="${compiler==='baseline'?'white':color}" stroke="${color}" stroke-width="2"><title>${processor} ${compilerNames[compiler]} · split ${row.split_seed} · ${metric}: ${row[metric]} · r=${row.reps}, scale=${row.scale}, C=${row.C}</title></circle>`;
  }
  $('robust-chart').innerHTML=svg(body,'0 0 900 295',`${title} across five split seeds. Exact values are available in the accompanying summary table and downloadable data.`);
  $('robust-table').innerHTML=data.noisySummary.map(r=>`<tr><td>${r.processor} / ${compilerNames[r.compiler]}</td><td>${percent(r.accuracy_mean)} ± ${percent(r.accuracy_std)}</td><td>${percent(r.accuracy_min)}–${percent(r.accuracy_max)}</td><td>${r.kernel_rmse_mean.toFixed(4)}</td><td>${r.cx_mean.toFixed(1)}</td></tr>`).join('');
  const a=data.noisySummary.find(r=>r.processor==='A'&&r.compiler==='fused_parity'),b=data.noisySummary.find(r=>r.processor==='B'&&r.compiler==='fused_parity');
  const improved=data.pairedDifferences.filter(r=>r.kernel_rmse_reduction>0).length;
  $('robust-finding').innerHTML=`<strong>Native parity: ${percent(a.accuracy_mean)} mean accuracy on A, ${percent(b.accuracy_mean)} on B across five partitions.</strong><p>Raw kernel distortion decreases in ${improved} of 10 paired processor/split comparisons. These are descriptive results on overlapping partitions of the same dataset. Inspect accuracy alongside kernel error and gate cost; no statistical significance or population superiority is claimed.</p>`;
}

function updatePair(){
  const row=Number($('pair-row').value),column=Number($('pair-column').value);
  selectedRow=Number.isFinite(row)?Math.max(0,Math.min(79,Math.round(row))):0;
  selectedColumn=Number.isFinite(column)?Math.max(0,Math.min(79,Math.round(column))):0;
  $('pair-row').value=selectedRow;$('pair-column').value=selectedColumn;
  renderHeatmap();renderPair();
}

async function load(){
  $('error').hidden=true;$('status').hidden=false;$('workspace').hidden=true;
  try{
    const response=await fetch('data/experiment.json',{cache:'no-cache'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    data=await response.json();
    if(data.schemaVersion!==1||data.conditions.length!==6||data.noisySplits.length!==20)throw new Error('Incomplete experiment export');
    renderCondition();renderArchitectures();renderRobustness();
    $('provenance').textContent=`Artifact run: ${new Date(data.generatedAt).toISOString().slice(0,10)} · saved simulation`;
    $('status').hidden=true;$('workspace').hidden=false;
    if(['#experiment','#architectures','#robustness','#method','#main'].includes(location.hash))document.querySelector(location.hash)?.scrollIntoView();
  }catch(error){$('status').hidden=true;$('error').hidden=false;console.error('Experiment explorer:',error);}
}

for(const id of ['processor','compiler'])$(id).addEventListener('change',renderCondition);
for(const id of ['interaction','router'])$(id).addEventListener('change',renderArchitectures);
$('robust-metric').addEventListener('change',renderRobustness);
for(const id of ['pair-row','pair-column']){
  $(id).addEventListener('input',updatePair);
  $(id).addEventListener('change',updatePair);
}
$('heatmap').addEventListener('click',event=>{
  const box=event.currentTarget.getBoundingClientRect();
  $('pair-row').value=Math.floor((event.clientY-box.top)/box.height*80);
  $('pair-column').value=Math.floor((event.clientX-box.left)/box.width*80);updatePair();
});
$('retry').addEventListener('click',load);
function updateNavigation(){
  const hash=location.hash||'#experiment';
  document.querySelectorAll('.nav-link').forEach(a=>{const active=a.getAttribute('href')===hash;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
}
window.addEventListener('hashchange',updateNavigation);updateNavigation();load();
