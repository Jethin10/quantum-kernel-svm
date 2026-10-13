/* Exact browser lab plus a separate explorer of saved noisy measurements. */
'use strict';
const $ = id => document.getElementById(id);
const BLUE = '#2565b2', CORAL = '#bf5b35';
const compilerNames = {baseline:'Baseline SABRE',fused_sabre:'Fused SABRE',fused_parity:'Native parity'};
const graphNames = {A:'A · Chain',A_tree:'A · Guide tree',A_cycle:'A · Cycle + tail',B:'B · Square lattice',B_reference:'B · Guide reference',B_tree:'B · Tree'};
let data, condition, selectedRow = 0, selectedColumn = 1,selectedArchitecture='A',study='baselines';
const percent = x => `${(100*x).toFixed(1)}%`;
const number = x => Number(x).toLocaleString('en-US');
const svg = (body,viewBox='0 0 480 320',label='') => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" role="${body.includes('role="button"')?'group':'img'}" aria-label="${label}">${body}</svg>`;

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
  $('condition-finding').innerHTML=`<strong>${Math.round(condition.accuracy*20)} / 20 correct in this seeded run.</strong><p>Conditional shot mean ${percent(condition.resampling.mean)}, SD ${percent(condition.resampling.std)}. Original acquisition ${condition.aerSeconds.toFixed(1)} simulator seconds.</p>`;
  const config = data.architectures.find(r=>r.name===condition.processor);
  $('active-graph').innerHTML = graph(config,condition.layout);
  $('graph-title').textContent = graphNames[config.name];
  renderHeatmap(); renderMoons(); renderPair();renderCompilerComparison();
}

function renderHeatmap(){
  const canvas=$('heatmap'), ctx=canvas.getContext('2d'),size=canvas.width/80;
  const mode=$('matrix-mode').value;
  $('matrix-title').textContent={raw:'Raw noisy kernel',exact:'Exact reference',psd:'PSD-corrected kernel',error:'Absolute raw kernel error'}[mode];
  $('color-max').textContent=mode==='error'?'0.25+':'1';
  document.querySelector('.color-key i').style.background=mode==='error'?'linear-gradient(90deg,#fff7ed,#c36d26)':'linear-gradient(90deg,#f1f6fb,#123d79)';
  for(let i=0;i<80;i++) for(let j=0;j<80;j++){
    const value=mode==='exact'?data.exactTrain[i][j]:mode==='psd'?condition.psdTrain[i][j]:mode==='error'?Math.abs(condition.rawTrain[i][j]-data.exactTrain[i][j])*4:condition.rawTrain[i][j];
    const v=Math.max(0,Math.min(1,value));
    ctx.fillStyle=mode==='error'?`rgb(${Math.round(255-60*v)},${Math.round(247-138*v)},${Math.round(237-199*v)})`:`rgb(${Math.round(241-223*v)},${Math.round(246-185*v)},${Math.round(251-130*v)})`;
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
    body+=`<circle ${training?`data-train-row="${data.dataset.trainIndices.indexOf(i)}" tabindex="0" role="button" aria-label="Inspect training sample ${data.dataset.trainIndices.indexOf(i)}"`:''} cx="${cx}" cy="${cy}" r="${training?3.3:4.6}" fill="${training?'white':color}" stroke="${color}" stroke-width="${training?1.2:1.6}" opacity="${training?.72:1}"><title>Sample ${i} · ${training?'training':`held-out, predicted ${predicted}`} · true class ${y[i]}</title></circle>`;
  });
  body+='<text x="250" y="318" text-anchor="middle" class="plot-label">Feature x₁</text><text x="12" y="160" transform="rotate(-90 12 160)" text-anchor="middle" class="plot-label">Feature x₂</text>';
  $('moons').innerHTML=svg(body,'0 0 480 330','Two-moons dataset. Hollow training points, filled held-out points; incorrect predictions marked by diamonds.');
}

function renderArchitectures(){
  const topology=$('interaction').value,router=$('router').value;
  const rows=data.topologyGrid.filter(r=>r.interaction===topology&&r.strategy===router);
  $('topology-atlas').innerHTML=data.architectures.map(config=>{
    const row=rows.find(r=>r.processor===config.name),layout=JSON.parse(row.initial_layout.replaceAll("'",'"'));
    return `<button class="topology-card ${selectedArchitecture===config.name?'selected':''}" data-architecture="${config.name}" aria-pressed="${selectedArchitecture===config.name}"><h3>${graphNames[config.name]}</h3><small>${config.num_qubits} physical qubits · ${config.edges.length} couplings</small><div class="graph">${graph(config,layout)}</div><div class="graph-stats"><span>CX<strong>${row.cx_count}</strong></span><span>Depth<strong>${row.depth}</strong></span><span>SWAPs<strong>${row.routing_swaps}</strong></span></div></button>`;
  }).join('');
  const a=rows.find(r=>r.processor==='A'),b=rows.find(r=>r.processor==='B');
  $('topology-finding').textContent=`${topology==='full'?'Full K4':topology==='ring'?'Ring':'Linear'} with ${router==='sabre'?'SABRE':'BasicSwap'}: chain A uses ${a.cx_count} CX / depth ${a.depth}; square B uses ${b.cx_count} CX / depth ${b.depth}.`;
  const own=data.topologyGrid.filter(r=>r.processor===selectedArchitecture&&r.interaction===topology);
  $('architecture-detail').innerHTML=`<h2>${graphNames[selectedArchitecture]} / ${topology} routing</h2>`+horizontalBars(own.map(r=>({label:r.strategy==='sabre'?'SABRE':'BasicSwap',value:r.cx_count,note:`Depth ${r.depth}, ${r.routing_swaps} SWAPs`,color:BLUE})),{label:'CX gates per overlap'});
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
    for(const row of group)body+=`<circle data-split="${row.split_seed}" tabindex="0" role="button" aria-label="Inspect split ${row.split_seed}, ${processor} ${compilerNames[compiler]}" cx="${x(row.split_seed)}" cy="${y(row[metric])}" r="${compiler==='baseline'?4:5}" fill="${compiler==='baseline'?'white':color}" stroke="${color}" stroke-width="2"><title>${processor} ${compilerNames[compiler]} · split ${row.split_seed} · ${metric}: ${row[metric]} · r=${row.reps}, scale=${row.scale}, C=${row.C}</title></circle>`;
  }
  $('robust-chart').innerHTML=svg(body,'0 0 900 295',`${title} across five split seeds. Exact values are available in the accompanying summary table and downloadable data.`);
  $('robust-table').innerHTML=data.noisySummary.map(r=>`<tr><td>${r.processor} / ${compilerNames[r.compiler]}</td><td>${percent(r.accuracy_mean)} ± ${percent(r.accuracy_std)}</td><td>${percent(r.accuracy_min)}–${percent(r.accuracy_max)}</td><td>${r.kernel_rmse_mean.toFixed(4)}</td><td>${r.cx_mean.toFixed(1)}</td></tr>`).join('');
  const a=data.noisySummary.find(r=>r.processor==='A'&&r.compiler==='fused_parity'),b=data.noisySummary.find(r=>r.processor==='B'&&r.compiler==='fused_parity');
  const improved=data.pairedDifferences.filter(r=>r.kernel_rmse_reduction>0).length;
  $('robust-finding').innerHTML=`<strong>Native parity mean: A ${percent(a.accuracy_mean)}, B ${percent(b.accuracy_mean)}.</strong> Lower raw kernel error in ${improved} / 10 paired comparisons. Overlapping splits, descriptive results.`;
  renderSplit();
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
    renderCondition();renderArchitectures();renderRobustness();initLive();renderStudy();renderGallery();
    $('provenance').textContent=`Artifact run: ${new Date(data.generatedAt).toISOString().slice(0,10)} · saved simulation`;
    $('status').hidden=true;$('workspace').hidden=false;
    updateNavigation();
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
  const allowed=['#playground','#experiment','#architectures','#robustness','#study','#evidence'];
  const hash=allowed.includes(location.hash)?location.hash:'#playground';
  document.querySelectorAll('.view').forEach(section=>{section.hidden='#'+section.id!==hash;});
  document.querySelectorAll('.nav-link').forEach(a=>{const active=a.getAttribute('href')===hash;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
}
window.addEventListener('hashchange',()=>{updateNavigation();window.scrollTo({top:0});});

function horizontalBars(rows,{label='',percentage=false}={}){
  const max=percentage?1.05:Math.max(...rows.map(r=>r.value),.001)*1.13,height=rows.length*56+40;
  let body='';
  rows.forEach((r,i)=>{
    const y=18+i*56;
    body+=`<text x="0" y="${y+16}" font-size="12" fill="#345071">${r.label}</text><rect x="245" y="${y}" width="${Math.max(1,r.value/max*555)}" height="24" rx="4" fill="${r.color||BLUE}"/><text x="${250+r.value/max*555}" y="${y+16}" font-size="12" fill="#345071">${percentage?percent(r.value):r.value<1?r.value.toFixed(4):r.value.toFixed(0)}</text><text x="245" y="${y+42}" font-size="10" fill="#617384">${r.note||''}</text>`;
  });
  return svg(body,`0 0 900 ${height}`,label);
}

function renderCompilerComparison(){
  const metric=$('compiler-metric').value,rows=data.conditions.filter(r=>r.processor===condition.processor);
  $('compiler-comparison').innerHTML=horizontalBars(rows.map(r=>({label:compilerNames[r.compiler],value:r[metric],note:`${r.cx} CX, depth ${r.depth}, ${r.swaps} SWAPs`,color:r.compiler==='fused_parity'?'#0c8c78':BLUE})),{label:metric,percentage:metric==='accuracy'});
}

function renderSplit(){
  const seed=Number($('robust-seed').value),rows=data.noisySplits.filter(r=>r.split_seed===seed),choice=rows[0];
  $('split-title').textContent=`Seed ${seed}: r=${choice.reps}, scale=${choice.scale}, C=${choice.C}`;
  $('split-cards').innerHTML='<div class="split-cards">'+rows.map(r=>`<div class="split-card">${r.processor} / ${compilerNames[r.compiler]}<strong>${percent(r.accuracy)}</strong>${r.cx_count} CX · raw RMSE ${r.kernel_rmse.toFixed(4)}</div>`).join('')+'</div>';
  const groups=rows.map(r=>({label:`${r.processor} ${compilerNames[r.compiler]}`,values:data.conditionalTrials.filter(t=>t.split_seed===seed&&t.processor===r.processor&&t.compiler===r.compiler).map(t=>t.accuracy),color:r.processor==='A'?BLUE:CORAL}));
  let body='';
  for(let i=0;i<groups.length;i++){
    const g=groups[i],y=35+i*52;body+=`<text x="0" y="${y+4}" font-size="10" fill="#617384">${g.label}</text><line x1="175" y1="${y}" x2="480" y2="${y}" stroke="#dbe3ee"/>`;
    const counts={};g.values.forEach(v=>{counts[v]=(counts[v]||0)+1;});
    for(const [v,count] of Object.entries(counts))for(let j=0;j<count;j++)body+=`<circle cx="${175+(Number(v)-.5)*610+(Math.floor(j/5)-(Math.ceil(count/5)-1)/2)*7}" cy="${y+12-(j%5)*6}" r="3" fill="${g.color}"><title>${g.label}: ${percent(Number(v))}, ${count} of 20 trials</title></circle>`;
  }
  for(const v of [.5,.75,1])body+=`<text x="${175+(v-.5)*610}" y="250" text-anchor="middle" font-size="10" fill="#617384">${percent(v)}</text>`;
  $('shot-distribution').innerHTML=svg(body,'0 0 515 270','Twenty conditional shot trials for each condition on the selected split, accuracy from 50 to 100 percent.');
}

let studyRows=[];
function renderStudy(){
  const device=$('study-device').value;
  document.querySelectorAll('[data-study]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.study===study)));
  $('study-device').disabled=['baselines','encoding','selection','splits','features'].includes(study);
  const notes={
    baselines:'Same holdout, independently tuned training folds. Twenty test points give five-percentage-point accuracy resolution.',
    shots:'Twenty conditional binomial resamples per shot budget, from fixed saved noisy probabilities. More shots reduce sampling variance, not noise bias.',
    tradeoffs:'Training-only C selection at the primary angle scale. The displayed frontier is an observed tradeoff, not a global optimum or a held-out model choice.',
    encoding:'Both encodings are compared inside training folds at the frozen primary repetitions and angle scale.',
    selection:'Select a tile to inspect its five-fold CV result. Every fold fits preprocessing on its own training subset.',
    splits:'Ten predeclared partitions repeat training-only model selection. Exact kernels only, on overlapping partitions of the same dataset.',
    mitigation:'Saved primary holdout accuracy. PSD correction uses training eigenspaces. Readout mitigation assumes exact synthetic-channel knowledge.',
    trees:'Measured noisy kernels on the guide A tree and a nine-node B tree, with the frozen primary model. More qubits alone do not imply fewer gates.',
    secondary:'The secondary linear feature map is selected by a training-only cost rule. It has its own exact reference. Conditional shot means expose the seeded-score tradeoff.',
    features:'Post-selection exact controls at the frozen primary angle scale and C. These held-out scores do not select a new model.',
  };
  const titles={baselines:'Quantum and classical classifiers',shots:'Shot budget versus accuracy',tradeoffs:'Native CX cost versus training quality',encoding:'Two ways to encode two features',selection:'Which feature map did training choose?',splits:'Exact accuracy across ten splits',mitigation:'Noise, PSD repair and readout correction',trees:'The measured tree controls',secondary:'A cheaper kernel, with a tradeoff',features:'Exact feature-map controls'};
  $('study-chart-title').textContent=titles[study];$('study-note').textContent=notes[study];
  $('study-detail').textContent='Select a chart point or tile to inspect the settings.';
  studyRows=[];
  if(study==='tradeoffs'){
    studyRows=data.trainingTradeoffs.filter(r=>r.processor===device);
    const max=Math.max(...studyRows.map(r=>r.cx_count))*1.1;
    let body='';
    for(const v of [.8,.9,1])body+=`<line x1="55" y1="${300-(v-.75)/.3*260}" x2="835" y2="${300-(v-.75)/.3*260}" stroke="#e2e9f2"/><text x="42" y="${304-(v-.75)/.3*260}" text-anchor="end" font-size="11" fill="#617384">${percent(v)}</text>`;
    studyRows.forEach((r,i)=>{const x=55+r.cx_count/max*780,y=300-(r.cv_mean-.75)/.3*260,color={linear:'#0c8c78',ring:BLUE,full:'#8457b3'}[r.interaction];body+=`<circle data-study-row="${i}" role="button" tabindex="0" aria-label="Inspect ${r.interaction}, ${r.reps} layers" cx="${x}" cy="${y}" r="${r.pareto?9:6}" fill="${color}" stroke="white" stroke-width="2"><title>${r.interaction}, r=${r.reps}, C=${r.C}: ${r.cx_count} CX, CV ${percent(r.cv_mean)}</title></circle><text x="${x+12}" y="${y+4}" font-size="11" fill="${color}">${r.interaction} r${r.reps}${r.pareto?' ★':''}</text>`;});
    body+='<text x="420" y="345" text-anchor="middle" font-size="12" fill="#617384">CX gates per overlap, larger points mark the observed Pareto frontier</text>';
    for(let v=0;v<max;v+=20)body+=`<text x="${55+v/max*780}" y="322" text-anchor="middle" font-size="10" fill="#617384">${v}</text>`;
    $('study-chart').innerHTML=svg(body,'0 0 950 370',titles[study]);
  }else if(study==='selection'){
    studyRows=data.modelSelection;
    const specs=[...new Set(studyRows.map(r=>`r${r.reps} / s${r.scale}`))];let body='';
    [.1,1,10,100].forEach((c,j)=>{body+=`<text x="${270+j*145}" y="22" text-anchor="middle" font-size="12" fill="#617384">C = ${c}</text>`;});
    specs.forEach((key,i)=>{
      body+=`<text x="35" y="${62+i*46}" font-size="12" fill="#345071">${key}</text>`;
      [.1,1,10,100].forEach((c,j)=>{const index=studyRows.findIndex(r=>`r${r.reps} / s${r.scale}`===key&&r.C===c),r=studyRows[index],opacity=.15+.85*r.cv_mean;body+=`<g data-study-row="${index}" role="button" tabindex="0" aria-label="Inspect ${key}, C ${c}, CV ${percent(r.cv_mean)}"><rect x="${200+j*145}" y="${37+i*46}" width="137" height="39" rx="5" fill="${BLUE}" fill-opacity="${opacity}"/><text x="${270+j*145}" y="${62+i*46}" text-anchor="middle" font-size="12" fill="white">${percent(r.cv_mean)}</text></g>`;});
    });
    $('study-chart').innerHTML=svg(body,`0 0 850 ${55+specs.length*46}`,titles[study]);
    $('study-detail').textContent=`Frozen primary choice: ring, r=${data.featureMap.reps}, scale=${data.featureMap.scale}, C=${data.C}. Click a tile for its CV spread.`;
  }else if(study==='splits'){
    const models=[...new Set(data.exactSplits.map(r=>r.model))];let body='';
    for(const v of [.6,.8,1])body+=`<line x1="55" y1="${300-v*270}" x2="860" y2="${300-v*270}" stroke="#e2e9f2"/><text x="42" y="${304-v*270}" text-anchor="end" font-size="11" fill="#617384">${percent(v)}</text>`;
    models.forEach((model,k)=>{const rows=data.exactSplits.filter(r=>r.model===model),color=[BLUE,CORAL,'#0c8c78'][k];body+=`<polyline points="${rows.map(r=>`${55+r.split_seed*89},${300-r.accuracy*270}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;rows.forEach(r=>{const index=studyRows.push(r)-1;body+=`<circle data-study-row="${index}" role="button" tabindex="0" aria-label="${model}, seed ${r.split_seed}" cx="${55+r.split_seed*89}" cy="${300-r.accuracy*270}" r="4" fill="${color}"><title>${model}: seed ${r.split_seed}, ${percent(r.accuracy)}</title></circle>`;});body+=`<text x="${50+k*270}" y="352" font-size="12" fill="${color}">${model}</text>`;});
    for(let seed=0;seed<10;seed++)body+=`<text x="${55+seed*89}" y="320" text-anchor="middle" font-size="11" fill="#617384">${seed}</text>`;
    $('study-chart').innerHTML=svg(body,'0 0 930 375',titles[study]);
  }else{
    let rows=[];
    if(study==='baselines')rows=data.classifiers.filter(r=>['Classical LINEAR','Classical RBF','Exact QSVM','Ideal shot QSVM'].includes(r.model)).map(r=>({label:r.model,value:r.accuracy,note:`F1 ${r.f1.toFixed(3)}`,color:r.model.includes('QSVM')?BLUE:'#0c8c78'}));
    if(study==='shots')rows=data.shotSensitivity.filter(r=>r.processor===device).map(r=>({label:`${number(r.shots)} shots`,value:r.accuracy_mean,note:`SD ${percent(r.accuracy_std)} · raw RMSE ${r.kernel_rmse_mean.toFixed(4)}`,color:BLUE}));
    if(study==='encoding')rows=['repeated_coordinates','mean_difference_coordinates'].map(encoding=>{const r=data.encodingControls.filter(r=>r.encoding===encoding).sort((a,b)=>b.cv_mean-a.cv_mean)[0];return {label:encoding==='repeated_coordinates'?'(a, b, a, b)':'(a, b, mean, difference)',value:r.cv_mean,note:`Best training CV at C=${r.C}, SD ${percent(r.cv_std)}`,color:encoding==='repeated_coordinates'?BLUE:'#8457b3'};});
    if(study==='mitigation')rows=data.classifiers.filter(r=>r.model.startsWith(device+' ')).map(r=>({label:r.model.slice(2),value:r.accuracy,note:`F1 ${r.f1.toFixed(3)} · seeded holdout`,color:BLUE}));
    if(study==='trees')rows=data.allMeasurements.filter(r=>r.processor.endsWith('tree')).map(r=>({label:graphNames[r.processor],value:r.accuracy,note:`${r.cx_count} CX · depth ${r.depth} · raw RMSE ${r.kernel_rmse.toFixed(4)}`,color:r.processor.startsWith('A')?BLUE:CORAL}));
    if(study==='features')rows=data.featureMapControls.map(r=>({label:`${r.topology} / ${r.reps} layers`,value:r.accuracy,note:`Exact held-out result, frozen C=${data.C}`,color:BLUE}));
    if(study==='secondary'){
      const ring=data.conditions.find(r=>r.processor===device&&r.compiler==='fused_parity'),linear=data.costAware.find(r=>r.processor===device),trials=data.costAwareTrials.find(r=>r.processor===device);
      rows=[{label:'Ring / seeded',value:ring.accuracy,note:`${ring.cx} CX · primary model`,color:BLUE},{label:'Linear / seeded',value:linear.accuracy,note:`${linear.cx_count} CX · separate selected kernel`,color:'#0c8c78'},{label:'Ring / conditional mean',value:ring.resampling.mean,note:`20 trials · SD ${percent(ring.resampling.std)}`,color:BLUE},{label:'Linear / conditional mean',value:trials.mean,note:`20 trials · SD ${percent(trials.std)}`,color:'#0c8c78'}];
    }
    $('study-chart').innerHTML=horizontalBars(rows,{label:titles[study],percentage:true});
    $('study-detail').textContent=study==='secondary'?'The 22-CX linear kernel is cheaper, but its conditional shot mean is lower. A perfect seeded score hides that tradeoff.':study==='shots'?'Compare the error and variability as you increase the shot budget. Gate noise remains even at high shot counts.':study==='baselines'?'The classical RBF and exact QSVM both classify all 20 held-out points correctly. This experiment does not establish quantum advantage.':study==='encoding'?'Duplicating coordinates preserves two independent features while using four qubits. The alternative encoding did not improve training CV here.':'The full comparison includes every recorded outcome, including conditions that perform worse.';
  }
}

function renderGallery(){
  $('figure-gallery').innerHTML=data.gallery.map((g,i)=>`<button class="figure-card" data-figure="${i}"><img src="${g.path}" alt="${g.name}" loading="lazy"><span>${g.name}</span></button>`).join('');
}

let livePoints,activePoint=0,liveStates,background='',dragging=null,sampleSeed=42;
const liveSpec=()=>({reps:Number($('live-reps').value),scale:Number($('live-scale').value),topology:$('live-topology').value});
const scaled=p=>p.map((v,i)=>v*data.liveModel.scalerScale[i]+data.liveModel.scalerMin[i]);
function isPrimary(){const s=liveSpec();return s.reps===data.featureMap.reps&&Math.abs(s.scale-data.featureMap.scale)<1e-10&&s.topology===data.featureMap.topology;}
const lx=x=>45+(x+1.4)/3.8*475,ly=y=>307-(y+.8)/2.3*270;
function initLive(){
  livePoints=[data.dataset.X[data.dataset.trainIndices[0]].slice(),data.dataset.X[data.dataset.trainIndices[1]].slice()];
  $('live-reps').value=data.featureMap.reps;$('live-scale').value=data.featureMap.scale;$('live-topology').value=data.featureMap.topology;
  background='';
  const nx=30,ny=20;
  for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){
    const point=[-1.4+(i+.5)/nx*3.8,-.8+(j+.5)/ny*2.3],score=QuantumEngine.decision(QuantumEngine.state(scaled(point),data.featureMap),data.liveModel);
    background+=`<rect x="${45+i*475/nx}" y="${37+(ny-1-j)*270/ny}" width="${475/nx+.5}" height="${270/ny+.5}" fill="${score>=0?'#bf5b35':'#2565b2'}" fill-opacity=".1"/>`;
  }
  renderLive(true);
}
function renderLive(resetStep=false){
  const spec=liveSpec();liveStates=livePoints.map(p=>QuantumEngine.state(scaled(p),spec));
  const fidelity=QuantumEngine.fidelity(...liveStates);
  $('live-fidelity').textContent=fidelity.toFixed(5);$('fidelity-meter').firstElementChild.style.width=`${fidelity*100}%`;
  $('scale-value').textContent=spec.scale.toFixed(2);
  $('live-x').value=livePoints[activePoint][0];$('live-y').value=livePoints[activePoint][1];
  $('live-x-value').textContent=livePoints[activePoint][0].toFixed(2);$('live-y-value').textContent=livePoints[activePoint][1].toFixed(2);
  $('choose-a').setAttribute('aria-pressed',String(activePoint===0));$('choose-b').setAttribute('aria-pressed',String(activePoint===1));
  const primary=isPrimary();$('decision-background').disabled=!primary;
  $('live-prediction').textContent=primary?`Exact SVM predictions: A → class ${QuantumEngine.decision(liveStates[0],data.liveModel)>=0?1:0}; B → class ${QuantumEngine.decision(liveStates[1],data.liveModel)>=0?1:0}. Same trained model, new query points.`:'Exploratory encoding. The trained classifier and decision regions are hidden because its kernel settings differ.';
  $('sample-result').textContent='Sample this overlap to see measurement variability.';
  let body=primary&&$('decision-background').checked?background:'';
  for(const x of [-1,0,1,2])body+=`<line x1="${lx(x)}" y1="37" x2="${lx(x)}" y2="307" stroke="#d9e4ef"/><text x="${lx(x)}" y="327" text-anchor="middle" class="plot-label">${x}</text>`;
  for(const y of [-.5,0,.5,1])body+=`<line x1="45" y1="${ly(y)}" x2="520" y2="${ly(y)}" stroke="#d9e4ef"/><text x="30" y="${ly(y)+4}" text-anchor="end" class="plot-label">${y}</text>`;
  data.dataset.X.forEach((p,i)=>{body+=`<circle cx="${lx(p[0])}" cy="${ly(p[1])}" r="3" fill="white" stroke="${data.dataset.y[i]===0?BLUE:CORAL}" opacity=".65"/>`;});
  livePoints.forEach((p,i)=>{const x=lx(p[0]),y=ly(p[1]),color=i===0?BLUE:CORAL;body+=`<g data-handle="${i}" class="handle" role="button" tabindex="0" aria-label="Select point ${i===0?'A':'B'}"><circle cx="${x}" cy="${y}" r="22" fill="${color}" fill-opacity=".13"/><circle cx="${x}" cy="${y}" r="11" fill="${color}" stroke="white" stroke-width="3"/><text x="${x+17}" y="${y-14}" font-size="14" font-weight="600" fill="${color}">${i===0?'A':'B'}</text></g>`;});
  body+='<text x="282" y="352" class="plot-label" text-anchor="middle">Feature x₁</text>';
  $('live-moons').innerHTML=svg(body,'0 0 560 365','Drag points A and B to calculate their four-qubit fidelity. The sliders provide a keyboard alternative.');
  const gates=QuantumEngine.instructions(scaled(livePoints[0]),spec);$('gate-step').max=gates.length;
  if(resetStep)$('gate-step').value=gates.length;
  renderCircuit();
}
function renderCircuit(){
  const spec=liveSpec(),gates=QuantumEngine.instructions(scaled(livePoints[0]),spec),step=Math.min(Number($('gate-step').value),gates.length);
  const width=110+spec.reps*(2+QuantumEngine.edges(spec.topology).length)*55;
  let body='';for(let q=0;q<4;q++)body+=`<text x="5" y="${41+q*35}" font-size="12" fill="#617384">q${q}</text><line x1="40" y1="${37+q*35}" x2="${width-15}" y2="${37+q*35}" stroke="#ccd9e8"/>`;
  gates.forEach((g,i)=>{
    const perLayer=8+QuantumEngine.edges(spec.topology).length,local=i%perLayer,col=local<4?0:local<8?1:2+local-8,x=65+(g.layer*(2+QuantumEngine.edges(spec.topology).length)+col)*55,y=37+g.q*35,color=i<step?BLUE:'#bacbdf';
    if(g.type==='ZZ')body+=`<g data-gate="${i+1}" role="button" tabindex="0" aria-label="Step through ZZ on q${g.q} and q${g.t}"><line x1="${x}" y1="${y}" x2="${x}" y2="${37+g.t*35}" stroke="${color}" stroke-width="2"/><circle cx="${x}" cy="${y}" r="6" fill="${color}"/><circle cx="${x}" cy="${37+g.t*35}" r="6" fill="${color}"/><text x="${x}" y="12" font-size="9" text-anchor="middle" fill="#617384">ZZ</text></g>`;
    else body+=`<g data-gate="${i+1}" role="button" tabindex="0" aria-label="Step through ${g.type} on q${g.q}"><rect x="${x-17}" y="${y-12}" width="34" height="24" rx="4" fill="${i<step?'#e2edff':'#f3f6fa'}" stroke="${color}"/><text x="${x}" y="${y+4}" font-size="10" text-anchor="middle" fill="${color}">${g.type}</text></g>`;
  });
  $('live-circuit').innerHTML=svg(body,`0 0 ${width} 165`,'Four-qubit feature map. Select a gate or use Next gate to inspect the state after that operation.').replace('<svg ','<svg class="circuit-svg" ');
  $('gate-label').textContent=step===0?'|0000⟩':`${step} / ${gates.length}: ${gates[step-1].type} on q${gates[step-1].q}`;
  $('circuit-next').disabled=step===gates.length;
  const state=QuantumEngine.state(scaled(livePoints[0]),spec,step),phase=$('state-mode').value==='phase';
  const values=state.map(([re,im])=>re*re+im*im),max=Math.max(...values,.001);
  $('state-bars').innerHTML='<div class="state-bars">'+state.map(([re,im],i)=>{const p=values[i],arg=Math.atan2(im,re),height=phase?(p>1e-12?Math.abs(arg)/Math.PI*100:0):p/max*100,color=phase?`hsl(${(arg+Math.PI)/(2*Math.PI)*300},65%,48%)`:BLUE;return `<div class="state-bar" title="${i.toString(2).padStart(4,'0')}: probability ${p.toFixed(6)}, phase ${arg.toFixed(4)} radians"><small>${phase?arg.toFixed(1):percent(p)}</small><i style="height:${Math.max(1,height*.9)}px;background:${color}"></i><span>${i.toString(2).padStart(4,'0')}</span></div>`;}).join('')+'</div>';
}

for(const id of ['live-reps','live-topology'])$(id).addEventListener('change',()=>renderLive(true));
$('live-scale').addEventListener('input',()=>renderLive(true));
for(const [id,axis] of [['live-x',0],['live-y',1]])$(id).addEventListener('input',()=>{livePoints[activePoint][axis]=Number($(id).value);renderLive();});
for(const [id,index] of [['choose-a',0],['choose-b',1]])$(id).addEventListener('click',()=>{activePoint=index;renderLive();});
$('same-point').addEventListener('click',()=>{livePoints[1]=livePoints[0].slice();renderLive();});
$('reset-points').addEventListener('click',()=>initLive());
$('primary-settings').addEventListener('click',()=>{$('live-reps').value=data.featureMap.reps;$('live-scale').value=data.featureMap.scale;$('live-topology').value=data.featureMap.topology;renderLive(true);});
$('decision-background').addEventListener('change',()=>renderLive());
$('gate-step').addEventListener('input',renderCircuit);$('state-mode').addEventListener('change',renderCircuit);
$('circuit-reset').addEventListener('click',()=>{$('gate-step').value=0;renderCircuit();});
$('circuit-next').addEventListener('click',()=>{$('gate-step').value=Number($('gate-step').value)+1;renderCircuit();});
$('circuit-finish').addEventListener('click',()=>{$('gate-step').value=$('gate-step').max;renderCircuit();});
$('sample-shots').addEventListener('click',()=>{const shots=Number($('live-shots').value),p=QuantumEngine.fidelity(...liveStates),seed=sampleSeed++,count=QuantumEngine.sample(p,shots,seed);$('sample-result').innerHTML=`<strong>${number(count)} / ${number(shots)} zeros</strong><br>Estimate ${(count/shots).toFixed(5)} · exact ${p.toFixed(5)}<br>Shot SE ${Math.sqrt(p*(1-p)/shots).toFixed(5)} · seed ${seed}`;});
$('live-shots').addEventListener('change',()=>{$('sample-result').textContent='Sample this overlap at the new shot budget.';});
const livePlot=$('live-moons');
livePlot.addEventListener('pointerdown',event=>{const handle=event.target.closest('[data-handle]');if(!handle)return;activePoint=Number(handle.dataset.handle);dragging=event.pointerId;livePlot.setPointerCapture(event.pointerId);renderLive();event.preventDefault();});
livePlot.addEventListener('pointermove',event=>{if(dragging!==event.pointerId)return;const rect=livePlot.querySelector('svg').getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width*560,y=(event.clientY-rect.top)/rect.height*365;livePoints[activePoint]=[Math.max(-1.4,Math.min(2.4,(x-45)/475*3.8-1.4)),Math.max(-.8,Math.min(1.5,(307-y)/270*2.3-.8))];renderLive();});
for(const type of ['pointerup','pointercancel'])livePlot.addEventListener(type,()=>{dragging=null;});
function activate(event,selector,action){if(event.type==='keydown'&&!['Enter',' '].includes(event.key))return;const target=event.target.closest(selector);if(target){if(event.type==='keydown')event.preventDefault();action(target);}}
for(const type of ['click','keydown']){
  livePlot.addEventListener(type,e=>activate(e,'[data-handle]',t=>{activePoint=Number(t.dataset.handle);renderLive();}));
  $('live-circuit').addEventListener(type,e=>activate(e,'[data-gate]',t=>{$('gate-step').value=t.dataset.gate;renderCircuit();}));
  $('moons').addEventListener(type,e=>activate(e,'[data-train-row]',t=>{$('pair-row').value=t.dataset.trainRow;updatePair();}));
  $('robust-chart').addEventListener(type,e=>activate(e,'[data-split]',t=>{$('robust-seed').value=t.dataset.split;renderSplit();}));
  $('study-chart').addEventListener(type,e=>activate(e,'[data-study-row]',t=>{const r=studyRows[Number(t.dataset.studyRow)];$('study-detail').textContent=Object.entries(r).filter(([k])=>!['fold_scores','test_predictions'].includes(k)).map(([k,v])=>`${k.replaceAll('_',' ')}: ${typeof v==='number'?Number(v.toFixed(4)):v}`).join(' / ');}));
}
$('matrix-mode').addEventListener('change',renderHeatmap);$('compiler-metric').addEventListener('change',renderCompilerComparison);
$('robust-seed').addEventListener('change',renderSplit);$('study-device').addEventListener('change',renderStudy);
document.querySelectorAll('[data-study]').forEach(b=>b.addEventListener('click',()=>{study=b.dataset.study;renderStudy();}));
$('topology-atlas').addEventListener('click',e=>{const t=e.target.closest('[data-architecture]');if(t){selectedArchitecture=t.dataset.architecture;renderArchitectures();}});
$('figure-gallery').addEventListener('click',e=>{const t=e.target.closest('[data-figure]');if(!t)return;const g=data.gallery[Number(t.dataset.figure)];$('figure-title').textContent=g.name;$('figure-image').src=g.path;$('figure-image').alt=g.name;$('figure-download').href=g.path;$('figure-dialog').showModal();});
$('close-figure').addEventListener('click',()=>$('figure-dialog').close());
updateNavigation();load();
