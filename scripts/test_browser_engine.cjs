/* Validate actual browser mathematics against Qiskit and the exported sklearn model. */
const fs=require('node:fs'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'..'),Q=require(path.join(root,'frontend/quantum.js'));
const data=JSON.parse(fs.readFileSync(path.join(root,'frontend/data/experiment.json'),'utf8'));
let checks=0;
for(const row of data.browserValidation){
  const states=row.points.map(p=>Q.state(p,row.spec));
  states.forEach((s,i)=>s.forEach((a,j)=>a.forEach((v,k)=>assert(Math.abs(v-row.states[i][j][k])<1e-12))));
  assert(Math.abs(Q.fidelity(...states)-row.fidelity)<1e-12);checks++;
}
data.dataset.testIndices.forEach((index,i)=>{
  const point=data.dataset.X[index].map((v,k)=>v*data.liveModel.scalerScale[k]+data.liveModel.scalerMin[k]);
  const score=Q.decision(Q.state(point,data.featureMap),data.liveModel);
  assert.equal(score>=0?1:0,data.liveModel.testPredictions[i]);checks++;
});
assert.equal(Q.sample(0,4096,42),0);assert.equal(Q.sample(1,4096,42),4096);
assert.equal(Q.sample(.5,4096,42),Q.sample(.5,4096,42));
console.log(`${checks} Qiskit state/fidelity and sklearn prediction cases passed; sampling boundary checks passed.`);
