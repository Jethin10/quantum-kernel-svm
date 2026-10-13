/* Exact 16-amplitude statevector simulation. Qubit 0 is the least-significant bit. */
(function(root){
  'use strict';
  const edges = name => name==='linear'?[[0,1],[1,2],[2,3]]:name==='ring'?[[0,1],[1,2],[2,3],[3,0]]:[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
  function instructions(point,spec){
    const [a,b]=point,coordinates=spec.encoding==='mean_difference'?[a,b,(a+b)/2,(a-b)/2]:[a,b,a,b];
    const theta=coordinates.map(v=>spec.scale*v),result=[];
    for(let layer=0;layer<spec.reps;layer++){
      for(let q=0;q<4;q++)result.push({type:'H',q,layer});
      for(let q=0;q<4;q++)result.push({type:'RZ',q,angle:theta[q],layer});
      for(const [q,t] of edges(spec.topology))result.push({type:'ZZ',q,t,angle:theta[q]*theta[t],layer});
    }
    return result;
  }
  function state(point,spec,steps=Infinity){
    const amplitudes=Array.from({length:16},()=>[0,0]);amplitudes[0][0]=1;
    const gates=instructions(point,spec).slice(0,steps);
    for(const gate of gates){
      if(gate.type==='H'){
        const bit=1<<gate.q;
        for(let i=0;i<16;i++)if(!(i&bit)){
          const j=i|bit,a=amplitudes[i],b=amplitudes[j];
          amplitudes[i]=[(a[0]+b[0])/Math.SQRT2,(a[1]+b[1])/Math.SQRT2];
          amplitudes[j]=[(a[0]-b[0])/Math.SQRT2,(a[1]-b[1])/Math.SQRT2];
        }
      }else{
        for(let i=0;i<16;i++){
          const parity=gate.type==='RZ'?((i>>gate.q)&1):(((i>>gate.q)^(i>>gate.t))&1);
          const angle=gate.angle*(parity?1:-1),c=Math.cos(angle),s=Math.sin(angle),[re,im]=amplitudes[i];
          amplitudes[i]=[re*c-im*s,re*s+im*c];
        }
      }
    }
    return amplitudes;
  }
  function fidelity(a,b){
    let re=0,im=0;
    for(let i=0;i<16;i++){re+=a[i][0]*b[i][0]+a[i][1]*b[i][1];im+=a[i][0]*b[i][1]-a[i][1]*b[i][0];}
    return Math.max(0,Math.min(1,re*re+im*im));
  }
  function decision(amplitudes,model){return model.intercept+model.supportStates.reduce((sum,s,i)=>sum+model.dual[i]*fidelity(amplitudes,s),0);}
  function sample(p,shots,seed){
    let value=seed>>>0,count=0;
    for(let i=0;i<shots;i++){
      value+=0x6D2B79F5;let t=value;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);
      if(((t^(t>>>14))>>>0)/4294967296<p)count++;
    }
    return count;
  }
  const api={edges,instructions,state,fidelity,decision,sample};
  if(typeof module==='object'&&module.exports)module.exports=api;else root.QuantumEngine=api;
})(globalThis);
