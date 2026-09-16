(function(){
  "use strict";
  var $=function(s,r){return (r||document).querySelector(s);};
  function all(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s));}

  // The page has one appearance and no toggle. A stale 'pbl-theme' key from
  // the old two-mode build is left where it is: nothing reads it, and clearing
  // it would mean shipping code whose only job is to undo code that is gone.

  // ---------- mobile sidebar ----------
  var side=$('#pbl-side'), bd=$('#pbl-backdrop'), menu=$('#pbl-menu');
  function closeSide(){ if(side)side.classList.remove('open'); if(bd)bd.classList.remove('show'); }
  if(menu) menu.addEventListener('click',function(){ side.classList.add('open'); bd.classList.add('show'); });
  if(bd) bd.addEventListener('click',closeSide);

  // ---------- sidebar icons ----------
  function icon(t){
    t=(t||'').toLowerCase();
    function s(p){return '<svg class="icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'+p+'</svg>';}
    if(/introduction|overview/.test(t)) return s('<circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/>');
    if(/getting started/.test(t)) return s('<path d="M5 3l14 9-14 9V3z"/>');
    if(/architecture/.test(t)) return s('<path d="M3 21V8l9-5 9 5v13"/>');
    if(/gui|tour/.test(t)) return s('<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8"/>');
    if(/project/.test(t)) return s('<path d="M3 7h6l2 2h10v10H3z"/>');
    if(/board|mcu/.test(t)) return s('<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M6 14h.01M10 12h8"/>');
    if(/camera|zone|set up/.test(t)) return s('<path d="M4 21v-6M4 11V3M12 21v-9M12 8V3M20 21v-4M20 13V3M1 15h6M9 8h6M17 17h6"/>');
    if(/track/.test(t)) return s('<circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>');
    if(/record/.test(t)) return s('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/>');
    if(/breakout/.test(t)) return s('<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9zM9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>');
    if(/audio/.test(t)) return s('<path d="M11 5L6 9H2v6h4l5 4V5zM15 9a5 5 0 0 1 0 6"/>');
    if(/micro|mic/.test(t)) return s('<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v4"/>');
    if(/photometry/.test(t)) return s('<path d="M3 12h4l3 8 4-16 3 8h4"/>');
    if(/writing|task api|task/.test(t)) return s('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>');
    if(/recipe|use case/.test(t)) return s('<path d="M4 5h16M4 12h16M4 19h10"/>');
    if(/api|schema|pipeline|pycboard|recorder|gui\//.test(t)) return s('<path d="M8 6l-6 6 6 6M16 6l6 6-6 6"/>');
    if(/concept|frame|time|lineage|snapshot/.test(t)) return s('<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6h4a4 4 0 0 1 4 4M8 18h4a4 4 0 0 0 4-4"/>');
    if(/reference|format|setting|glossary|env/.test(t)) return s('<path d="M4 19V5a2 2 0 0 1 2-2h10l4 4v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/>');
    if(/test|build|contribut|dev|agent/.test(t)) return s('<path d="M14.7 6.3a4 4 0 0 0-5 5L3 18v3h3l6.7-6.7a4 4 0 0 0 5-5z"/>');
    if(/troubleshoot/.test(t)) return s('<path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>');
    return s('<circle cx="12" cy="12" r="1.4"/><circle cx="12" cy="6" r="1.4"/><circle cx="12" cy="18" r="1.4"/>');
  }
  all('.nav a.reference').forEach(function(a){
    if(a.closest('li') && a.closest('li').classList.contains('toctree-l1')){
      a.insertAdjacentHTML('afterbegin', icon(a.textContent));
    }
  });

  // ---------- command palette ----------
  var pal=$('#pbl-palette'), palin=$('#pbl-palin'), palres=$('#pbl-palres');
  var pages=all('.nav a.reference').map(function(a){return {t:a.textContent.trim().replace(/\s+/g,' '), h:a.getAttribute('href')};});
  var sel=0, filt=pages;
  function render(){
    palres.innerHTML = filt.length ? filt.map(function(it,i){
      return '<div class="r'+(i===sel?' sel':'')+'" data-h="'+it.h+'">'+it.t+'<span class="rs">Docs</span></div>';
    }).join('') : '<div style="padding:22px;text-align:center;color:var(--fade)">No results</div>';
    all('.pbl-pal .r').forEach(function(r){ r.addEventListener('click',function(){ location.href=r.getAttribute('data-h'); }); });
  }
  function openPal(){ pal.classList.add('open'); palin.value=''; sel=0; filt=pages; render(); setTimeout(function(){palin.focus();},30); }
  function closePal(){ pal.classList.remove('open'); }
  var sbtn=$('#pbl-search'); if(sbtn) sbtn.addEventListener('click',openPal);
  if(pal) pal.addEventListener('click',function(e){ if(e.target===pal) closePal(); });
  if(palin) palin.addEventListener('input',function(){
    var q=palin.value.toLowerCase(); filt=pages.filter(function(p){return p.t.toLowerCase().indexOf(q)>=0;}); sel=0; render();
  });
  document.addEventListener('keydown',function(e){
    if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){ e.preventDefault(); pal.classList.contains('open')?closePal():openPal(); return; }
    if(!pal||!pal.classList.contains('open')) return;
    if(e.key==='Escape'){ closePal(); closeSide(); }
    if(e.key==='ArrowDown'){ e.preventDefault(); sel=Math.min(filt.length-1,sel+1); render(); }
    if(e.key==='ArrowUp'){ e.preventDefault(); sel=Math.max(0,sel-1); render(); }
    if(e.key==='Enter'&&filt[sel]){ location.href=filt[sel].h; }
  });

  // ---------- unwrap the page-title node so h2s become the top-level toc ----------
  var tocBox=$('#pbl-toc');
  if(tocBox){
    var tops=tocBox.querySelectorAll(':scope > ul > li');
    if(tops.length===1){
      var inner=tops[0].querySelector(':scope > ul');
      if(inner){ tocBox.innerHTML=''; tocBox.appendChild(inner); }
    }
  }

  // ---------- scrollspy + progress + ring ----------
  var tocLinks = all('#pbl-toc a.reference');
  var heads = tocLinks.map(function(a){
    var id=a.getAttribute('href'); if(!id||id.charAt(0)!=='#') return null;
    try{ return document.querySelector(id); }catch(e){ return null; }
  });
  var titles = tocLinks.map(function(a){return a.textContent.trim();});
  var bar=$('#pbl-progress'), ring=$('#pbl-ring'), ringp=$('#pbl-ringp'), cur=$('#pbl-cur');
  var C=94;
  function spy(){
    var y=window.scrollY, h=document.documentElement.scrollHeight-window.innerHeight, p=h>0?Math.min(1,y/h):0;
    if(bar) bar.style.width=(p*100)+'%';
    if(ring) ring.style.strokeDashoffset=C*(1-p);
    if(ringp) ringp.textContent=Math.round(p*100)+'%';
    var i=-1;
    for(var k=0;k<heads.length;k++){ if(heads[k] && heads[k].getBoundingClientRect().top<140) i=k; }
    tocLinks.forEach(function(l,k){ l.classList.toggle('pbl-active',k===i); });
    if(cur) cur.textContent = i>=0 ? titles[i] : (titles[0]||'');
  }
  tocLinks.forEach(function(l){ l.addEventListener('click',function(){ closeSide(); }); });
  var top=$('#pbl-top'); if(top) top.addEventListener('click',function(){ window.scrollTo({top:0,behavior:'smooth'}); });
  window.addEventListener('scroll',spy,{passive:true});
  window.addEventListener('resize',spy);
  spy();
})();
