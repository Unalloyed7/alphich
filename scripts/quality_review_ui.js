'use strict';
// Never expose fixture predictions or other reviewers' answers before authorization.
const $ = id => document.getElementById(id);
let token = null, currentUser = null, rows = [], selected = null;
const stateNames = {todo:'Не проверено',saved:'Сохранено',awaiting_reviews:'Ожидает экспертов',disagreement:'Разногласие',agreed:'Согласовано',adjudicated:'Арбитраж'};
const options = {
  geography: [['unknown','Не установлена'],['national','Вся страна'],['global_market','Мировой рынок'],['confirmed_operations','Операции подтверждены'],['registration_only','Только место регистрации'],['other_region','Другой регион'],['not_applicable','Не применимо']],
  scale: [['unknown','Масштаб не установлен'],['material','Существенное изменение'],['limited','Ограниченный масштаб'],['none','Нет влияния']],
  applicability: [['unknown','Не установлена'],['confirmed','Прямо подтверждена'],['conditional','Зависит от условий'],['not_confirmed','Не подтверждена']]
};
function el(tag,cls='',text='') {const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function put(parent,tag,cls,text){const e=el(tag,cls,text);parent.append(e);return e;}
function pill(value){const span=el('span','pill '+value,stateNames[value]||value);return span;}
async function req(url,data){const method=data?'POST':'GET';const r=await fetch(url,{method,headers:{'X-Review-Token':token,...(data?{'Content-Type':'application/json'}:{})},body:data?JSON.stringify(data):undefined,credentials:'omit',cache:'no-store'});let j=await r.json();if(!r.ok)throw Error(j.error||'HTTP '+r.status);return j;}
function errorAt(root,message){const box=el('div','error',message);root.prepend(box);return box;}
function loginState(authed){$('login').classList.toggle('hidden',authed);$('workspace').classList.toggle('hidden',!authed);$('identity').classList.toggle('hidden',!authed);}
async function login(t){token=t;try{const data=await req('/api/cases');currentUser={name:data.user,role:data.role};$('user-name').textContent=`${data.user} · ${data.role==='arbiter'?'Арбитр':'Эксперт'}`;loginState(true);loadSummary(data)}catch(e){token=null;$('login-message').replaceChildren(el('div','error',e.message));}}
$('login-form').addEventListener('submit',e=>{e.preventDefault();void login($('token').value.trim());$('token').value='';});
$('logout').addEventListener('click',()=>{token=null;rows=[];selected=null;currentUser=null;loginState(false);$('detail').replaceChildren(el('div','empty','Выберите новость и компанию слева.'));});
function loadSummary(data){rows=data.items;const companies=[...new Set(rows.map(r=>r.company_id))].sort();$('company').replaceChildren(el('option','', 'Все компании'));$('company').firstChild.value='';companies.forEach(c=>{const opt=put($('company'),'option','',c);opt.value=c;});$('total').textContent=String(data.total);const done=currentUser.role==='reviewer'?(data.counts.saved||0):(data.counts.adjudicated||0)+(data.counts.agreed||0);$('stats').textContent=currentUser.role==='reviewer'?`Сохранено ${done} из ${data.total}`:`Завершено ${done} из ${data.total} · Разногласия: ${data.counts.disagreement||0}`;$('progress').style.width=(100*done/(data.total||1))+'%';renderList();}
function renderList(){const holder=$('case-list');holder.replaceChildren();const q=$('search').value.toLowerCase().trim(),ds=$('dataset').value,st=$('status').value,co=$('company').value;const chosen=rows.filter(x=>(!ds||x.dataset===ds)&&(!st||x.state===st)&&(!co||x.company_id===co)&&(!q||(x.title+' '+x.company_id+' '+x.event_id).toLowerCase().includes(q)));for(const row of chosen){const b=el('button','case-button'+(row.id===selected?' active':''));b.type='button';const h=put(b,'div','row');put(h,'span','meta',row.dataset+' · '+row.company_id);h.append(pill(row.state));put(b,'span','name',row.title);put(b,'span','meta',row.event_id);b.onclick=()=>void showCase(row.id);holder.append(b);}if(!chosen.length)put(holder,'p','muted tiny','Нет подходящих пар');}
['search','dataset','status','company'].forEach(id=>$(id).addEventListener('input',renderList));
function addSection(root,title){return put(root,'h3','section-title',title);}
function field(root,title,name,value='',type='textarea'){
 const label=put(root,'label','field');put(label,'span','',title);let input;
 if(type==='textarea'){input=put(label,'textarea');input.rows=3;input.maxLength=2000;input.value=value||'';}
 else if(type==='number'){input=put(label,'select');for(let n=0;n<=5;n++){let opt=put(input,'option','',String(n));opt.value=String(n);}input.value=value!=null?String(value):'0';}
 else {input=put(label,'select');for(const [val,text] of options[name]){let opt=put(input,'option','',text);opt.value=val;}input.value=value||'unknown';}
 input.name=name;input.required=true;return input;
}
function reviewForm(root, id, initial, adjudicating){
 const form=put(root,'form');form.className='review-form';
 const top=put(form,'div','form-grid');field(top,'Значимость 0–5','grade',initial?.grade??0,'number');field(top,'География','geography',initial?.geography||'unknown','select');field(top,'Масштаб','scale',initial?.scale||'unknown','select');field(top,'Применимость','applicability',initial?.applicability||'unknown','select');
 field(form,'Почему именно эта оценка?','justification',initial?.justification||'');
 const dims=put(form,'div','form-grid');field(dims,'Ограничения географии','geographic_limit',initial?.geographic_limit||'Не установлено');field(dims,'Ограничения масштаба','scale_limit',initial?.scale_limit||'Не установлено');field(dims,'Ограничения применимости','applicability_limit',initial?.applicability_limit||'Не установлено');
 put(form,'p','hint','Это человеческая оценка. Не копируйте первоначальную разметку AI. При отсутствии доказательств явно укажите пробел.');
 const actions=put(form,'div','actions');const save=put(actions,'button','primary',adjudicating?'Зафиксировать окончательное решение':'Сохранить мою оценку');save.type='submit';const msg=put(actions,'span','muted tiny','');
 form.onsubmit=async event=>{event.preventDefault();save.disabled=true;msg.textContent='Сохранение…';const data={id};for(const e of form.elements){if(e.name)data[e.name]=e.name==='grade'?Number(e.value):e.value;}
 try{await req(adjudicating?'/api/adjudicate':'/api/review',data);await refresh(id);msg.textContent='Сохранено';}catch(err){errorAt(form,err.message);save.disabled=false;msg.textContent='';}};
 return form;
}
function reviewCard(root,row){const box=put(root,'div','review');const head=put(box,'div','row');put(head,'strong','',row.reviewer||row.arbiter||'Арбитр');put(head,'span','pill',row.grade+'/5');put(box,'p','','Обоснование: '+row.justification);put(box,'p','muted','География: '+row.geography+' · Масштаб: '+row.scale+' · Применимость: '+row.applicability);put(box,'p','muted','Ограничения: '+[row.geographic_limit,row.scale_limit,row.applicability_limit].join(' / '));put(box,'p','tiny muted',row.created_at+' · ревизия '+row.revision);return box;}
async function refresh(id){const summary=await req('/api/cases');loadSummary(summary);await showCase(id);}
async function showCase(id){selected=id;renderList();const root=$('detail');root.replaceChildren(el('p','muted','Загружаем…'));try{
 const c=await req('/api/case?id='+encodeURIComponent(id));root.replaceChildren();const header=put(root,'div','row');put(header,'span','eyebrow',c.dataset+' · '+c.company_id);header.append(pill(c.state));put(root,'h2','title',c.title);put(root,'p','meta-line','ID: '+c.event_id+' · Дата публикации: '+(c.published_at||'не предоставлена в исходной разметке'));
 if(c.url&&c.url.startsWith('https://')){const a=put(root,'a','','Открыть первоисточник ↗');a.href=c.url;a.target='_blank';a.rel='noopener noreferrer';}
 addSection(root,'Текст новости');if(c.article_text)put(root,'div','source-box',c.article_text);else put(root,'div','notice warn','Полный текст не включён в v2/v3. Откройте первоисточник; для архивируемой экспертизы загрузите проверенный локальный снимок статьи.');
 addSection(root,'Подтверждённые признаки компании');
 if(c.company_context&&Array.isArray(c.company_context.features)){
  put(root,'p','tiny muted','Проверено: '+c.company_context.verified_at+'. Признаки загружены отдельно от исходных AI-оценок.');const grid=put(root,'div','features');
  for(const f of c.company_context.features){const item=put(grid,'div','feature');put(item,'strong','',f.name);put(item,'pre','',typeof f.value==='string'?f.value:JSON.stringify(f.value,null,2));put(item,'small','muted','Источник: '+f.source_ref);}
 }else put(root,'div','notice warn','Проверенный контекст компании не загружен. Исходные AI-обоснования и подсказки НЕ показываются. Без признаков компании обосновать высокую применимость может быть невозможно.');
 root.append(el('hr','divider'));
 if(currentUser.role==='reviewer'){
   addSection(root,c.own?'Редактировать свою оценку (сохранится новая ревизия)':'Ваша независимая оценка');
   if(c.own)put(root,'p','success','Предыдущая оценка сохранена. Оценки другого эксперта и AI не показываются.');
   reviewForm(root,id,c.own,false);
 }else if(c.review_count<2){addSection(root,'Арбитраж');put(root,'div','notice','Нужно дождаться обеих независимых оценок.');}
 else{addSection(root,c.state==='agreed'?'Оценки совпали':'Разбор двух независимых оценок');const grid=put(root,'div','split');for(const r of c.reviews)reviewCard(grid,r);
 if(c.adjudication){addSection(root,'Окончательное решение');reviewCard(root,c.adjudication);}
 else if(c.state==='agreed'){
   put(root,'p','success','Два эксперта согласны по баллу и трём измерениям. Эти оценки можно экспортировать как согласованный итог без арбитража.');
   const details=put(root,'details');put(details,'summary','','Открыть дополнительный арбитраж, если объяснения существенно отличаются');
   reviewForm(details,id,c.reviews[0],true);
 }
 else{addSection(root,'Решение арбитра');put(root,'p','muted','Разногласия разрешаются только после двух сохранённых оценок.');reviewForm(root,id,null,true);}
 }
 }catch(e){root.replaceChildren();errorAt(root,e.message);}}
