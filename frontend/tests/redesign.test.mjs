import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { mockApi } from './fixtures.mjs';
const baseURL=process.env.UI_BASE_URL || 'http://127.0.0.1:5174';
const screenshotDir=process.env.UI_SCREENSHOT_DIR;
let browser;
before(async()=>{browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHANNEL ? {channel:process.env.PLAYWRIGHT_CHANNEL} : {})});});
after(async()=>{await browser?.close();});
async function setup(options={},viewport={width:1440,height:1050}) {
  const context=await browser.newContext({viewport});const page=await context.newPage();
  if(options.clock) await page.clock.install();
  const errors=[];page.on('pageerror',e=>errors.push(e.message));const state=await mockApi(page,options);await page.goto(baseURL);return {context,page,state,errors};
}
async function login(page) {await page.getByRole('button',{name:'Перейти к аналитике'}).click();await page.locator('.briefing-layout').waitFor();await page.locator('.feed-skeleton').waitFor({state:'hidden'});}
async function screenshot(page,name,fullPage){if(screenshotDir){await fs.mkdir(screenshotDir,{recursive:true});await page.screenshot({path:path.join(screenshotDir,name),fullPage:fullPage??!(await page.getByRole('dialog').count())});}}
async function accessible(page){const result=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(result.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))})),[]);}
async function cardCount(page,count) {await page.waitForFunction(expected=>document.querySelectorAll('.brief-card').length===expected,count);assert.equal(await page.locator('.brief-card').count(),count);}
async function noOverflow(page) {assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);}

test('A slow feed request is shared, 504 keeps the last feed and automatic retries back off',async()=>{
 const {context,page}=await setup({clock:true});
 let release;
 try {
  await login(page);await cardCount(page,3);
  let attempts=0,markStarted;
  const started=new Promise(resolve=>{markStarted=resolve;});
  const blocked=new Promise(resolve=>{release=resolve;});
  await page.route('**/api/v1/companies/*/feed',async route=>{
   attempts++;
   if(attempts===1) {
    markStarted();await blocked;
    await route.fulfill({status:504,contentType:'text/html',body:'<h1>504 Gateway Time-out</h1>'});
   } else await route.fallback();
  });
  await page.clock.fastForward(20_000);await started;
  await page.clock.fastForward(40_000);
  await page.getByRole('button',{name:'Обновить аналитику',exact:true}).click();
  assert.equal(attempts,1); // Timer ticks and a manual refresh share the pending read.
  release();
  const alert=page.getByRole('alert').filter({hasText:'Сервер не успел подготовить данные'});
  await alert.waitFor();await cardCount(page,3);
  await page.clock.fastForward(20_000);
  assert.equal(attempts,1); // No immediate automatic retry after the timeout.
  await alert.getByRole('button',{name:'Повторить',exact:true}).click();
  await alert.waitFor({state:'hidden'});
  assert.equal(attempts,2); // An explicit retry bypasses the cooldown.
  await cardCount(page,3);
 } finally {release?.();await context.close();}
});

test('Briefing prioritizes insights, filters correctly, and never starts paid work on load',async()=>{
 const {context,page,state,errors}=await setup();try {
  await login(page);await cardCount(page,3);assert.deepEqual(state.requests.filter(r=>r.method==='POST').map(r=>r.path),['/demo-session']);await screenshot(page,'desktop-synthetic.png');
  await page.getByRole('button',{name:'Значимость 5/5',exact:true}).click();await cardCount(page,1);
  await page.getByRole('button',{name:'С позицией банка',exact:true}).click();await cardCount(page,1);
  await page.getByRole('button',{name:'Все',exact:true}).click();await cardCount(page,3);
  await page.getByRole('button',{name:'Вся персональная аналитика'}).click();
  await page.getByRole('searchbox',{name:'Поиск по персональной ленте'}).fill('Стоимость');await cardCount(page,1);
  await page.getByRole('searchbox',{name:'Поиск по персональной ленте'}).fill('нет_результатов');await page.getByRole('heading',{name:'Нет событий по заданным условиям'}).waitFor();
  await page.getByRole('button',{name:'Сбросить фильтры'}).click();await cardCount(page,3);await noOverflow(page);assert.deepEqual(errors,[]);
 } finally {await context.close();}
});

test('Event interpretation, evidence, original source, question, focus trap and return focus',async()=>{
 const {context,page,state,errors}=await setup();try {
  await login(page);const opener=page.locator('.brief-card h3 button').first();await opener.click();const dialog=page.getByRole('dialog');await dialog.waitFor();await page.getByRole('heading',{name:'Что проверить дальше'}).waitFor();await screenshot(page,'event-synthetic.png');
  await page.getByRole('tab',{name:'Обоснование',exact:true}).click();await dialog.getByRole('button',{name:'Источник 1',exact:true}).first().click();await page.getByText('Подтверждающий фрагмент из тестового документа.',{exact:true}).waitFor();await page.getByRole('button',{name:'Закрыть подтверждающий фрагмент'}).click();
  await page.getByRole('tab',{name:'Первоисточники',exact:false}).click();
  const sourceButton=page.getByRole('button',{name:'Тестовый новостной источник',exact:true});
  const sourcePanel=page.getByRole('region',{name:'Текст первоисточника',exact:true});
  const sourceText=page.getByText('Полный текст тестового первоисточника.',{exact:false});
  const sourceRequests=()=>state.requests.filter(r=>r.path.startsWith('/sources/')).length;
  await sourceButton.click();await sourceText.waitFor();assert.equal(await sourceButton.getAttribute('aria-expanded'),'true');assert.equal(sourceRequests(),1);assert.equal(await page.evaluate(()=>window.injected),undefined);
  await sourceButton.click();await sourcePanel.waitFor({state:'hidden'});assert.equal(await sourceButton.getAttribute('aria-expanded'),'false');assert.equal(sourceRequests(),1);
  await sourceButton.click();await sourceText.waitFor();assert.equal(await sourceButton.getAttribute('aria-expanded'),'true');assert.equal(sourceRequests(),2);
  await page.getByRole('tab',{name:'Влияние на бизнес',exact:true}).click();await page.getByRole('tab',{name:'Вопросы по событию',exact:true}).click();await page.getByRole('button',{name:'Какие данные нужны для решения?',exact:true}).click();assert.equal(state.requests.filter(r=>r.path.endsWith('/questions')).length,0);
  await page.getByRole('button',{name:'Задать вопрос',exact:true}).click();await page.getByRole('heading',{name:'Какие данные нужны для решения?',exact:true}).waitFor();const question=state.requests.find(r=>r.path.endsWith('/questions'));assert.ok(question.headers['idempotency-key']);assert.equal(question.headers['x-demo-session'],'test-DEMO_A');await accessible(page);
  await page.getByRole('button',{name:'Закрыть событие'}).focus();await page.keyboard.press('Shift+Tab');assert.equal(await page.locator(':focus').evaluate(el=>!!el.closest('[role=dialog]')),true);
  await page.keyboard.press('Escape');await dialog.waitFor({state:'hidden'});assert.equal(await opener.evaluate(el=>document.activeElement===el),true);assert.deepEqual(errors,[]);
 } finally {await context.close();}
});

test('Responsive layouts and primary screens meet automated WCAG checks',async()=>{
 const {context,page,errors}=await setup();try {
  await accessible(page);await login(page);await accessible(page);
  for(const width of [1440,1024,768,390,320]) {await page.setViewportSize({width,height:844});await noOverflow(page);if(width===390){await screenshot(page,'mobile-synthetic.png');await accessible(page);}}
  await page.getByRole('button',{name:'Открыть быстрый поиск',exact:true}).click();await page.getByRole('dialog',{name:'Быстрый поиск'}).waitFor();await noOverflow(page);await accessible(page);await screenshot(page,'mobile-search.png');await page.getByRole('button',{name:'Закрыть поиск'}).click();
  await page.locator('.brief-card h3 button').first().click();await page.getByRole('heading',{name:'Оценка анализа'}).waitFor();await noOverflow(page);await accessible(page);await screenshot(page,'mobile-event.png');await page.getByRole('button',{name:'Закрыть событие'}).click();await page.getByRole('dialog').waitFor({state:'hidden'});
  await page.getByRole('button',{name:'Моя компания',exact:true}).click();await page.getByRole('heading',{name:'Факторы влияния на компанию'}).waitFor();await page.getByText('Подтверждённые признаки и их значение',{exact:false}).click();await page.getByText('Признак отсутствует',{exact:true}).waitFor();await noOverflow(page);await accessible(page);
  await page.getByRole('button',{name:'Добавить новость',exact:true}).click();await page.getByRole('heading',{name:'Добавить публикацию',exact:true}).waitFor();await noOverflow(page);await accessible(page);assert.deepEqual(errors,[]);
 } finally {await context.close();}
});

test('Empty, failed and unconfigured feed states remain actionable',async()=>{
 for(const options of [{empty:true},{feedError:true},{unconfigured:true}]) {
  const {context,page,state}=await setup(options);try {
   await login(page);if(options.empty) await page.getByRole('heading',{name:'Событий для компании пока нет'}).waitFor();
   if(options.feedError) {await page.getByRole('alert').filter({hasText:'Не удалось обновить ленту'}).waitFor();state.feedError=false;await page.getByRole('button',{name:'Повторить',exact:true}).click();await page.locator('.brief-card').first().waitFor();}
   if(options.unconfigured) {assert.equal(await page.getByRole('button',{name:'Подобрать новости'}).isDisabled(),true);await page.getByText('Подбор недоступен:',{exact:false}).waitFor();}
   assert.equal(state.requests.filter(r=>r.method==='POST' && r.path!=='/demo-session').length,0);
  } finally {await context.close();}
 }
});

test('Generation, discovery and publication are explicit actions; company switch clears session data',async()=>{
 const {context,page,state}=await setup({noInsight:true});try {
  await login(page);await page.getByRole('button',{name:'Подобрать новости'}).click();await page.getByText('Новых подходящих публикаций',{exact:false}).waitFor();
  await page.locator('.brief-card h3 button').first().click();await page.getByRole('button',{name:'Подготовить анализ',exact:true}).click();await page.getByRole('heading',{name:'Что проверить дальше'}).waitFor();await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'Добавить новость',exact:true}).click();await page.getByLabel('Заголовок',{exact:false}).fill('Тестовая публикация для проверки формы');await page.getByLabel('Текст публикации',{exact:false}).fill('Это синтетический текст публикации для браузерной проверки интерфейса, без реального вызова модели.');await page.getByRole('button',{name:'Отправить на обработку',exact:true}).click();await page.getByText('Публикация отправлена.',{exact:false}).waitFor();
  for(const endpoint of ['/publications','/companies/DEMO_A/insights','/companies/DEMO_A/discovery']) {const req=state.requests.find(r=>r.path===endpoint && r.method==='POST');assert.ok(req,endpoint);assert.ok(req.headers['idempotency-key']);}
  await page.getByRole('button',{name:'Сменить',exact:true}).click();await page.getByLabel('Идентификатор компании').fill('DEMO_B');await page.getByRole('combobox',{name:'Ваша роль'}).selectOption('ceo');await login(page);await page.getByRole('heading',{name:'Стратегические вопросы',exact:false}).waitFor();assert.ok((await page.locator('.user-badge').innerText()).includes('DEMO_B'));
  const stored=JSON.parse(await page.evaluate(()=>sessionStorage.getItem('alphich.demo-session.v1')));assert.equal(stored.company_id,'DEMO_B');assert.equal(stored.role,'ceo');
 } finally {await context.close();}
});

test('Dashboard widgets use feed values and navigate to matching events',async()=>{
 const {context,page,state}=await setup();try {
  await login(page);await page.getByTestId('summary-total').filter({hasText:'03'}).waitFor();
  assert.equal(await page.getByTestId('summary-critical').innerText(),'01');
  assert.equal(await page.getByTestId('summary-saved').innerText(),'03');
  assert.equal(await page.getByTestId('summary-bank').innerText(),'1 из 3');
  assert.equal(await page.locator('.widget-actions button').count(),1); // repeated advice is shown once
  const reads=state.requests.filter(r=>r.path.endsWith('/feed')).length;
  await page.getByRole('button',{name:'Открыть события с позицией банка',exact:true}).click();await cardCount(page,1);
  assert.equal(await page.getByRole('button',{name:'С позицией банка',exact:true}).getAttribute('aria-pressed'),'true');
  assert.equal(state.requests.filter(r=>r.path.endsWith('/feed')).length,reads); // same data source across views
  await page.getByRole('button',{name:'Обзор',exact:true}).click();
  await page.getByRole('button',{name:'Открыть события со значимостью 5 из 5',exact:true}).click();await cardCount(page,1);
  assert.equal(await page.getByRole('button',{name:'Значимость 5/5',exact:true}).getAttribute('aria-pressed'),'true');
  await page.getByRole('button',{name:'Обзор',exact:true}).click();await page.locator('.widget-actions button').first().click();await page.getByRole('dialog').waitFor();await page.getByRole('heading',{name:'Что проверить дальше'}).waitFor();
  assert.equal(state.requests.filter(r=>r.method==='POST' && r.path!=='/demo-session').length,0);
 } finally {await context.close();}
});

test('Dashboard distinguishes empty, unavailable and ungraded data',async()=>{
 for(const options of [{empty:true},{feedError:true},{legacy:true}]) {
  const {context,page}=await setup(options);try {
   await login(page);
   if(options.empty) {assert.equal(await page.getByTestId('summary-total').innerText(),'00');assert.equal(await page.getByTestId('summary-bank').innerText(),'0 из 0');assert.equal(await page.locator('.widget-actions button').count(),0);}
   if(options.feedError) {assert.equal(await page.getByTestId('summary-total').innerText(),'—');assert.equal(await page.getByTestId('summary-critical').innerText(),'—');}
   if(options.legacy) {assert.equal(await page.getByTestId('summary-total').innerText(),'03');assert.equal(await page.getByTestId('summary-critical').innerText(),'00');await page.locator('.pulse-legend').getByText('Прочие',{exact:false}).waitFor();assert.equal(await page.locator('.widget-actions button').count(),0);}
  } finally {await context.close();}
 }
});

test('Bookmarks and reading state survive reloads and stay within the company and role',async()=>{
 const {context,page,state}=await setup();try {
  await login(page);await cardCount(page,3);
  await page.getByRole('button',{name:'Добавить в избранное:',exact:false}).first().click();
  await page.getByRole('button',{name:'Избранное',exact:true}).click();await cardCount(page,1);
  await page.reload();await cardCount(page,1);assert.equal(await page.getByRole('button',{name:'Убрать из избранного:',exact:false}).first().getAttribute('aria-pressed'),'true');
  await page.locator('.brief-card h3 button').first().click();await page.getByRole('dialog').waitFor();assert.ok(page.url().includes('event=event-1'));
  assert.equal(await page.locator(':focus').evaluate(el=>!!el.closest('[role=dialog]')),true);
  await page.goBack();await page.getByRole('dialog').waitFor({state:'hidden'});
  await page.goForward();await page.getByRole('dialog').waitFor();await page.getByRole('button',{name:'Закрыть событие'}).click();await page.getByRole('dialog').waitFor({state:'hidden'});
  await page.getByRole('button',{name:'Непрочитанные',exact:true}).click();await cardCount(page,2);
  await page.getByRole('button',{name:'Сменить',exact:true}).click();await page.getByLabel('Идентификатор компании').fill('DEMO_B');await page.getByRole('combobox',{name:'Ваша роль'}).selectOption('ceo');await login(page);await cardCount(page,3);
  await page.getByRole('button',{name:'Избранное',exact:true}).click();assert.equal(await page.locator('.brief-card').count(),0);
  assert.equal(state.requests.filter(r=>r.method==='POST' && r.path!=='/demo-session').length,0);
 } finally {await context.close();}
});

test('Quick search works with keyboard, explicit scope, empty results and accessible focus',async()=>{
 const {context,page}=await setup();try {
  await login(page);await page.keyboard.press('/');const search=page.getByRole('dialog',{name:'Быстрый поиск'});await search.waitFor();
  const input=page.getByRole('combobox',{name:'Событие или раздел'});await input.fill('Стоимость');assert.equal(await page.getByRole('listbox',{name:'Результаты быстрого поиска'}).getByRole('option').count(),1);await accessible(page);await screenshot(page,'quick-search.png');
  await input.press('Enter');await search.waitFor({state:'hidden'});await page.getByRole('dialog').waitFor();assert.ok(page.url().includes('event=event-2'));
  await page.keyboard.press('Escape');await page.getByRole('dialog').waitFor({state:'hidden'});
  await page.keyboard.press('Control+k');await search.waitFor();await input.fill('совпадений_точно_нет');await page.getByText('Ничего не найдено',{exact:true}).waitFor();await input.fill('');await input.press('ArrowDown');await input.press('Enter');await page.getByRole('heading',{name:'Моя аналитика'}).waitFor();assert.ok(page.url().includes('/insights'));
 } finally {await context.close();}
});

test('A deep event link survives reload; tab arrows, provenance and audit stay understandable',async()=>{
 const {context,page}=await setup();try {
  await login(page);await page.goto(baseURL+'/#/insights?event=event-1');await page.getByRole('dialog').waitFor();await page.getByRole('heading',{name:'Что проверить дальше'}).waitFor();await page.getByText('Дата материала: 20 октября 2026 г. · стр. 2',{exact:true}).waitFor();
  await page.evaluate(()=>Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined}));await page.getByRole('button',{name:'Скопировать ссылку на событие'}).click();await page.getByText('Не удалось скопировать. Ссылку можно взять из адресной строки.',{exact:true}).waitFor();
  await page.getByRole('tab',{name:'Сводка',exact:true}).focus();await page.keyboard.press('ArrowRight');assert.equal(await page.getByRole('tab',{name:'Обоснование',exact:true}).getAttribute('aria-selected'),'true');
  await page.getByRole('button',{name:'Выводы модели',exact:false}).first().click();assert.equal(await page.locator('.insight-claim').count(),1);await page.getByRole('button',{name:'Показать журнал анализа'}).click();await page.getByRole('heading',{name:'Как подготовлен анализ'}).waitFor();await page.getByText('Выполнен поиск аналитики банка',{exact:true}).waitFor();
  await page.reload();await page.getByRole('dialog').waitFor();await page.getByRole('heading',{name:'Что проверить дальше'}).waitFor();await page.getByRole('button',{name:'Закрыть событие'}).click();await page.getByRole('dialog').waitFor({state:'hidden'});assert.ok(page.url().includes('/insights'));assert.equal(page.url().includes('event='),false);
 } finally {await context.close();}
});

test('Feedback is saved explicitly, restored, and has an honest unavailable state',async()=>{
 const {context,page,state}=await setup();try {
  await login(page);await page.locator('.brief-card h3 button').first().click();await page.getByRole('heading',{name:'Оценка анализа'}).waitFor();await page.getByRole('radio',{name:'Есть неточность',exact:true}).check();
  assert.equal(state.requests.filter(r=>r.method==='PUT').length,0);await page.getByLabel('Комментарий',{exact:false}).fill('Нужно проверить условия договора.');await page.getByRole('button',{name:'Сохранить оценку',exact:true}).click();await page.getByText('Оценка сохранена для проверки качества.',{exact:true}).waitFor();
  const write=state.requests.find(r=>r.method==='PUT');assert.equal(write.body.value,'needs_review');assert.equal(write.headers['x-demo-session'],'test-DEMO_A');await screenshot(page,'feedback.png');await accessible(page);
  await page.reload();await page.getByRole('radio',{name:'Есть неточность',exact:true}).waitFor();assert.equal(await page.getByRole('radio',{name:'Есть неточность',exact:true}).isChecked(),true);assert.equal(await page.getByLabel('Комментарий',{exact:false}).inputValue(),'Нужно проверить условия договора.');
 } finally {await context.close();}
 const other=await setup({feedbackUnavailable:true});try {await login(other.page);await other.page.locator('.brief-card h3 button').first().click();await other.page.getByText('Оценка анализа пока недоступна на этом сервере.',{exact:true}).waitFor();assert.equal(await other.page.getByRole('radio',{name:'Полезно',exact:true}).isDisabled(),true);} finally {await other.context.close();}
});


test('Profile makes the active role and company context visible, including after a role switch',async()=>{
 const {context,page,state,errors}=await setup();try {
  await login(page);await page.getByRole('button',{name:'Моя компания',exact:true}).click();
  const identity=page.getByRole('region',{name:'Финансовый директор CFO'});await identity.waitFor();
  await identity.getByText('Финансовые вопросы',{exact:true}).waitFor();await page.getByRole('heading',{name:'Данные компании'}).waitFor();
  const passport=page.getByRole('region',{name:'Данные компании'});
  await passport.getByText('Строительство инженерных сооружений',{exact:true}).waitFor();
  await passport.getByText('Единица измерения не подтверждена',{exact:true}).waitFor();
  await passport.locator('.profile-product-list').getByText('Кредитные продукты',{exact:true}).waitFor();
  await passport.getByText('Все признаки продуктов',{exact:true}).click();await passport.getByText('Нет данных',{exact:true}).first().waitFor();
  await accessible(page);await passport.getByText('Все признаки продуктов',{exact:true}).click();await page.locator('.page-title h1').click();await screenshot(page,'profile-cfo.png',false);
  for(const width of [1024,768,390,320]) {await page.setViewportSize({width,height:844});await noOverflow(page);}
  await accessible(page);await page.locator('.page-title h1').click();await screenshot(page,'profile-mobile.png',false);
  await page.getByRole('button',{name:'Сменить роль или компанию'}).click();
  assert.equal(await page.getByLabel('Идентификатор компании').inputValue(),'DEMO_A');
  await page.getByRole('combobox',{name:'Ваша роль'}).selectOption('ceo');
  await page.getByText('Спрос, инвестиции, изменения на рынках и развитие бизнеса.',{exact:true}).waitFor();
  await login(page);await page.getByRole('button',{name:'Моя компания',exact:true}).click();
  await page.getByRole('region',{name:'Генеральный директор CEO'}).waitFor();
  assert.equal(await page.getByRole('region',{name:'Финансовый директор CFO'}).count(),0);
  await page.getByRole('heading',{name:'Стратегические вопросы',exact:true}).waitFor();
  await passport.getByText('Строительство инженерных сооружений',{exact:true}).waitFor();
  assert.equal(state.requests.filter(r=>r.path==='/companies/DEMO_A/profile').at(-1).headers['x-demo-session'],'test-DEMO_A-ceo');
  assert.equal(state.requests.filter(r=>r.method==='POST'&&r.path!=='/demo-session').length,0);assert.deepEqual(errors,[]);
 } finally {await context.close();}
});

test('Profile handles unavailable data, retries and distinguishes absent product flags from unknown',async()=>{
 const {context,page,state}=await setup({profileError:true,profileMissing:true});try {
  await login(page);await page.getByRole('button',{name:'Моя компания',exact:true}).click();
  await page.getByRole('region',{name:'Финансовый директор CFO'}).waitFor();
  await page.getByRole('alert').filter({hasText:'Не удалось загрузить профиль компании'}).waitFor();
  state.profileError=false;await page.getByRole('button',{name:'Повторить загрузку профиля'}).click();
  const passport=page.getByRole('region',{name:'Данные компании'});
  await passport.getByText('Не указана',{exact:true}).waitFor();await passport.getByText('Нет данных о признаке ФТС',{exact:true}).waitFor();
  await passport.getByText('Все признаки продуктов',{exact:true}).click();
  await passport.getByText('Не отмечено в профиле',{exact:true}).waitFor();await passport.getByText('Нет данных',{exact:true}).first().waitFor();
  assert.equal(await passport.locator('.profile-product-list span').count(),0);
  await accessible(page);
 } finally {await context.close();}
});
