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
  const errors=[];page.on('pageerror',e=>errors.push(e.message));const state=await mockApi(page,options);await page.goto(baseURL);return {context,page,state,errors};
}
async function login(page) {await page.getByRole('button',{name:'Перейти к аналитике'}).click();await page.locator('.briefing-layout').waitFor();await page.locator('.feed-skeleton').waitFor({state:'hidden'});}
async function screenshot(page,name){if(screenshotDir){await fs.mkdir(screenshotDir,{recursive:true});await page.screenshot({path:path.join(screenshotDir,name),fullPage:true});}}
async function accessible(page){const result=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(result.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))})),[]);}
async function cardCount(page,count) {await page.waitForFunction(expected=>document.querySelectorAll('.brief-card').length===expected,count);assert.equal(await page.locator('.brief-card').count(),count);}
async function noOverflow(page) {assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);}

test('Briefing prioritizes insights, filters correctly, and never starts paid work on load',async()=>{
 const {context,page,state,errors}=await setup();try {
  await login(page);await cardCount(page,3);assert.deepEqual(state.requests.filter(r=>r.method==='POST').map(r=>r.path),['/demo-session']);await screenshot(page,'desktop-synthetic.png');
  await page.getByRole('button',{name:'Значимость 5/5',exact:true}).click();await cardCount(page,1);
  await page.getByRole('button',{name:'С позицией банка',exact:true}).click();await cardCount(page,1);
  await page.getByRole('button',{name:'Все важное',exact:true}).click();await cardCount(page,3);
  await page.getByRole('button',{name:'Вся персональная аналитика'}).click();
  await page.getByRole('searchbox',{name:'Поиск по персональной ленте'}).fill('Стоимость');await cardCount(page,1);
  await page.getByRole('searchbox',{name:'Поиск по персональной ленте'}).fill('нет_результатов');await page.getByRole('heading',{name:'Под этот фильтр ничего не нашлось'}).waitFor();
  await page.getByRole('button',{name:'Сбросить фильтры'}).click();await cardCount(page,3);await noOverflow(page);assert.deepEqual(errors,[]);
 } finally {await context.close();}
});

test('Event interpretation, evidence, original source, question, focus trap and return focus',async()=>{
 const {context,page,state,errors}=await setup();try {
  await login(page);const opener=page.locator('.brief-card h3 button').first();await opener.click();const dialog=page.getByRole('dialog');await dialog.waitFor();await page.getByRole('heading',{name:'Что проверить дальше'}).waitFor();await screenshot(page,'event-synthetic.png');
  await page.getByRole('button',{name:'Обоснование',exact:true}).click();await dialog.getByRole('button',{name:'Источник 1',exact:true}).first().click();await page.getByText('Подтверждающий фрагмент из тестового документа.',{exact:true}).waitFor();await page.getByRole('button',{name:'Закрыть подтверждающий фрагмент'}).click();
  await page.getByRole('button',{name:'Первоисточники',exact:false}).click();await page.getByRole('button',{name:'Тестовый новостной источник',exact:true}).click();await page.getByText('Полный текст тестового первоисточника.',{exact:false}).waitFor();assert.equal(await page.evaluate(()=>window.injected),undefined);
  await page.getByRole('button',{name:'Влияние на бизнес',exact:true}).click();await page.getByRole('button',{name:'Спросить аналитика',exact:true}).click();await page.getByRole('button',{name:'Какие данные нужны для решения?',exact:true}).click();assert.equal(state.requests.filter(r=>r.path.endsWith('/questions')).length,0);
  await page.getByRole('button',{name:'Задать вопрос',exact:true}).click();await page.getByRole('heading',{name:'Какие данные нужны для решения?',exact:true}).waitFor();const question=state.requests.find(r=>r.path.endsWith('/questions'));assert.ok(question.headers['idempotency-key']);assert.equal(question.headers['x-demo-session'],'test-DEMO_A');await accessible(page);
  await page.getByRole('button',{name:'Закрыть событие'}).focus();await page.keyboard.press('Shift+Tab');assert.equal(await page.locator(':focus').evaluate(el=>!!el.closest('[role=dialog]')),true);
  await page.keyboard.press('Escape');await dialog.waitFor({state:'hidden'});assert.equal(await opener.evaluate(el=>document.activeElement===el),true);assert.deepEqual(errors,[]);
 } finally {await context.close();}
});

test('Responsive layouts and primary screens meet automated WCAG checks',async()=>{
 const {context,page,errors}=await setup();try {
  await accessible(page);await login(page);await accessible(page);
  for(const width of [1440,1024,768,390,320]) {await page.setViewportSize({width,height:844});await noOverflow(page);if(width===390){await screenshot(page,'mobile-synthetic.png');await accessible(page);}}
  await page.getByRole('button',{name:'Моя компания',exact:true}).click();await page.getByRole('heading',{name:'Что может влиять на бизнес'}).waitFor();await page.getByText('Признак отсутствует',{exact:true}).waitFor();await noOverflow(page);await accessible(page);
  await page.getByRole('button',{name:'Добавить новость',exact:true}).click();await page.getByRole('heading',{name:'Добавить публикацию',exact:true}).waitFor();await noOverflow(page);await accessible(page);assert.deepEqual(errors,[]);
 } finally {await context.close();}
});

test('Empty, failed and unconfigured feed states remain actionable',async()=>{
 for(const options of [{empty:true},{feedError:true},{unconfigured:true}]) {
  const {context,page,state}=await setup(options);try {
   await login(page);if(options.empty) await page.getByRole('heading',{name:'Сначала — важное для вас'}).waitFor();
   if(options.feedError) {await page.getByRole('alert').filter({hasText:'Не удалось обновить ленту'}).waitFor();state.feedError=false;await page.getByRole('button',{name:'Повторить',exact:true}).click();await page.locator('.brief-card').first().waitFor();}
   if(options.unconfigured) {assert.equal(await page.getByRole('button',{name:'Найти важное'}).isDisabled(),true);await page.getByText('Подбор недоступен:',{exact:false}).waitFor();}
   assert.equal(state.requests.filter(r=>r.method==='POST' && r.path!=='/demo-session').length,0);
  } finally {await context.close();}
 }
});

test('Generation, discovery and publication are explicit actions; company switch clears session data',async()=>{
 const {context,page,state}=await setup({noInsight:true});try {
  await login(page);await page.getByRole('button',{name:'Найти важное'}).click();await page.getByText('Новых подходящих публикаций',{exact:false}).waitFor();
  await page.locator('.brief-card h3 button').first().click();await page.getByRole('button',{name:'Разобрать влияние',exact:true}).click();await page.getByRole('heading',{name:'Что проверить дальше'}).waitFor();await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'Добавить новость',exact:true}).click();await page.getByLabel('Заголовок',{exact:false}).fill('Тестовая публикация для проверки формы');await page.getByLabel('Текст публикации',{exact:false}).fill('Это синтетический текст публикации для браузерной проверки интерфейса, без реального вызова модели.');await page.getByRole('button',{name:'Отправить на обработку',exact:true}).click();await page.getByText('Публикация отправлена.',{exact:false}).waitFor();
  for(const endpoint of ['/publications','/companies/DEMO_A/insights','/companies/DEMO_A/discovery']) {const req=state.requests.find(r=>r.path===endpoint && r.method==='POST');assert.ok(req,endpoint);assert.ok(req.headers['idempotency-key']);}
  await page.getByRole('button',{name:'Сменить',exact:true}).click();await page.getByLabel('Идентификатор компании').fill('DEMO_B');await page.getByRole('combobox',{name:'Ваша роль'}).selectOption('ceo');await login(page);await page.getByRole('heading',{name:'Стратегический фокус',exact:false}).waitFor();assert.ok((await page.locator('.user-badge').innerText()).includes('DEMO_B'));
  const stored=JSON.parse(await page.evaluate(()=>sessionStorage.getItem('alphich.demo-session.v1')));assert.equal(stored.company_id,'DEMO_B');assert.equal(stored.role,'ceo');
 } finally {await context.close();}
});
