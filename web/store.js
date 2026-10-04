// 游戏页和题库页共用的两件事：
//   1. 本地进度（localStorage）—— 读的是同一个键，逻辑只写一遍就不会两边跑偏
//   2. API 凭据 + 「设置」对话框 —— 开源之后每个人要用自己的 key
// 用普通 <script> 引入（不是 module），所以这些函数是全局的。

const LS_KEY = "soupgame.v1";

function loadStore() {
  try {
    const data = JSON.parse(localStorage.getItem(LS_KEY) || "null");
    return data && typeof data === "object" ? data : {};
  } catch {
    return {};   // 隐私模式 / 存储被禁 / 内容坏了：降级成「不记忆」，其他功能照常
  }
}

// ⚠️ 名字带 Data 是有意的：两个页面的业务代码里各自有一个 saveStore()（无参数，
// 直接写自己的那个 store 变量），同名会撞。
function saveStoreData(data) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(data));
  } catch { /* 存不上就算了，不影响玩 */ }
}


// ══════════════════════════════════════════════════════════════════════════════
// API 凭据
// ══════════════════════════════════════════════════════════════════════════════
// key 存**浏览器**，每次请求用请求头带给后端（见 soup/creds.py）。服务端只用它、
// 不存它 —— 所以别人用你的部署时花的是自己的额度。
//
// ⚠️ 因为存在 localStorage，这个界面只该在自己信任的机器上用。
//    真要部署到公网，务必套 HTTPS，否则 key 在网络上明文传输。

const CRED_KEY = "soupgame.creds.v1";

function loadCreds() {
  try {
    const data = JSON.parse(localStorage.getItem(CRED_KEY) || "null");
    return data && typeof data === "object" ? data : {};
  } catch {
    return {};
  }
}

function saveCreds(creds) {
  try {
    localStorage.setItem(CRED_KEY, JSON.stringify(creds));
  } catch { /* 存不上，那这次就只在内存里用着 */ }
}

// 拼请求头。只带**有值**的项 —— 没带就等于「这次没有这把 key」，
// 后端不会去别处找（服务端一个 key 都不存，见 soup/creds.py）。
function credHeaders() {
  const c = loadCreds();
  const h = { "Content-Type": "application/json" };
  if (c.transport) h["X-Jev-Transport"] = c.transport;
  if (c.jevKey) h["X-Jev-Key"] = c.jevKey;
  if (c.deepseekKey) h["X-DeepSeek-Key"] = c.deepseekKey;
  return h;
}

// 统一的 fetch 包装：自动带凭据头，并把后端的 detail 变成可读的错误。
async function credFetch(url, options = {}) {
  // ⚠️ 必须这样合并 header，不能写 `{ headers: credHeaders(), ...options }` ——
  //    调用方大多自带 `headers: {"Content-Type": ...}`，那种写法会让它**整个覆盖掉**
  //    凭据头，于是这次请求就没带 key。表现是「我明明填了 key，却说是没填」。
  const { headers, ...rest } = options;
  const r = await fetch(url, { ...rest, headers: { ...credHeaders(), ...headers } });
  if (!r.ok) {
    let detail = `请求失败（${r.status}）`;
    try {
      const d = await r.json();
      detail = d.detail || d.reply || detail;
    } catch { /* 响应不是 JSON，就用默认文案 */ }
    const err = new Error(detail);
    err.status = r.status;
    throw err;
  }
  return r;
}


function errText(e) {
  // 后端明确回了错误（credFetch 抛的，带 status）→ 那句话本来就是写给人看的，直接用。
  // 否则是 fetch 自己炸的（服务没起来、断网）→ 才叫「连不上后端」。
  //
  // ⚠️ 别把两种情况都写成「连不上后端」：没填 key 会回 401，那种错是「你去设置里填」
  //    而不是「服务挂了」。显示成「连不上后端」会让人去查一个根本不存在的故障。
  //    也别直接 `${e}` —— 那会带上 "Error: " 前缀，像程序崩了。
  if (e && e.status) return e.message || `请求失败（${e.status}）`;
  return `连不上后端：${(e && e.message) || e}`;
}


// ══════════════════════════════════════════════════════════════════════════════
// 「设置」对话框
// ══════════════════════════════════════════════════════════════════════════════
// 自带样式（.cred- 前缀），不依赖各页面自己的 dialog 规则 —— 这样两个页面
// 只要调一次 mountSettings() 就行，不必各自维护一份弹窗。

const CRED_CSS = `
.cred-btn { display: inline-flex; align-items: center; gap: 6px; }
.cred-btn.attention { border-color: #e8c86a; background: #fdf6e3; color: #8a6d1f; }
.cred-dialog::backdrop { background: rgba(22,24,29,.32); }
.cred-dialog {
  border: 1px solid var(--line, #e4e7ec); border-radius: 12px; padding: 22px;
  background: var(--card, #fff); color: var(--fg, #16181d);
  width: min(560px, 92vw); box-shadow: 0 12px 40px rgba(22,24,29,.18);
}
.cred-dialog h3 { margin: 0 0 6px; font-size: 16px; }
.cred-dialog .cred-sub { font-size: 12px; color: var(--faint, #9aa1ac); line-height: 1.8; margin-bottom: 14px; }
.cred-dialog label { display: block; font-size: 12px; color: var(--dim, #6b7280); margin: 14px 0 5px; }
.cred-dialog select, .cred-dialog input[type=text], .cred-dialog input[type=password] {
  width: 100%; background: var(--card, #fff); color: inherit;
  border: 1px solid var(--line, #e4e7ec); border-radius: 7px;
  padding: 8px 11px; font-size: 13px; font-family: inherit;
}
.cred-dialog input:focus, .cred-dialog select:focus { outline: none; border-color: #7b93c9; }
.cred-dialog .cred-row { display: flex; gap: 8px; align-items: center; }
.cred-dialog .cred-row input { flex: 1; }
.cred-dialog code {
  font-family: ui-monospace, Consolas, monospace; font-size: 11px;
  background: var(--line-soft, #eef0f3); padding: 1px 5px; border-radius: 4px;
}
.cred-dialog .cred-note {
  font-size: 12px; line-height: 1.8; color: var(--dim, #6b7280);
  background: var(--line-soft, #eef0f3); border-radius: 7px; padding: 9px 12px; margin-top: 8px;
}
.cred-dialog .cred-note a { color: #4b7bec; }
.cred-dialog .cred-foot { display: flex; gap: 8px; align-items: center; margin-top: 20px; }
.cred-dialog .cred-foot .spacer { flex: 1; }
.cred-dialog .cred-msg { margin-top: 12px; font-size: 13px; line-height: 1.7; }
.cred-dialog .cred-msg.ok { color: #17a34a; }
.cred-dialog .cred-msg.bad { color: #dc2626; }
.cred-dialog .cred-msg.warn { color: #b45309; }
.cred-dialog .cred-msg.busy { color: var(--dim, #6b7280); }
.cred-dialog .cred-show { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--dim, #6b7280); margin-top: 10px; }
.cred-dialog .cred-show input { width: auto; margin: 0; }
.cred-dialog .cred-show label { margin: 0; }
`;

let credOptions = null;   // 缓存的 /api/settings 结果

async function fetchCredOptions() {
  if (!credOptions) credOptions = await (await fetch("/api/settings")).json();
  return credOptions;
}

function credDialogHTML() {
  return `
  <dialog class="cred-dialog" id="credDialog">
    <h3>设置 API</h3>
    <div class="cred-sub" id="credSub">
      key 只存在你自己的浏览器里，每次请求带给本机后端，服务端不保存它。
    </div>

    <label>Jev 接口</label>
    <select id="credTransport"></select>
    <div class="cred-note" id="credTransportNote"></div>

    <label>Jev API key</label>
    <div class="cred-row">
      <input type="password" id="credJev" placeholder="粘贴 key" autocomplete="off" spellcheck="false" />
    </div>

    <label>DeepSeek API key（可选）</label>
    <div class="cred-row">
      <input type="password" id="credDs" placeholder="粘贴 key" autocomplete="off" spellcheck="false" />
    </div>
    <div class="cred-note">
      不填只会少掉「二次复核 / 进度提示 / 通关判定」这三个功能，是非题本身照常能玩。
      这里要的是 DeepSeek 自己的 key，和 Jev 的 key <b>不通用</b>，在
      <a href="https://platform.deepseek.com/api_keys" target="_blank" rel="noopener">platform.deepseek.com</a>
      申请。
    </div>

    <div class="cred-show">
      <input type="checkbox" id="credShow" /><label for="credShow">显示 key</label>
    </div>

    <div class="cred-msg" id="credMsg"></div>

    <div class="cred-foot">
      <button class="ghost" id="credTest" type="button">测试连接</button>
      <span class="spacer"></span>
      <button class="ghost" id="credCancel" type="button">取消</button>
      <button class="primary" id="credSave" type="button">保存</button>
    </div>
  </dialog>`;
}

function credEl(id) { return document.getElementById(id); }

function syncTransportNote() {
  if (!credOptions) return;
  const name = credEl("credTransport").value;
  const t = credOptions.transports.find((x) => x.name === name);
  if (!t) return;
  // 服务端不再有兜底 key，所以这里只有一种说法：必须填你自己的。
  credEl("credSub").innerHTML =
    "key 只存在你自己的浏览器里，每次请求带给后端，服务端不保存它。<br />"
    + "这个项目<b>不提供</b>公共 key，每个人用自己的。";
  const bits = [t.note];
  if (credOptions.keys_are_user_supplied) {
    bits.push("⚠️ 必须填你自己的 key，否则会被拒绝。");
  }
  bits.push(`用到的模型：<code>${t.model}</code>`);
  bits.push(`<a href="${t.keys_url}" target="_blank" rel="noopener">在哪里申请 key</a>`);
  credEl("credTransportNote").innerHTML = bits.join("<br />");
  credEl("credJev").placeholder = t.key_hint || "粘贴 key";
  updateCredBadge();
}

async function mountSettings(host) {
  if (!host) return;
  if (!credEl("credStyle")) {
    const style = document.createElement("style");
    style.id = "credStyle";
    style.textContent = CRED_CSS;
    document.head.appendChild(style);
  }
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "ghost cred-btn";
  btn.id = "credOpen";
  btn.textContent = "设置";
  btn.title = "配置你自己的 API key";
  host.appendChild(btn);

  const holder = document.createElement("div");
  holder.innerHTML = credDialogHTML();
  document.body.appendChild(holder.firstElementChild);

  const fillTransports = () => {
    if (!credOptions) return;
    credEl("credTransport").innerHTML = credOptions.transports
      .map((t) => `<option value="${t.name}">${t.label}</option>`)
      .join("");
    const saved = loadCreds().transport;
    if (saved) credEl("credTransport").value = saved;
  };

  try {
    await fetchCredOptions();
    fillTransports();
  } catch { /* 拿不到选项就先留着空，打开弹窗时会再试 */ }

  const creds = loadCreds();
  credEl("credJev").value = creds.jevKey || "";
  credEl("credDs").value = creds.deepseekKey || "";
  syncTransportNote();

  credEl("credOpen").addEventListener("click", async () => {
    try {
      await fetchCredOptions();
      fillTransports();
      syncTransportNote();
    } catch { /* 用已有选项继续 */ }
    credEl("credMsg").textContent = "";
    credEl("credDialog").showModal();
  });
  credEl("credTransport").addEventListener("change", syncTransportNote);
  credEl("credShow").addEventListener("change", (e) => {
    const type = e.target.checked ? "text" : "password";
    credEl("credJev").type = type;
    credEl("credDs").type = type;
  });
  credEl("credCancel").addEventListener("click", () => credEl("credDialog").close());
  credEl("credSave").addEventListener("click", () => {
    saveCreds({
      transport: credEl("credTransport").value,
      jevKey: credEl("credJev").value.trim(),
      deepseekKey: credEl("credDs").value.trim(),
    });
    credEl("credDialog").close();
    updateCredBadge();
  });
  credEl("credTest").addEventListener("click", testConnection);
}

// 按钮上的小提示：没填 key 就标黄提醒一下 —— 现在服务端不提供 key，所以这条总会生效。
function updateCredBadge() {
  const btn = credEl("credOpen");
  if (!btn || !credOptions) return;
  const c = loadCreds();
  btn.classList.toggle("attention", !c.jevKey);
}

async function testConnection() {
  const msg = credEl("credMsg");
  msg.className = "cred-msg busy";
  msg.textContent = "正在用你填的 key 试一次调用…";
  credEl("credTest").disabled = true;
  try {
    // 测的是**当前输入框里**的值，不是已保存的 —— 这样「填完先测再存」很顺。
    // ⚠️ DeepSeek 那把也要带上：不带的话后端根本不知道要测它，只会回一句
    //    「Jev 连接正常」，让人以为两个 key 都没问题。
    const r = await fetch("/api/settings/check", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Jev-Transport": credEl("credTransport").value,
        "X-Jev-Key": credEl("credJev").value.trim(),
        "X-DeepSeek-Key": credEl("credDs").value.trim(),
      },
      body: "{}",
    });
    const d = await r.json().catch(() => ({}));
    // 只填了 Jev、DeepSeek 空着 → 整体算 ok，但要把「少一半」说出来
    const warn = d.ok && d.deepseek_ok === null;
    msg.className = `cred-msg ${!d.ok ? "bad" : warn ? "warn" : "ok"}`;
    msg.textContent = d.message || (d.ok ? "连接正常。" : `测试失败（${r.status}）`);
  } catch (e) {
    msg.className = "cred-msg bad";
    msg.textContent = `请求失败：${e.message}`;
  } finally {
    credEl("credTest").disabled = false;
  }
}
