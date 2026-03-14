"""
app.py — Flask backend za Eduza AI asistent
Pokreni: python app.py
Otvori:  http://localhost:5000
"""

import warnings, logging, os
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify, render_template_string
from recommender import recommend

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="hr">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Eduza AI Asistent</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --blue:    #2563eb;
      --blue-l:  #3b82f6;
      --blue-xl: #eff6ff;
      --dark:    #0f172a;
      --gray:    #64748b;
      --light:   #f1f5f9;
      --white:   #ffffff;
      --border:  #e2e8f0;
      --radius:  14px;
    }

    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: #f8fafc;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      color: var(--dark);
    }

    /* ── HEADER ── */
    header {
      width: 100%;
      background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #3b82f6 100%);
      padding: 0 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 64px;
      box-shadow: 0 4px 20px rgba(37,99,235,0.3);
      position: sticky; top: 0; z-index: 100;
    }
    .header-left { display: flex; align-items: center; gap: 12px; }
    .header-logo {
      width: 36px; height: 36px; background: rgba(255,255,255,0.2);
      border-radius: 10px; display: flex; align-items: center;
      justify-content: center; font-size: 20px;
      backdrop-filter: blur(8px);
    }
    header h1 { font-size: 17px; font-weight: 700; color: #fff; }
    header p  { font-size: 12px; color: rgba(255,255,255,0.75); margin-top: 1px; }
    .header-badge {
      background: rgba(255,255,255,0.15);
      color: #fff;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.25);
      backdrop-filter: blur(8px);
    }

    /* ── LAYOUT ── */
    .layout {
      display: flex;
      position: fixed;
      top: 64px; left: 0; right: 0; bottom: 0;
      max-width: 1100px;
      margin: 0 auto;
      gap: 0;
      padding: 0;
    }

    /* ── SIDEBAR ── */
    .sidebar {
      width: 272px;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow-y: auto;
      height: 100%;
      padding: 16px 16px 16px 16px;
      border-right: 1px solid var(--border);
      background: #f8fafc;
    }
    .sidebar::-webkit-scrollbar { width: 0; }
    .sidebar-card {
      background: var(--white);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .sidebar-card h3 {
      font-size: 12px;
      font-weight: 700;
      color: var(--gray);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 12px;
    }
    .suggestion {
      display: block;
      width: 100%;
      text-align: left;
      background: var(--blue-xl);
      border: 1px solid #bfdbfe;
      color: var(--blue);
      font-size: 12px;
      font-weight: 500;
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
      margin-bottom: 6px;
      transition: all 0.15s;
      font-family: inherit;
    }
    .suggestion:hover { background: #dbeafe; border-color: var(--blue-l); }
    .suggestion:last-child { margin-bottom: 0; }

    .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); }
    .stat-row:last-child { border-bottom: none; }
    .stat-label { font-size: 12px; color: var(--gray); }
    .stat-value { font-size: 13px; font-weight: 700; color: var(--dark); }

    /* ── CHAT AREA ── */
    .chat-col {
      flex: 1; display: flex; flex-direction: column;
      min-width: 0; height: 100%; overflow: hidden;
      padding: 0 0 0 0;
    }

    .messages {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 20px;
      overflow-y: auto;
      padding: 20px 20px 12px 20px;
      scroll-behavior: smooth;
    }
    .messages::-webkit-scrollbar { width: 4px; }
    .messages::-webkit-scrollbar-track { background: transparent; }
    .messages::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

    .msg { display: flex; gap: 10px; align-items: flex-start; }
    .msg.user { flex-direction: row-reverse; }

    .avatar {
      width: 34px; height: 34px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px; flex-shrink: 0; border: 2px solid var(--border);
      background: var(--white);
    }
    .msg.user .avatar {
      background: linear-gradient(135deg, #2563eb, #3b82f6);
      border-color: transparent; color: #fff; font-size: 14px; font-weight: 700;
    }

    .bubble {
      max-width: 82%;
      padding: 13px 16px;
      border-radius: 4px 16px 16px 16px;
      font-size: 14px;
      line-height: 1.65;
      color: var(--dark);
      background: var(--white);
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
      border: 1px solid var(--border);
    }
    .msg.user .bubble {
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: #fff;
      border: none;
      border-radius: 16px 4px 16px 16px;
      box-shadow: 0 4px 14px rgba(37,99,235,0.35);
    }
    .bubble strong { font-weight: 600; }
    .bubble ul { padding-left: 18px; margin: 6px 0; }
    .bubble li { margin-bottom: 4px; }

    /* ── COURSE CARDS ── */
    .courses { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }

    .course-card {
      background: #f8faff;
      border: 1px solid #dbeafe;
      border-left: 3px solid var(--blue);
      border-radius: 10px;
      padding: 10px 14px;
      transition: box-shadow 0.15s;
    }
    .course-card:hover { box-shadow: 0 4px 12px rgba(37,99,235,0.12); }

    .course-card a {
      font-size: 13px;
      font-weight: 600;
      color: #1e40af;
      text-decoration: none;
    }
    .course-card a:hover { text-decoration: underline; }

    .course-meta { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
    .badge {
      font-size: 11px; font-weight: 500;
      padding: 2px 8px; border-radius: 20px;
      background: #e0f2fe; color: #0369a1;
    }
    .badge.green  { background: #dcfce7; color: #15803d; }
    .badge.yellow { background: #fef9c3; color: #92400e; }
    .badge.gray   { background: #f1f5f9; color: #64748b; }
    .badge.cat    { background: #f3e8ff; color: #7c3aed; }

    .disclaimer {
      font-size: 11px; color: #94a3b8; font-style: italic;
      margin-top: 10px; padding-top: 8px;
      border-top: 1px solid var(--border);
    }

    /* ── TYPING ── */
    .typing-wrap { display: flex; gap: 10px; align-items: center; }
    .typing {
      display: flex; gap: 5px; align-items: center;
      padding: 13px 18px;
      background: var(--white);
      border: 1px solid var(--border);
      border-radius: 4px 16px 16px 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: #93c5fd;
      animation: bounce 1.2s infinite;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
      30% { transform: translateY(-7px); opacity: 1; }
    }

    /* ── INPUT BAR ── */
    .input-bar {
      flex-shrink: 0;
      background: rgba(248,250,252,0.97);
      backdrop-filter: blur(12px);
      border-top: 1px solid var(--border);
      padding: 12px 20px 16px 20px;
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }
    .input-bar textarea {
      flex: 1;
      border: 1.5px solid var(--border);
      border-radius: 12px;
      padding: 11px 16px;
      font-size: 14px;
      font-family: inherit;
      resize: none;
      outline: none;
      background: var(--white);
      color: var(--dark);
      transition: border-color 0.15s, box-shadow 0.15s;
      height: 44px;
      max-height: 160px;
      overflow: hidden;
      line-height: 1.5;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .input-bar textarea:focus {
      border-color: var(--blue-l);
      box-shadow: 0 0 0 3px rgba(59,130,246,0.15);
    }
    .input-bar button {
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: #fff; border: none; border-radius: 12px;
      width: 44px; height: 44px; font-size: 18px; cursor: pointer;
      transition: all 0.15s; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 12px rgba(37,99,235,0.35);
    }
    .input-bar button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(37,99,235,0.4); }
    .input-bar button:disabled { background: #93c5fd; box-shadow: none; transform: none; cursor: not-allowed; }

    @media (max-width: 768px) {
      .sidebar { display: none; }
    }
  </style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="header-logo">🎓</div>
    <div>
      <h1>Eduza AI Asistent</h1>
      <p>Pronađite savršeni tečaj za vas</p>
    </div>
  </div>
  <span class="header-badge">{{ course_count }} tečaja · eduza.hr</span>
</header>

<div class="layout">

  <!-- SIDEBAR -->
  <aside class="sidebar">

    <div class="sidebar-card">
      <h3>📈 Iskorištenost danas</h3>
      <div class="stat-row">
        <span class="stat-label">Upita danas</span>
        <span class="stat-value" id="req-count">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Dnevni limit</span>
        <span class="stat-value">1,000</span>
      </div>
      <div style="margin-top:10px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748b;margin-bottom:4px;">
          <span>Iskorišteno</span><span id="req-pct">0%</span>
        </div>
        <div style="background:#e2e8f0;border-radius:999px;height:6px;overflow:hidden;">
          <div id="req-bar" style="height:100%;width:0%;background:linear-gradient(90deg,#22c55e,#16a34a);border-radius:999px;transition:width 0.4s;"></div>
        </div>
      </div>
      <div style="font-size:11px;color:#94a3b8;margin-top:8px;">Reset u ponoć</div>
    </div>

    <div class="sidebar-card">
      <h3>💡 Primjeri upita</h3>
      <button class="suggestion" onclick="fillInput(this)">Zanima me digitalni marketing</button>
      <button class="suggestion" onclick="fillInput(this)">Kako poboljšati komunikaciju s klijentima</button>
      <button class="suggestion" onclick="fillInput(this)">Trebam naučiti voditi tim</button>
      <button class="suggestion" onclick="fillInput(this)">Želim pokrenuti vlastiti biznis</button>
      <button class="suggestion" onclick="fillInput(this)">Kako se bolje prezentirati</button>
      <button class="suggestion" onclick="fillInput(this)">Zanima me računovodstvo</button>
    </div>

    <div class="sidebar-card">
      <h3>📊 Statistike</h3>
      <div class="stat-row">
        <span class="stat-label">Tečajeva u bazi</span>
        <span class="stat-value">{{ course_count }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">AI model</span>
        <span class="stat-value">Llama 3.3 70B</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Pretraga</span>
        <span class="stat-value">ChromaDB</span>
      </div>
    </div>

    <div class="sidebar-card">
      <h3>ℹ️ O asistentu</h3>
      <p style="font-size:12px;color:#64748b;line-height:1.6;">
        AI analizira vaš upit, pretražuje vektorsku bazu i preporučuje najprikladnije tečajeve s eduza.hr.
        <br><br>
        Preporuke su automatizirane — provjerite detalje na <a href="https://eduza.hr" target="_blank" style="color:#2563eb;">eduza.hr</a>.
      </p>
    </div>

    <div class="sidebar-card">
      <h3>🔢 Kako računamo postotak?</h3>
      <p style="font-size:12px;color:#64748b;line-height:1.6;">
        <strong style="color:#1e293b;">% podudaranje</strong> = kosinusna sličnost između vašeg upita i opisa tečaja, izračunata vektorskim modelom <em>paraphrase-multilingual-MiniLM-L12-v2</em>.
        <br><br>
        Oba teksta (upit + opis tečaja) pretvaraju se u numeričke vektore; postotak pokazuje koliko su ti vektori "blizu" u semantičkom prostoru.
        <br><br>
        <strong style="color:#1e293b;">Napomena:</strong> visok postotak znači semantičku sličnost teksta, ne garanciju kvalitete tečaja. Uvijek provjerite detalje na eduza.hr.
      </p>
      <div style="margin-top:10px;padding-top:10px;border-top:1px solid #e2e8f0;">
        <p style="font-size:11px;color:#94a3b8;line-height:1.5;">
          🎯 Rezultati su diversificirani po kategorijama (max 2 po kategoriji) kako bi se izbjegla dominacija jedne teme.
        </p>
      </div>
    </div>

  </aside>

  <!-- CHAT -->
  <div class="chat-col">
    <div class="messages" id="messages">
      <div class="msg bot">
        <div class="avatar">🤖</div>
        <div class="bubble">
          Pozdrav! 👋 Ja sam vaš AI asistent za edukacije.<br><br>
          Opišite mi <strong>što želite naučiti</strong>, koji problem rješavate ili za što se pripremate — preporučit ću vam najprikladnije tečajeve s <strong>eduza.hr</strong>.<br><br>
          Možete koristiti primjere s lijeve strane ili upisati vlastiti upit. 😊
        </div>
      </div>
    </div>
    <div class="input-bar">
      <textarea id="input" rows="1" placeholder="Opišite što tražite... (Enter za slanje)" onkeydown="handleKey(event)"></textarea>
      <button id="sendBtn" onclick="sendMessage()" title="Pošalji">➤</button>
    </div>
  </div>

</div>

<script>
  const messages = document.getElementById('messages');
  const input    = document.getElementById('input');
  const sendBtn  = document.getElementById('sendBtn');
  const MAX_DOM_MSGS = 30; // max poruka u DOM-u da tab ne usporava
  let chatHistory = []; // povijest za AI kontekst

  function fillInput(btn) {
    input.value = btn.textContent;
    input.focus();
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }

  function escHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function formatAI(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .split('\\n\\n').join('<br><br>')
      .split('\\n').join('<br>');
  }

  function scrollDown() {
    messages.scrollTo({ top: messages.scrollHeight, behavior: 'smooth' });
  }

  function addMsg(role, html) {
    const isUser = role === 'user';
    const div = document.createElement('div');
    div.className = 'msg ' + (isUser ? 'user' : 'bot');
    div.innerHTML = `<div class="avatar">${isUser ? '👤' : '🤖'}</div><div class="bubble">${html}</div>`;
    messages.appendChild(div);
    const allMsgs = messages.querySelectorAll('.msg');
    if (allMsgs.length > MAX_DOM_MSGS) allMsgs[1].remove();
    scrollDown();
    return div;
  }

  function showTyping() {
    const div = document.createElement('div');
    div.className = 'msg bot'; div.id = 'typing';
    div.innerHTML = `
      <div class="avatar">🤖</div>
      <div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
    messages.appendChild(div);
    scrollDown();
  }
  function hideTyping() { const t = document.getElementById('typing'); if(t) t.remove(); }

  // Ključne riječi koje signaliziraju da korisnik želi slabije rezultate
  const SHOW_WEAK_KEYWORDS = ['još', 'više', 'sve', 'ostale', 'ostali', 'manje relevantne',
                               'ispod', 'alternativ', 'prikaži sve', 'prikaži još', 'drugi',
                               'druge', 'show more', 'more results'];
  let lastQuery = '';

  function wantsWeakResults(text) {
    const t = text.toLowerCase();
    return SHOW_WEAK_KEYWORDS.some(k => t.includes(k));
  }

  function renderCourses(courses, weakCount = 0) {
    if (!courses || !courses.length) return '';
    let html = '<div class="courses">';
    courses.forEach(c => {
      const score = Math.round((c.relevance_score || 0) * 100);
      const scoreClass = score >= 70 ? 'green' : score >= 50 ? 'yellow' : 'gray';
      const catOk = c.category && c.category.toLowerCase() !== c.title.toLowerCase() && c.category.length < 40;
      const cat   = catOk ? `<span class="badge cat">📁 ${escHtml(c.category)}</span>` : '';
      const price = c.price ? `<span class="badge">💰 ${escHtml(c.price)}</span>` : '';
      const durBad = !c.duration || c.duration.toLowerCase().includes('odaberite') || c.duration.toLowerCase().includes('termin');
      const dur   = durBad ? '' : `<span class="badge">⏱ ${escHtml(c.duration)}</span>`;
      html += `
        <div class="course-card">
          <a href="${escHtml(c.url)}" target="_blank">${escHtml(c.title)}</a>
          <div class="course-meta">
            ${cat}${price}${dur}
            <span class="badge ${scoreClass}">🎯 ${score}% podudaranje</span>
          </div>
        </div>`;
    });
    html += '</div>';

    if (weakCount > 0) {
      html += `<div class="disclaimer" style="margin-top:10px;">
        ℹ️ Prikazani su samo tečajevi s ≥50% semantičke sličnosti.
        Postoji još <strong>${weakCount}</strong> tečaja s manjim podudaranjem —
        upišite <em>"prikaži još"</em> ako ih želite vidjeti.
      </div>`;
    } else {
      html += `<div class="disclaimer">ℹ️ Preporuke generira AI na temelju opisa tečajeva. Provjeri detalje na eduza.hr prije rezervacije.</div>`;
    }
    return html;
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;
    input.value = ''; input.style.height = '44px'; input.style.overflow = 'hidden';
    sendBtn.disabled = true;

    addMsg('user', escHtml(text));
    showTyping();
    updateUsage(1);

    // Detektira li korisnik želi slabije rezultate
    const includeWeak = wantsWeakResults(text) && lastQuery !== '';
    const queryToSend = includeWeak ? lastQuery : text;
    if (!includeWeak) lastQuery = text;

    // Dodaj u povijest
    chatHistory.push({ role: 'user', content: text });
    if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);

    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: queryToSend, history: chatHistory, include_weak: includeWeak }),
      });
      const data = await resp.json();
      hideTyping();
      addMsg('bot', formatAI(data.reply) + renderCourses(data.courses, data.weak_count || 0));
      chatHistory.push({ role: 'assistant', content: data.reply });
      if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);
    } catch {
      hideTyping();
      addMsg('bot', '⚠️ Greška pri komunikaciji s AI modelom. Pokušajte ponovo.');
    }

    sendBtn.disabled = false;
    input.focus();
  }

  input.addEventListener('input', () => {
    input.style.height = '44px';
    const newH = Math.min(input.scrollHeight, 160);
    input.style.height = newH + 'px';
    input.style.overflow = input.scrollHeight > 160 ? 'auto' : 'hidden';
  });

  // ── Usage tracker (localStorage, reset u ponoc) ──
  const LIMIT = 1000;
  function getUsage() {
    const today = new Date().toDateString();
    const stored = JSON.parse(localStorage.getItem('eduza_usage') || '{}');
    if (stored.date !== today) return { date: today, count: 0 };
    return stored;
  }
  function updateUsage(add = 0) {
    const u = getUsage();
    u.count += add;
    localStorage.setItem('eduza_usage', JSON.stringify(u));
    const pct = Math.min((u.count / LIMIT) * 100, 100).toFixed(1);
    const color = u.count > 800 ? 'linear-gradient(90deg,#ef4444,#dc2626)'
                : u.count > 500 ? 'linear-gradient(90deg,#f59e0b,#d97706)'
                : 'linear-gradient(90deg,#22c55e,#16a34a)';
    document.getElementById('req-count').textContent = u.count.toLocaleString();
    document.getElementById('req-pct').textContent = pct + '%';
    document.getElementById('req-bar').style.width = pct + '%';
    document.getElementById('req-bar').style.background = color;
  }
  updateUsage(0); // inicijalni prikaz
</script>
</body>
</html>"""

@app.route("/")
def index():
    try:
        from recommender import get_collection
        count = get_collection().count()
    except Exception:
        count = 0
    return render_template_string(HTML, course_count=count)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg     = data.get("message", "").strip()
    history      = data.get("history", [])
    include_weak = data.get("include_weak", False)
    if not user_msg:
        return jsonify({"reply": "Molim unesite upit.", "courses": [], "weak_count": 0})
    try:
        ai_text, courses, weak_count = recommend(user_msg, history, include_weak)
        return jsonify({"reply": ai_text, "courses": courses, "weak_count": weak_count})
    except Exception as e:
        return jsonify({"reply": f"Greška: {e}", "courses": [], "weak_count": 0}), 500

if __name__ == "__main__":
    print("Pokrecem Eduza AI Asistent na http://localhost:5000")
    app.run(debug=False, port=5000)
