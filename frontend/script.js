const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (n) => Math.max(0, Math.min(100, Math.round(Number(n) || 0)));
const chips = (label, list, cls) =>
  list && list.length
    ? `<div class="chips"><b>${label}</b>${list.map((s) => `<span class="chip ${cls}">${esc(s)}</span>`).join("")}</div>`
    : "";

/* screens */
function show(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  $(id).classList.add("active");
  document.body.classList.toggle("on-landing", id === "landing");
  window.scrollTo(0, 0);
}
$("brandBtn").addEventListener("click", () => show("landing"));
$("startBtn").addEventListener("click", () => show("roles"));
document.querySelectorAll("[data-role]").forEach((b) => b.addEventListener("click", () => show(b.dataset.role)));
document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => show(b.dataset.go)));

/*  file drop boxes  */
function setupDrop(dropId, inputId, textId, defaultText, onPick) {
  const drop = $(dropId);
  const pick = (f) => { onPick(f); $(textId).textContent = f ? f.name : defaultText; };
  $(inputId).addEventListener("change", (e) => pick(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));
}
let resumeFile = null;
let jobFile = null;
setupDrop("drop", "file", "dropText", "Drop your resume here or click to browse", (f) => (resumeFile = f));
setupDrop("jobDrop", "jobFile", "jobDropText", "Drop the Job Description here or click to browse", (f) => (jobFile = f));

/* server calls */
async function post(url, fd) {
  const res = await fetch(url, { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Something went wrong.");
  return data.results || [];
}

async function run(btn, out, work) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Matching...";
  out.innerHTML = `<p class="msg">Reading and comparing, this can take a few seconds.</p>`;
  try {
    await work();
  } catch (err) {
    out.innerHTML = `<p class="msg error">${esc(err.message)} Check that the server is running.</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function animateBars(root) {
  requestAnimationFrame(() => root.querySelectorAll(".bar > i").forEach((i) => (i.style.width = i.dataset.w + "%")));
}

/*  downloads  */
function saveBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function downloadCsv(rows, name) {
  const cell = (v) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const text = "\ufeff" + rows.map((r) => r.map(cell).join(",")).join("\n");
  saveBlob(new Blob([text], { type: "text/csv;charset=utf-8" }), name);
}

function downloadPdf(title, blocks, name) {
  if (!window.jspdf) { alert("The PDF library could not load (check your internet). Please use CSV instead."); return; }
  const doc = new window.jspdf.jsPDF({ unit: "pt", format: "a4" });
  const M = 40, W = 515;
  let y = M;
  const room = (h) => { if (y + h > 800) { doc.addPage(); y = M; } };
    doc.setFont("helvetica", "bold"); doc.setFontSize(18); doc.text(title, M, y); y += 20;
    doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.text(new Date().toLocaleDateString(), M, y); y += 26;
  blocks.forEach((b) => {
    room(50);
    doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.text(doc.splitTextToSize(b.h, W), M, y); y += 17;
    doc.setFont("helvetica", "normal"); doc.setFontSize(10);
    b.lines.forEach((l) => {
      const parts = doc.splitTextToSize(l, W);
      room(parts.length * 13);
      doc.text(parts, M, y);
      y += parts.length * 13;
    });
    y += 12;
  });
  doc.save(name);
}

const dlRow = (title, note) => `
  <div class="dl-row">
    <div><b>${title}</b><span>${note}</span></div>
    <div class="dl-btns"><button class="dl" data-dl="pdf">Download PDF</button><button class="dl" data-dl="csv">Download CSV</button></div>
  </div>`;

/*  cards  */
const coverage = (r) => {
  if (r.coverage != null) return pct(r.coverage);
  const total = (r.matched || []).length + (r.missing || []).length;
  return total ? pct(((r.matched || []).length / total) * 100) : 0;
};
const metric = (label, v) =>
  v == null ? "" : `<div class="mrow"><span>${label}</span><b>${v}%</b><div class="bar"><i data-w="${v}"></i></div></div>`;

function card(title, sub, r, student) {
  const score = pct(r.score);
  return `
    <article class="card">
      <div class="card-head">
        <div><h3>${esc(title)}</h3><p class="sub">${esc(sub || "")}</p></div>
        <div class="pct">${score}%</div>
      </div>
      <div class="bar"><i data-w="${score}"></i></div>
      ${student ? `<div class="metrics">${metric("Semantic similarity", r.semantic != null ? pct(r.semantic) : null)}${metric("Required skill coverage", coverage(r))}</div>` : ""}
      ${chips(student ? "You have" : "Matching skills", r.matched, "have")}
      ${student && r.missing && r.missing.length
        ? `<details class="missing"><summary>Skills you are missing (${r.missing.length})</summary><div class="chips">${r.missing.map((s) => `<span class="chip need">${esc(s)}</span>`).join("")}</div></details>`
        : ""}
      ${student && r.explanation ? `<details><summary>Why this match?</summary><p>${esc(r.explanation)}</p></details>` : ""}
    </article>`;
}

/* STUDENT  */
let positions = [];

function renderStudent() {
  const out = $("studentOut");
  if (!positions.length) { out.innerHTML = `<p class="msg">No matches found. Try a longer resume.</p>`; return; }
  out.innerHTML =
    positions.map((r) => card(r.title, r.company, r, true)).join("") +
    dlRow("Skill gap report", "The skills you are missing for every position");
  animateBars(out);
  out.querySelector('[data-dl="csv"]').addEventListener("click", studentCsv);
  out.querySelector('[data-dl="pdf"]').addEventListener("click", studentPdf);
}

function studentCsv() {
  const rows = [["Company", "Position", "Match %", "Semantic similarity %", "Required skill coverage %", "Matching skills", "Missing skills"]];
  positions.forEach((r) =>
    rows.push([r.company, r.title, pct(r.score), r.semantic != null ? pct(r.semantic) : "", coverage(r), (r.matched || []).join("; "), (r.missing || []).join("; ")])
  );
  downloadCsv(rows, "skill-gap-report.csv");
}

function studentPdf() {
  const count = {};
  positions.forEach((r) => (r.missing || []).forEach((s) => (count[s] = (count[s] || 0) + 1)));
  const top = Object.entries(count).sort((a, b) => b[1] - a[1]).map(([s, n]) => `${s} (${n})`);
  const blocks = [{ h: "Most needed skills (number of positions)", lines: [top.join(", ") || "None"] }];
  positions.forEach((r) =>
    blocks.push({
      h: `${r.title} - ${r.company}`,
      lines: [
        `Match ${pct(r.score)}%  |  Semantic similarity ${r.semantic != null ? pct(r.semantic) + "%" : "n/a"}  |  Skill coverage ${coverage(r)}%`,
        `Matching skills: ${(r.matched || []).join(", ") || "none"}`,
        `Missing skills: ${(r.missing || []).join(", ") || "none"}`,
      ],
    })
  );
  downloadPdf("Skill Gap Report", blocks, "skill-gap-report.pdf");
}

$("findBtn").addEventListener("click", () =>
  run($("findBtn"), $("studentOut"), async () => {
    const fd = new FormData();
    if (resumeFile) fd.append("file", resumeFile);
    fd.append("text", $("resumeText").value);
    positions = (await post("/api/match-resume", fd)).sort((a, b) => b.score - a.score);
    renderStudent();
  })
);

/*  RECRUITER  */
let candidates = [];
const TOP_N = 10;
const filtered = () => candidates.filter((r) => r.score >= Number($("minMatch").value));

function renderRecruiter() {
  const out = $("recruiterOut");
  if (!candidates.length) return;
  const list = filtered();
  const min = Number($("minMatch").value);
  if (!list.length) { out.innerHTML = `<p class="msg">No candidates above ${min}%. Try a lower value.</p>`; return; }
  const shown = list.slice(0, TOP_N);
  out.innerHTML =
    `<p class="msg">Showing top ${shown.length} of ${list.length} candidates${list.length > TOP_N ? ". Download the list to see everyone." : ""}</p>` +
    shown.map((r) => card(r.name, "", r, false)).join("") +
    dlRow("Candidate list", `All ${list.length} candidates${min ? " above " + min + "%" : ""}`);
  animateBars(out);
  out.querySelector('[data-dl="csv"]').addEventListener("click", recruiterCsv);
  out.querySelector('[data-dl="pdf"]').addEventListener("click", recruiterPdf);
}

function recruiterCsv() {
  const rows = [["Rank", "Candidate", "Match %", "Matching skills"]];
  filtered().forEach((r, i) => rows.push([i + 1, r.name, pct(r.score), (r.matched || []).join("; ")]));
  downloadCsv(rows, "candidate-list.csv");
}

function recruiterPdf() {
  const list = filtered();
  const blocks = list.map((r, i) => ({
    h: `${i + 1}. ${r.name} - ${pct(r.score)}%`,
    lines: [`Matching skills: ${(r.matched || []).join(", ") || "none"}`],
  }));
  downloadPdf(`Candidate List (${list.length})`, blocks, "candidate-list.pdf");
}

$("minMatch").addEventListener("change", renderRecruiter);

$("rankBtn").addEventListener("click", () =>
  run($("rankBtn"), $("recruiterOut"), async () => {
    const fd = new FormData();
    if (jobFile) fd.append("file", jobFile);
    fd.append("text", $("jobText").value);
    candidates = (await post("/api/match-internship", fd)).sort((a, b) => b.score - a.score);
    if (!candidates.length) { $("recruiterOut").innerHTML = `<p class="msg">No candidates found.</p>`; return; }
    renderRecruiter();
  })
);
