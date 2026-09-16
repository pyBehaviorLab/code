/* Rig filter: show the pages that apply to the rig you actually have.
 *
 * The complaint this answers is "things are there but you have to search a
 * lot". The docs cover four rigs that share most of a codebase and almost
 * none of a shopping list: an operant box with no camera needs no GPU and
 * never opens the tracking dialog, while a maze needs both. Presented as one
 * flat list, every reader reads past three quarters of it.
 *
 * So each page declares which rigs it applies to, the sidebar gets a chooser,
 * and the pages that cannot apply to your rig are dimmed rather than removed:
 * hiding them would make the docs look smaller than they are and leave you
 * wondering where a page went.
 */
(function () {
  "use strict";

  /* Each rig carries what it takes to get that rig running, because that is
     the question the chooser is actually being asked. Dimming the pages you
     do not need answers "what can I ignore"; it never answered "what do I
     need, and what do I do first", which is what someone with the hardware
     on the bench in front of them wants. So a chosen rig now states the
     parts, the install steps for THAT option set, and the pages in order. */
  var RIGS = [
    { id: "operant", label: "Operant", hint: "Chamber only, no camera. No GPU.",
      title: "Operant chamber, no camera",
      needs: "One microcontroller per box, wired to the chamber and on USB. " +
             "No camera, no GPU.",
      install: [["Application", "installation.html#install-the-application"],
                ["Microcontroller", "installation.html#prepare-the-microcontroller"]],
      path: [["Getting started", "getting-started.html"],
             ["Projects", "user-guide/projects.html"],
             ["Writing tasks", "tasks/writing-tasks.html"],
             ["Boards", "user-guide/boards.html"]] },

    { id: "video", label: "+ Video", hint: "Chamber with video recording. No GPU.",
      title: "Operant chamber with video",
      needs: "The operant parts, plus one camera per box or a shared camera " +
             "split by region. Written straight to disk, so no GPU.",
      install: [["Application", "installation.html#install-the-application"],
                ["Microcontroller", "installation.html#prepare-the-microcontroller"]],
      path: [["Getting started", "getting-started.html"],
             ["Cameras", "user-guide/cameras.html"],
             ["Recording", "user-guide/recording.html"],
             ["Offline analysis", "user-guide/analysis.html"]] },

    { id: "pose", label: "+ Pose", hint: "Live tracking into the task. CUDA GPU.",
      title: "Operant chamber with pose tracking",
      needs: "The video parts, plus a CUDA GPU and a trained pose model. " +
             "The pose stack is a separate install and its versions are pinned.",
      install: [["Application", "installation.html#install-the-application"],
                ["Microcontroller", "installation.html#prepare-the-microcontroller"],
                ["Pose stack", "installation.html#add-pose-estimation"],
                ["Verify", "installation.html#verify"]],
      path: [["Getting started", "getting-started.html"],
             ["Cameras", "user-guide/cameras.html"],
             ["Tracking", "user-guide/tracking.html"],
             ["Cameras, tracking, zones",
              "user-guide/setup-cameras-tracking-zones.html"],
             ["Task API class", "user-guide/api-class.html"]] },

    { id: "maze", label: "Maze", hint: "Maze arena. Same needs as pose.",
      title: "Maze arena",
      needs: "One camera per arena and one microcontroller per arena, plus a " +
             "CUDA GPU and a trained pose model. Same install as pose tracking.",
      install: [["Application", "installation.html#install-the-application"],
                ["Microcontroller", "installation.html#prepare-the-microcontroller"],
                ["Pose stack", "installation.html#add-pose-estimation"],
                ["Verify", "installation.html#verify"]],
      path: [["Getting started", "getting-started.html"],
             ["Cameras", "user-guide/cameras.html"],
             ["Cameras, tracking, zones",
              "user-guide/setup-cameras-tracking-zones.html"],
             ["Tracking", "user-guide/tracking.html"],
             ["Writing tasks", "tasks/writing-tasks.html"]] }
  ];
  var ALL = ["operant", "video", "pose", "maze"];

  /* Page -> the rigs it applies to. Anything not listed applies to all of
     them, so a new page is visible until someone decides otherwise, which is
     the safe direction for a filter to fail in. */
  var TAGS = {
    "user-guide/cameras": ["video", "pose", "maze"],
    "user-guide/setup-cameras-tracking-zones": ["video", "pose", "maze"],
    "user-guide/recording": ["video", "pose", "maze"],
    "user-guide/analysis": ["video", "pose", "maze"],
    "user-guide/tracking": ["pose", "maze"],
    "user-guide/api-class": ["pose", "maze"],
    "concepts/frame-pipeline": ["video", "pose", "maze"],
    "concepts/tracking-pipeline": ["pose", "maze"],
    "hardware/operant-chamber": ["operant", "video", "pose"],
    "reference/performance": ["video", "pose", "maze"]
  };

  var KEY = "pbl-rig";
  var GPU = { operant: false, video: false, pose: true, maze: true };

  function docPath(href) {
    // "../user-guide/cameras.html#zones" -> "user-guide/cameras"
    if (!href) return "";
    var a = document.createElement("a");
    a.href = href;
    var p = a.pathname.replace(/\\/g, "/");
    var i = p.lastIndexOf("/_build/html/");
    p = i >= 0 ? p.slice(i + 13) : p.replace(/^.*\/html\//, "");
    return p.replace(/\.html$/, "").replace(/^\//, "");
  }

  function tagsFor(page) { return TAGS[page] || ALL; }

  function current() {
    try { return localStorage.getItem(KEY) || ""; } catch (e) { return ""; }
  }
  function store(v) {
    try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); }
    catch (e) { /* private mode: the filter just does not persist */ }
  }

  function apply(rig) {
    var links = document.querySelectorAll(".nav a.reference");
    for (var i = 0; i < links.length; i++) {
      var t = tagsFor(docPath(links[i].getAttribute("href")));
      var off = rig && t.indexOf(rig) === -1;
      links[i].classList.toggle("rig-off", !!off);
      if (off) links[i].setAttribute("title", "Not used on a " + rig + " rig");
      else links[i].removeAttribute("title");
    }
    var chips = document.querySelectorAll(".rig-chip");
    for (var j = 0; j < chips.length; j++) {
      chips[j].classList.toggle("on", chips[j].dataset.rig === rig);
      chips[j].setAttribute("aria-pressed", chips[j].dataset.rig === rig);
    }
    renderPanel(rig);
  }

  /* Relative path from this page back to the doc root.

     The panel links out to pages at fixed doc paths, but it is rendered on
     every page, and a page two directories deep needs "../../". Rather than
     count directories, take a link the toctree already wrote and subtract
     the document it points at: whatever is left is the prefix. */
  function baseUrl() {
    var a = document.querySelector(".nav a.reference");
    var href = a && a.getAttribute("href");
    if (!href) return "";
    var tail = docPath(href) + ".html";
    var i = href.indexOf(tail);
    return i >= 0 ? href.slice(0, i) : "";
  }

  function rigById(id) {
    for (var i = 0; i < RIGS.length; i++) {
      if (RIGS[i].id === id) return RIGS[i];
    }
    return null;
  }

  function linkRow(cls, label, pairs, sep) {
    var base = baseUrl();
    var row = document.createElement("div");
    row.className = cls;
    var k = document.createElement("span");
    k.className = "rig-k";
    k.textContent = label;
    row.appendChild(k);
    var v = document.createElement("span");
    v.className = "rig-v";
    pairs.forEach(function (pair, i) {
      if (i) v.appendChild(document.createTextNode(sep));
      var a = document.createElement("a");
      a.href = base + pair[1];
      a.textContent = pair[0];
      v.appendChild(a);
    });
    row.appendChild(v);
    return row;
  }

  function renderPanel(rig) {
    var host = document.getElementById("rig-note");
    if (!host) return;
    host.textContent = "";
    var r = rig && rigById(rig);
    if (!r) {
      host.className = "rig-note";
      host.textContent = "Pick one to see what it needs and where to start.";
      return;
    }
    host.className = "rig-note rig-ready";

    var h = document.createElement("div");
    h.className = "rig-title";
    h.textContent = r.title;
    host.appendChild(h);

    var needs = document.createElement("div");
    needs.className = "rig-needs";
    needs.textContent = r.needs;
    host.appendChild(needs);

    var gpu = document.createElement("div");
    gpu.className = "rig-gpu" + (GPU[r.id] ? " on" : "");
    gpu.textContent = GPU[r.id] ? "Needs a CUDA GPU" : "No GPU needed";
    host.appendChild(gpu);

    host.appendChild(linkRow("rig-row", "Install", r.install, " → "));
    host.appendChild(linkRow("rig-row", "Then", r.path, " → "));
  }

  function buildChooser() {
    var nav = document.getElementById("pbl-nav");
    if (!nav || !nav.parentNode) return;
    var box = document.createElement("div");
    box.className = "rig-filter";
    var head = document.createElement("div");
    head.className = "rig-head";
    head.textContent = "My rig";
    box.appendChild(head);
    var row = document.createElement("div");
    row.className = "rig-chips";
    RIGS.forEach(function (r) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "rig-chip";
      b.dataset.rig = r.id;
      b.textContent = r.label;
      b.title = r.hint;
      b.addEventListener("click", function () {
        var next = current() === r.id ? "" : r.id;
        store(next);
        apply(next);
      });
      row.appendChild(b);
    });
    box.appendChild(row);
    var note = document.createElement("div");
    note.className = "rig-note";
    note.id = "rig-note";
    box.appendChild(note);
    nav.parentNode.insertBefore(box, nav);
  }

  function stampPage() {
    var page = docPath(window.location.href);
    var t = tagsFor(page);
    if (t.length === ALL.length) return;   // applies everywhere, say nothing
    var art = document.querySelector(".content");
    var h1 = art && art.querySelector("h1");
    if (!h1) return;
    var wrap = document.createElement("div");
    wrap.className = "rig-badges";
    var lead = document.createElement("span");
    lead.className = "rig-badges-lead";
    lead.textContent = "Applies to";
    wrap.appendChild(lead);
    RIGS.forEach(function (r) {
      if (t.indexOf(r.id) === -1) return;
      var s = document.createElement("span");
      s.className = "rig-badge" + (GPU[r.id] ? " gpu" : "");
      s.textContent = r.label;
      s.title = r.hint;
      wrap.appendChild(s);
    });
    h1.parentNode.insertBefore(wrap, h1.nextSibling);
  }

  function init() {
    buildChooser();
    stampPage();
    apply(current());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
