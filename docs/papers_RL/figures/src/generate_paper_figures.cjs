const fs = require('fs');
const path = require('path');

const figureDir = path.resolve(__dirname, '..');
const dataDir = path.join(figureDir, 'data');

const palette = {
  ink: '#1C2738',
  muted: '#637185',
  grid: '#D7E0EA',
  soft: '#F7F9FC',
  blue: '#2F63D8',
  blueText: '#2456BE',
  blueSoft: '#EAF0FD',
  orange: '#D8891C',
  orangeText: '#A65D00',
  orangeSoft: '#FFF3E2',
  teal: '#138276',
  tealText: '#0D6D64',
  tealSoft: '#E8F5F2',
  coral: '#D95F59',
  coralText: '#B8433D',
  coralSoft: '#FFF1EF',
  gray: '#8B98A8',
  darkGray: '#4D5B6C',
  white: '#FFFFFF'
};

function readJson(file) {
  return JSON.parse(fs.readFileSync(path.join(dataDir, file), 'utf8'));
}

function metadata(value) {
  return JSON.stringify(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;');
}

function svgShell({ width, height, title, desc, meta, defs = '', body }) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby="title desc" focusable="false" shape-rendering="geometricPrecision" text-rendering="geometricPrecision">
  <title id="title">${title}</title>
  <desc id="desc">${desc}</desc>
  <metadata>${metadata(meta)}</metadata>
  <defs>
    <style>
      .t { font-family: "Helvetica Neue", "Source Sans 3", "Noto Sans", "Apple SD Gothic Neo", Arial, sans-serif; fill: ${palette.ink}; font-kerning: normal; }
      .panel { font-size: 22px; font-weight: 650; }
      .stage { font-size: 28px; font-weight: 700; letter-spacing: 0.2px; }
      .label { font-size: 24px; font-weight: 600; }
      .small { font-size: 20px; font-weight: 550; }
      .tiny { font-size: 18px; font-weight: 650; }
      .micro { font-size: 16px; font-weight: 650; }
      .value { font-size: 21px; font-weight: 700; }
      .focus-title { font-size: 20px; font-weight: 750; letter-spacing: -0.15px; }
      .muted { fill: ${palette.muted}; }
      .on-dark { fill: ${palette.white}; }
      .blue-text { fill: ${palette.blueText}; }
      .orange-text { fill: ${palette.orangeText}; }
      .teal-text { fill: ${palette.tealText}; }
      .coral-text { fill: ${palette.coralText}; }
      .dark-text { fill: ${palette.darkGray}; }
      .line { fill: none; stroke-linecap: round; stroke-linejoin: round; }
    </style>
    <marker id="arrow-ink" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="${palette.ink}"/></marker>
    <marker id="arrow-blue" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="${palette.blue}"/></marker>
    <marker id="arrow-orange" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="${palette.orange}"/></marker>
    <marker id="arrow-teal" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="${palette.teal}"/></marker>
    <marker id="arrow-blue-small" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="${palette.blue}"/></marker>
    <marker id="arrow-orange-small" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="${palette.orange}"/></marker>
    <marker id="arrow-coral-small" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="${palette.coral}"/></marker>
    <pattern id="hatch-coral" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(30)"><line x1="0" y1="0" x2="0" y2="9" stroke="${palette.coral}" stroke-width="3"/></pattern>
    ${defs}
  </defs>
  <rect width="${width}" height="${height}" fill="${palette.white}"/>
  ${body}
</svg>
`;
}

function promptGroupIcon(x, y, group, width = 116, height = 72) {
  return `<g class="prompt-group-icon">
    <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="11" fill="${palette.tealSoft}" stroke="${palette.teal}" stroke-width="2"/>
    <rect x="${x + 12}" y="${y + 15}" width="36" height="${height - 30}" rx="8" fill="${palette.teal}"/>
    <text x="${x + 30}" y="${y + height / 2 + 6}" text-anchor="middle" class="t tiny on-dark">${group}</text>
    <path d="M${x + 59} ${y + height / 2 - 7} H${x + width - 16} M${x + 59} ${y + height / 2 + 7} H${x + width - 27}" class="line" stroke="${palette.darkGray}" stroke-width="2.6" opacity="0.72"/>
  </g>`;
}

function groupTray(x, y, group, filled = 4, width = 126, height = 72) {
  const slotWidth = 18;
  const gap = 6;
  const slots = Array.from({ length: 4 }, (_, index) => {
    const sx = x + 13 + index * (slotWidth + gap);
    const active = index < filled;
    return `<rect x="${sx}" y="${y + 35}" width="${slotWidth}" height="21" rx="5" fill="${active ? palette.blue : palette.white}" stroke="${active ? palette.blue : palette.gray}" stroke-width="1.7" ${active ? '' : 'stroke-dasharray="4 4"'}/>`;
  }).join('');
  return `<g class="group-tray">
    <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="13" fill="${filled === 4 ? palette.tealSoft : palette.coralSoft}" stroke="${filled === 4 ? palette.teal : palette.coral}" stroke-width="2"/>
    <rect x="${x + 12}" y="${y + 9}" width="34" height="21" rx="7" fill="${filled === 4 ? palette.teal : palette.coral}"/>
    <text x="${x + 29}" y="${y + 25}" text-anchor="middle" class="t tiny on-dark">${group}</text>
    ${slots}
  </g>`;
}

function rolloutStackIcon(x, y, width = 116, height = 70) {
  return `<g class="rollout-stack-icon">
    <rect x="${x + 7}" y="${y}" width="${width - 7}" height="${height - 7}" rx="9" fill="${palette.white}" stroke="${palette.blue}" stroke-width="1.5" opacity="0.7"/>
    <rect x="${x}" y="${y + 7}" width="${width - 7}" height="${height - 7}" rx="9" fill="${palette.blueSoft}" stroke="${palette.blue}" stroke-width="2"/>
    <path d="M${x + 15} ${y + 27} H${x + width - 24} M${x + 15} ${y + 42} H${x + width - 38}" class="line" stroke="${palette.blue}" stroke-width="2.8"/>
  </g>`;
}

function expertDocumentIcon(x, y, width = 104, height = 72) {
  return `<g class="expert-document-icon">
    <path d="M${x} ${y + 10} Q${x} ${y} ${x + 10} ${y} H${x + width - 24} L${x + width} ${y + 24} V${y + height - 10} Q${x + width} ${y + height} ${x + width - 10} ${y + height} H${x + 10} Q${x} ${y + height} ${x} ${y + height - 10} Z" fill="${palette.orangeSoft}" stroke="${palette.orange}" stroke-width="2"/>
    <path d="M${x + width - 24} ${y} V${y + 24} H${x + width}" class="line" stroke="${palette.orange}" stroke-width="2"/>
    <path d="M${x + 16} ${y + 32} H${x + width - 20} M${x + 16} ${y + 47} H${x + width - 34}" class="line" stroke="${palette.orange}" stroke-width="3"/>
  </g>`;
}

function figure1() {
  const body = `
  <g id="naive-trade-off">
    <text x="34" y="54" class="t panel muted">(a)</text>
    <text x="78" y="58" class="t stage">Naive trade-off</text>
    <path d="M78 75 H165" class="line" stroke="${palette.coral}" stroke-width="4"/>

    <text x="276" y="211" text-anchor="middle" class="t small muted">Rollout attempts</text>
    ${promptGroupIcon(44, 262, 'g', 100, 60)}
    <text x="94" y="351" text-anchor="middle" class="t tiny muted">Prompt</text>
    <path d="M158 250 H250 M158 276 H292 M158 302 H336 M158 328 H355" class="line" stroke="${palette.blue}" stroke-width="4"/>
    <circle cx="250" cy="250" r="5" fill="${palette.blue}"/>
    <circle cx="292" cy="276" r="5" fill="${palette.blue}"/>
    <circle cx="336" cy="302" r="5" fill="${palette.blue}"/>
    <path d="M250 250 H398 M292 276 H398 M336 302 H398 M355 328 H398" class="line" stroke="${palette.gray}" stroke-width="2.3" stroke-dasharray="6 7" opacity="0.62"/>
    <path d="M398 242 V336 M398 289 H410" class="line" stroke="${palette.darkGray}" stroke-width="2"/>
    <circle cx="413" cy="289" r="10" fill="${palette.white}" stroke="${palette.darkGray}" stroke-width="2"/>

    <g id="wait-outcome">
      <text x="481" y="126" text-anchor="middle" class="t label coral-text">Wait</text>
      <path d="M422 283 C454 274 448 175 474 175" class="line" stroke="${palette.coral}" stroke-width="3" marker-end="url(#arrow-coral-small)"/>
      <path d="M482 146 V215" class="line" stroke="${palette.coral}" stroke-width="4"/>
      <rect x="503" y="148" width="72" height="64" rx="9" fill="${palette.coralSoft}" stroke="${palette.coral}" stroke-width="2"/>
      <path d="M519 164 H559 M519 177 H550" class="line" stroke="${palette.coral}" stroke-width="2.5"/>
      <circle cx="539" cy="197" r="6" fill="${palette.coral}"/>
      <text x="539" y="242" text-anchor="middle" class="t tiny coral-text">Learner idle</text>
    </g>

    <g id="early-outcome">
      <text x="486" y="396" text-anchor="middle" class="t label coral-text">Decide early</text>
      <path d="M422 296 C455 307 444 451 456 451" class="line" stroke="${palette.coral}" stroke-width="3" marker-end="url(#arrow-coral-small)"/>
      ${groupTray(462, 416, 'g', 3, 108, 72)}
      <path d="M571 452 H576" class="line" stroke="${palette.coral}" stroke-width="3"/>
      <path d="M596 430 L618 452 L596 474 L574 452 Z" fill="${palette.coralSoft}" stroke="${palette.coral}" stroke-width="2.2"/>
      <path d="M588 444 L604 460 M604 444 L588 460" class="line" stroke="${palette.coral}" stroke-width="3"/>
      <text x="535" y="523" text-anchor="middle" class="t tiny coral-text">Wrong source</text>
    </g>
  </g>

  <line x1="640" y1="24" x2="640" y2="576" stroke="${palette.grid}" stroke-width="1.4"/>

  <g id="streamweave">
    <text x="674" y="54" class="t panel muted">(b)</text>
    <text x="718" y="58" class="t stage">StreamWeave</text>
    <path d="M718 75 H805" class="line" stroke="${palette.teal}" stroke-width="4"/>
    <text x="830" y="111" text-anchor="middle" class="t tiny muted">Independent attempts</text>
    <text x="1028" y="111" text-anchor="middle" class="t tiny muted">Complete groups</text>
    <text x="1434" y="111" text-anchor="middle" class="t tiny muted">Update</text>

    <g id="streamweave-group-1">
      ${promptGroupIcon(674, 145, 'g1', 84, 54)}
      <path d="M776 151 H830 M776 166 H866 M776 181 H904 M776 196 H938" class="line" stroke="${palette.blue}" stroke-width="3.5"/>
      <circle cx="830" cy="151" r="4.5" fill="${palette.blue}"/><circle cx="866" cy="166" r="4.5" fill="${palette.blue}"/>
      <circle cx="904" cy="181" r="4.5" fill="${palette.blue}"/><circle cx="938" cy="196" r="4.5" fill="${palette.blue}"/>
      ${groupTray(964, 142, 'g1', 4, 126, 70)}
      <path d="M1092 177 H1112" class="line" stroke="${palette.teal}" stroke-width="3" marker-end="url(#arrow-teal)"/>
      <path d="M1140 149 L1168 177 L1140 205 L1112 177 Z" fill="${palette.teal}"/>
      <path d="M1127 177 H1140 M1140 177 L1153 166 M1140 177 L1153 188" class="line" stroke="${palette.white}" stroke-width="3"/>
      <path d="M1168 177 H1188" class="line" stroke="${palette.blue}" stroke-width="3.2" marker-end="url(#arrow-blue-small)"/>
      ${rolloutStackIcon(1192, 145, 88, 64)}
      <text x="1236" y="231" text-anchor="middle" class="t tiny blue-text">Policy</text>
      <path d="M1282 177 H1338" class="line" stroke="${palette.blue}" stroke-width="3" marker-end="url(#arrow-blue-small)"/>
      <rect x="1342" y="143" width="152" height="68" rx="8" fill="${palette.blueSoft}" stroke="${palette.blue}" stroke-width="1.8"/>
      <text x="1418" y="184" text-anchor="middle" class="t label blue-text">RL update</text>
    </g>

    <g id="streamweave-group-2">
      ${promptGroupIcon(674, 305, 'g2', 84, 54)}
      <path d="M776 311 H816 M776 326 H854 M776 341 H894 M776 356 H928" class="line" stroke="${palette.blue}" stroke-width="3.5"/>
      <circle cx="816" cy="311" r="4.5" fill="${palette.blue}"/><circle cx="854" cy="326" r="4.5" fill="${palette.blue}"/>
      <circle cx="894" cy="341" r="4.5" fill="${palette.blue}"/><circle cx="928" cy="356" r="4.5" fill="${palette.blue}"/>
      ${groupTray(964, 302, 'g2', 4, 126, 70)}
      <path d="M1092 337 H1112" class="line" stroke="${palette.teal}" stroke-width="3" marker-end="url(#arrow-teal)"/>
      <path d="M1140 309 L1168 337 L1140 365 L1112 337 Z" fill="${palette.teal}"/>
      <path d="M1127 337 H1140 M1140 337 L1153 326 M1140 337 L1153 348" class="line" stroke="${palette.white}" stroke-width="3"/>
      <path d="M1168 337 H1190" class="line" stroke="${palette.orange}" stroke-width="3.2" marker-end="url(#arrow-orange-small)"/>
      ${expertDocumentIcon(1194, 303, 84, 68)}
      <text x="1236" y="393" text-anchor="middle" class="t tiny orange-text">Expert</text>
      <path d="M1280 337 H1338" class="line" stroke="${palette.orange}" stroke-width="3" marker-end="url(#arrow-orange-small)"/>
      <rect x="1342" y="303" width="152" height="68" rx="8" fill="${palette.orangeSoft}" stroke="${palette.orange}" stroke-width="1.8"/>
      <text x="1418" y="344" text-anchor="middle" class="t label orange-text">SFT update</text>
    </g>

    <g id="streamweave-group-3-running">
      ${promptGroupIcon(674, 465, 'g3', 84, 54)}
      <path d="M776 471 H824 M776 486 H870 M776 501 H912 M776 516 H898" class="line" stroke="${palette.blue}" stroke-width="3.5"/>
      <circle cx="824" cy="471" r="4.5" fill="${palette.blue}"/><circle cx="870" cy="486" r="4.5" fill="${palette.blue}"/>
      <circle cx="912" cy="501" r="4.5" fill="${palette.blue}"/>
      <path d="M898 516 H938" class="line" stroke="${palette.gray}" stroke-width="2.3" stroke-dasharray="6 7"/>
      ${groupTray(964, 462, 'g3', 3, 126, 70)}
      <text x="1027" y="558" text-anchor="middle" class="t tiny muted">In progress</text>
    </g>

    <g id="source-aware-merge">
      <path d="M1494 177 H1524 V240 M1494 337 H1524 V274" class="line" stroke="${palette.teal}" stroke-width="2.8"/>
      <circle cx="1524" cy="257" r="16" fill="${palette.tealSoft}" stroke="${palette.teal}" stroke-width="2"/>
      <path d="M1516 257 H1532 M1524 249 V265" class="line" stroke="${palette.teal}" stroke-width="2.6"/>
      <path d="M1524 274 V432" class="line" stroke="${palette.ink}" stroke-width="3" marker-end="url(#arrow-ink)"/>
      <rect x="1462" y="438" width="124" height="60" rx="8" fill="${palette.ink}"/>
      <circle cx="1486" cy="468" r="11" fill="${palette.white}"/>
      <text x="1534" y="475" text-anchor="middle" class="t tiny on-dark">Model</text>
    </g>
  </g>`;

  return svgShell({
    width: 1600,
    height: 600,
    title: 'StreamWeave overview',
    desc: 'Panel a shows the naive trade-off: waiting for an incomplete prompt group idles the learner, while deciding early can select the wrong training source. Panel b shows StreamWeave advancing rollout attempts independently, making source decisions only for complete groups, and updating the model while another group remains in progress.',
    meta: {
      figure_id: 'figure1_streamweave_overview',
      status: 'ready',
      visible_text_policy: 'short component and state labels; no explanatory prose'
    },
    body
  });
}

function figure2TrainingPipeline() {
  const body = `
  <g id="rollout-plane">
    <path d="M36 52 H58" class="line" stroke="${palette.blue}" stroke-width="5"/>
    <text x="72" y="60" class="t panel">Fully-asynchronous rollout</text>
    <line x1="36" y1="82" x2="646" y2="82" stroke="${palette.grid}" stroke-width="1.2"/>

    <g id="prompt-groups">
      <text x="90" y="132" text-anchor="middle" class="t tiny muted">Prompts</text>
      ${promptGroupIcon(42, 160, 'g1', 96, 58)}
      ${promptGroupIcon(42, 270, 'g2', 96, 58)}
      ${promptGroupIcon(42, 380, 'g3', 96, 58)}
    </g>

    <g id="attempt-scheduler">
      <text x="188" y="132" text-anchor="middle" class="t tiny muted">Dispatch</text>
      <circle cx="188" cy="299" r="11" fill="${palette.tealSoft}" stroke="${palette.teal}" stroke-width="2"/>
      <path d="M140 189 C164 189 156 299 176 299 M140 299 H176 M140 409 C164 409 156 299 176 299" class="line" stroke="${palette.teal}" stroke-width="2.8"/>
      <path d="M200 299 C220 299 218 193 248 193 M200 299 H248 M200 299 C220 299 218 409 248 409" class="line" stroke="${palette.blue}" stroke-width="2.8" marker-end="url(#arrow-blue-small)"/>
    </g>

    <g id="rollouter-pool">
      <text x="330" y="132" text-anchor="middle" class="t tiny muted">Rollout workers</text>
      <rect x="250" y="158" width="160" height="70" rx="8" fill="${palette.blueSoft}" stroke="${palette.blue}" stroke-width="1.5"/>
      <circle cx="273" cy="193" r="11" fill="${palette.ink}"/>
      <path d="M296 183 H352 M296 202 H386" class="line" stroke="${palette.blue}" stroke-width="3.4"/>
      <circle cx="386" cy="202" r="5" fill="${palette.teal}"/>

      <rect x="250" y="268" width="160" height="70" rx="8" fill="${palette.blueSoft}" stroke="${palette.blue}" stroke-width="1.5"/>
      <circle cx="273" cy="303" r="11" fill="${palette.ink}"/>
      <path d="M296 293 H382 M296 312 H350" class="line" stroke="${palette.blue}" stroke-width="3.4"/>
      <circle cx="350" cy="312" r="5" fill="${palette.teal}"/>

      <rect x="250" y="378" width="160" height="70" rx="8" fill="${palette.blueSoft}" stroke="${palette.blue}" stroke-width="1.5"/>
      <circle cx="273" cy="413" r="11" fill="${palette.ink}"/>
      <path d="M296 403 H340 M296 422 H390" class="line" stroke="${palette.blue}" stroke-width="3.4"/>
      <circle cx="390" cy="422" r="5" fill="${palette.teal}"/>
    </g>

    <g id="group-reconstruction">
      <text x="531" y="132" text-anchor="middle" class="t tiny muted">Reconstruct</text>
      <path d="M412 193 H468 M412 303 H468 M412 413 H468" class="line" stroke="${palette.teal}" stroke-width="2.8" marker-end="url(#arrow-teal)"/>
      ${groupTray(470, 155, 'g1', 4, 122, 70)}
      ${groupTray(470, 265, 'g2', 4, 122, 70)}
      ${groupTray(470, 375, 'g3', 3, 122, 70)}
      <path d="M592 190 H614 Q630 190 630 206 V245 M592 300 H614 Q630 300 630 284 V245" class="line" stroke="${palette.teal}" stroke-width="2.8"/>
      <circle cx="630" cy="245" r="5" fill="${palette.teal}" stroke="${palette.white}" stroke-width="1.5"/>
    </g>
  </g>

  <g id="composition-plane">
    <path d="M674 52 H700" class="line" stroke="${palette.teal}" stroke-width="6"/>
    <text x="714" y="60" class="t focus-title">Provenance-aware composition</text>
    <line x1="674" y1="82" x2="1108" y2="82" stroke="${palette.grid}" stroke-width="1.2"/>

    <g id="source-choice">
      <text x="700" y="132" text-anchor="middle" class="t tiny muted">Select source</text>
      <path d="M630 245 H663" class="line" stroke="${palette.teal}" stroke-width="3" marker-end="url(#arrow-teal)"/>
      <path d="M700 208 L737 245 L700 282 L663 245 Z" fill="${palette.teal}"/>
      <path d="M684 245 H699 M699 245 L716 232 M699 245 L716 258" class="line" stroke="${palette.white}" stroke-width="3.4"/>
      <path d="M737 235 C751 235 751 192 768 192" class="line" stroke="${palette.blue}" stroke-width="3.2" marker-end="url(#arrow-blue-small)"/>
      <path d="M700 282 V380 H768" class="line" stroke="${palette.orange}" stroke-width="3.2" marker-end="url(#arrow-orange-small)"/>
    </g>

    <g id="selected-policy-payload">
      ${rolloutStackIcon(772, 157, 96, 70)}
      <text x="820" y="252" text-anchor="middle" class="t tiny blue-text">Policy group</text>
      <path d="M870 192 H994 V260 H1018" class="line" stroke="${palette.blue}" stroke-width="2.8"/>
    </g>

    <g id="expert-store-and-payload">
      ${expertDocumentIcon(772, 345, 96, 72)}
      <text x="820" y="443" text-anchor="middle" class="t tiny orange-text">Expert trajectory</text>
      <path d="M870 381 H994 V318 H1018" class="line" stroke="${palette.orange}" stroke-width="2.8"/>
    </g>

    <g id="bounded-mixed-stream">
      <rect x="1020" y="228" width="42" height="120" rx="6" fill="${palette.ink}"/>
      <rect x="1028" y="239" width="26" height="20" rx="4" fill="${palette.blue}"/>
      <rect x="1028" y="265" width="26" height="20" rx="4" fill="${palette.orange}"/>
      <rect x="1028" y="291" width="26" height="20" rx="4" fill="${palette.blue}"/>
      <rect x="1028" y="317" width="26" height="20" rx="4" fill="${palette.orange}"/>
      <text x="1041" y="378" text-anchor="middle" class="t tiny">Stream</text>
    </g>

    <g id="provenance-context">
      <text x="891" y="477" text-anchor="middle" class="t small teal-text">Provenance</text>
      <line x1="758" y1="493" x2="1041" y2="493" stroke="${palette.teal}" stroke-width="2.2"/>
      <text x="891" y="520" text-anchor="middle" class="t tiny muted">group / source / policy</text>
      <path d="M1041 493 H1076 V350 H1064" class="line" stroke="${palette.teal}" stroke-width="2" stroke-dasharray="5 6"/>
    </g>
  </g>

  <g id="learning-plane">
    <path d="M1138 52 H1160" class="line" stroke="${palette.ink}" stroke-width="5"/>
    <text x="1174" y="60" class="t panel">Source-aware update</text>
    <line x1="1138" y1="82" x2="1572" y2="82" stroke="${palette.grid}" stroke-width="1.2"/>

    <path d="M1062 288 H1122" class="line" stroke="${palette.ink}" stroke-width="3" marker-end="url(#arrow-ink)"/>
    <circle cx="1138" cy="288" r="11" fill="${palette.white}" stroke="${palette.darkGray}" stroke-width="2"/>
    <path d="M1149 288 C1168 288 1160 219 1188 219 M1149 288 C1168 288 1160 369 1188 369" class="line" stroke="${palette.darkGray}" stroke-width="2.8"/>

    <rect x="1190" y="180" width="178" height="78" rx="8" fill="${palette.blueSoft}" stroke="${palette.blue}" stroke-width="1.7"/>
    <text x="1279" y="227" text-anchor="middle" class="t label blue-text">RL operators</text>

    <rect x="1190" y="330" width="178" height="78" rx="8" fill="${palette.orangeSoft}" stroke="${palette.orange}" stroke-width="1.7"/>
    <text x="1279" y="377" text-anchor="middle" class="t label orange-text">SFT update</text>

    <path d="M1368 219 H1398 Q1418 219 1418 239 V288 M1368 369 H1398 Q1418 369 1418 349 V288" class="line" stroke="${palette.teal}" stroke-width="2.8"/>
    <circle cx="1418" cy="288" r="16" fill="${palette.tealSoft}" stroke="${palette.teal}" stroke-width="2"/>
    <path d="M1410 288 H1426 M1418 280 V296" class="line" stroke="${palette.teal}" stroke-width="2.6"/>
    <path d="M1434 288 H1458" class="line" stroke="${palette.ink}" stroke-width="3" marker-end="url(#arrow-ink)"/>
    <rect x="1462" y="253" width="116" height="70" rx="8" fill="${palette.ink}"/>
    <circle cx="1487" cy="288" r="12" fill="${palette.white}"/>
    <text x="1535" y="295" text-anchor="middle" class="t tiny on-dark">Model</text>
  </g>

  <g id="control-plane">
    <path d="M1520 323 V550 H330 V450" class="line" stroke="${palette.gray}" stroke-width="2" stroke-dasharray="8 8" marker-end="url(#arrow-ink)"/>
    <rect x="808" y="534" width="188" height="30" fill="${palette.white}"/>
    <text x="902" y="558" text-anchor="middle" class="t small muted">parameter refresh</text>
  </g>`;

  return svgShell({
    width: 1600,
    height: 610,
    title: 'StreamWeave end-to-end training pipeline',
    desc: 'Prompt groups are dispatched into independent rollout workers and reconstructed at complete-group context. A source selector emits either policy rollouts or an expert trajectory, preserves provenance through a mixed stream, and applies source-specific operators before updating the model. A separate control path refreshes rollout workers.',
    meta: {
      figure_id: 'figure2_training_pipeline',
      status: 'ready',
      visible_text_policy: 'component identity and short state labels only'
    },
    body
  });
}

function figure2(data) {
  const width = 640;
  const height = 390;
  const left = 140;
  const right = 610;
  const min = 34;
  const max = 40;
  const mapX = (value) => left + ((value - min) / (max - min)) * (right - left);
  const rows = [130, 225];

  const ticks = [];
  for (let value = min; value <= max; value += 1) {
    const x = mapX(value);
    ticks.push(`<line x1="${x}" y1="78" x2="${x}" y2="286" stroke="${palette.grid}" stroke-width="1"/>`);
    ticks.push(`<text x="${x}" y="322" text-anchor="middle" class="t small muted">${value}</text>`);
  }

  const rowSvg = data.windows.map((window, index) => {
    const y = rows[index];
    const xRl = mapX(window.rl_only);
    const xSw = mapX(window.streamweave);
    const rlLabel = window.rl_only.toFixed(1);
    const swLabel = window.streamweave.toFixed(1);
    return `<g id="window-${window.label}">
      <text x="116" y="${y + 7}" text-anchor="end" class="t label">${window.label}</text>
      <line x1="${xRl}" y1="${y}" x2="${xSw}" y2="${y}" stroke="${palette.gray}" stroke-width="3"/>
      <path d="M${xRl} ${y - 9} L${xRl + 9} ${y} L${xRl} ${y + 9} L${xRl - 9} ${y} Z" fill="${palette.white}" stroke="${palette.darkGray}" stroke-width="2.4"/>
      <circle cx="${xSw}" cy="${y}" r="9" fill="${palette.blue}" stroke="${palette.white}" stroke-width="2"/>
      <text x="${xRl - 14}" y="${y - 16}" text-anchor="end" class="t value dark-text">${rlLabel}</text>
      <text x="${xSw + 14}" y="${y - 16}" text-anchor="start" class="t value blue-text">${swLabel}</text>
    </g>`;
  }).join('\n');

  const body = `
  <g id="legend">
    <circle cx="270" cy="39" r="8" fill="${palette.blue}"/><text x="288" y="46" class="t small">StreamWeave</text>
    <path d="M488 31 L496 39 L488 47 L480 39 Z" fill="${palette.white}" stroke="${palette.darkGray}" stroke-width="2.2"/><text x="506" y="46" class="t small">RL-only</text>
  </g>
  <g id="axis">
    ${ticks.join('\n')}
    <line x1="${left}" y1="286" x2="${right}" y2="286" stroke="${palette.ink}" stroke-width="1.5"/>
    <text x="${(left + right) / 2}" y="370" text-anchor="middle" class="t label muted">score</text>
  </g>
  <g id="windowed-comparison">
    ${rowSvg}
  </g>`;

  return svgShell({
    width,
    height,
    title: 'Windowed learning effect of expert supervision',
    desc: 'A paired dot plot comparing StreamWeave and an RL-only control in an early and a late training window. The gap is small early and larger late.',
    meta: data,
    body
  });
}

function figure3(data) {
  const width = 1500;
  const height = 600;
  const timelineX = 205;
  const timelineW = 590;
  const timelineMax = 50;
  const mapTime = (seconds) => timelineX + (seconds / timelineMax) * timelineW;
  const sync = data.timeline_seconds.sync;
  const async = data.timeline_seconds.streamweave;
  const generationEnd = mapTime(sync.generation);
  const activeGenerationEnd = mapTime(sync.generation - sync.generation_tail_wait);
  const learningEnd = mapTime(sync.generation + sync.learning);
  const syncEnd = mapTime(sync.total);
  const asyncEnd = mapTime(async.total);

  const timeTicks = [];
  for (let value = 0; value <= timelineMax; value += 10) {
    const x = mapTime(value);
    timeTicks.push(`<line x1="${x}" y1="126" x2="${x}" y2="420" stroke="${palette.grid}" stroke-width="1"/>`);
    timeTicks.push(`<text x="${x}" y="452" text-anchor="middle" class="t small muted">${value}</text>`);
  }

  const metricLeft = 1005;
  const metricRight = 1430;
  const metricRows = [160, 300, 440];
  const metricSvg = data.metrics.map((metric, index) => {
    const y = metricRows[index];
    const mapMetric = (value) => metricLeft + (value / metric.max) * (metricRight - metricLeft);
    const xSync = mapMetric(metric.sync);
    const xSw = mapMetric(metric.streamweave);
    const syncText = metric.sync.toFixed(metric.decimals);
    const swText = metric.streamweave.toFixed(metric.decimals);
    return `<g id="metric-${index + 1}">
      <text x="965" y="${y + 6}" text-anchor="end" class="t label">${metric.label}</text>
      <line x1="${metricLeft}" y1="${y}" x2="${metricRight}" y2="${y}" stroke="${palette.grid}" stroke-width="2"/>
      <line x1="${Math.min(xSync, xSw)}" y1="${y}" x2="${Math.max(xSync, xSw)}" y2="${y}" stroke="${palette.teal}" stroke-width="3"/>
      <path d="M${xSync} ${y - 9} L${xSync + 9} ${y} L${xSync} ${y + 9} L${xSync - 9} ${y} Z" fill="${palette.white}" stroke="${palette.darkGray}" stroke-width="2.3"/>
      <circle cx="${xSw}" cy="${y}" r="9" fill="${palette.teal}" stroke="${palette.white}" stroke-width="2"/>
      <text x="${xSync}" y="${y - 20}" text-anchor="middle" class="t value dark-text">${syncText}</text>
      <text x="${xSw}" y="${y + 29}" text-anchor="middle" class="t value teal-text">${swText}</text>
    </g>`;
  }).join('\n');

  const body = `
  <g id="timeline-panel">
    <text x="34" y="50" class="t panel muted">(a)</text>
    ${timeTicks.join('\n')}
    <line x1="${timelineX}" y1="420" x2="${mapTime(timelineMax)}" y2="420" stroke="${palette.ink}" stroke-width="1.5"/>
    <text x="${mapTime(timelineMax) + 18}" y="425" class="t small muted">s</text>

    <text x="175" y="190" text-anchor="end" class="t label">Sync</text>
    <rect x="${timelineX}" y="160" width="${generationEnd - timelineX}" height="48" rx="8" fill="${palette.orange}"/>
    <rect x="${activeGenerationEnd}" y="160" width="${generationEnd - activeGenerationEnd}" height="48" fill="url(#hatch-coral)" opacity="0.95"/>
    <rect x="${generationEnd}" y="160" width="${learningEnd - generationEnd}" height="48" fill="${palette.blue}"/>
    <rect x="${learningEnd}" y="160" width="${syncEnd - learningEnd}" height="48" rx="0 8 8 0" fill="${palette.gray}"/>
    <text x="${syncEnd + 14}" y="191" class="t value">46</text>

    <text x="175" y="327" text-anchor="end" class="t label">StreamWeave</text>
    <rect x="${timelineX}" y="286" width="${asyncEnd - timelineX}" height="34" rx="8" fill="${palette.orange}" opacity="0.9"/>
    <rect x="${timelineX}" y="330" width="${asyncEnd - timelineX}" height="34" rx="8" fill="${palette.blue}" opacity="0.9"/>
    <text x="${asyncEnd + 14}" y="334" class="t value">30</text>

    <g id="timeline-legend">
      <rect x="205" y="512" width="16" height="16" rx="3" fill="${palette.orange}"/><text x="230" y="525" class="t small muted">generation</text>
      <rect x="345" y="512" width="16" height="16" rx="3" fill="${palette.blue}"/><text x="370" y="525" class="t small muted">learning</text>
      <rect x="470" y="512" width="16" height="16" rx="3" fill="${palette.gray}"/><text x="495" y="525" class="t small muted">sync</text>
      <rect x="558" y="512" width="16" height="16" fill="url(#hatch-coral)" stroke="${palette.coral}" stroke-width="1"/><text x="583" y="525" class="t small muted">tail wait</text>
    </g>
  </g>

  <line x1="875" y1="32" x2="875" y2="568" stroke="${palette.grid}" stroke-width="1.2"/>

  <g id="metric-panel">
    <text x="906" y="50" class="t panel muted">(b)</text>
    <g id="metric-legend">
      <path d="M1192 38 L1200 46 L1192 54 L1184 46 Z" fill="${palette.white}" stroke="${palette.darkGray}" stroke-width="2.1"/><text x="1210" y="52" class="t small">Sync</text>
      <circle cx="1290" cy="46" r="8" fill="${palette.teal}"/><text x="1307" y="52" class="t small">StreamWeave</text>
    </g>
    ${metricSvg}
  </g>`;

  return svgShell({
    width,
    height,
    title: 'Matched execution efficiency',
    desc: 'A same-hardware comparison showing serial synchronous phases versus overlapped StreamWeave execution for 128 prompt groups, alongside throughput, trainer idle, and actor MFU.',
    meta: data,
    body
  });
}

const outputs = [
  ['figure1_streamweave_overview.svg', figure1()],
  ['figure2_training_pipeline.svg', figure2TrainingPipeline()],
  ['figure2_learning_effect.svg', figure2(readJson('figure2_learning_effect.json'))],
  ['figure3_execution_efficiency.svg', figure3(readJson('figure3_execution_efficiency.json'))]
];

const requested = new Set(process.argv.slice(2).map((file) => path.basename(file, path.extname(file))));
const selectedOutputs = requested.size === 0
  ? outputs
  : outputs.filter(([file]) => requested.has(path.basename(file, '.svg')));

if (requested.size > 0 && selectedOutputs.length !== requested.size) {
  const known = outputs.map(([file]) => path.basename(file, '.svg')).join(', ');
  throw new Error(`Unknown figure name. Available figures: ${known}`);
}

for (const [file, content] of selectedOutputs) {
  fs.writeFileSync(path.join(figureDir, file), content, 'utf8');
  process.stdout.write(`wrote ${file}\n`);
}
