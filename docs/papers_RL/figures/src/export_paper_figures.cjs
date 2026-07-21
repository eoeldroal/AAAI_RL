const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');

const figureDir = path.resolve(__dirname, '..');
const availableFigures = [
  'figure1_streamweave_overview',
  'figure2_training_pipeline',
  'figure2_learning_effect',
  'figure3_execution_efficiency'
];
const requested = new Set(process.argv.slice(2).map((file) => path.basename(file, path.extname(file))));
const figures = requested.size === 0
  ? availableFigures
  : availableFigures.filter((figure) => requested.has(figure));
const chromePath = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

if (requested.size > 0 && figures.length !== requested.size) {
  throw new Error(`Unknown figure name. Available figures: ${availableFigures.join(', ')}`);
}

function dimensions(svgPath) {
  const source = fs.readFileSync(svgPath, 'utf8');
  const width = Number(source.match(/<svg[^>]* width="([0-9.]+)"/)?.[1]);
  const height = Number(source.match(/<svg[^>]* height="([0-9.]+)"/)?.[1]);
  if (!width || !height) {
    throw new Error(`Missing SVG dimensions: ${svgPath}`);
  }
  return { width, height };
}

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: chromePath });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 900 },
    deviceScaleFactor: 2
  });
  const page = await context.newPage();
  try {
    for (const figure of figures) {
      const svgPath = path.join(figureDir, `${figure}.svg`);
      const { width, height } = dimensions(svgPath);
      await page.setViewportSize({ width, height });
      await page.goto(pathToFileURL(svgPath).href, { waitUntil: 'load' });
      await page.screenshot({
        path: path.join(figureDir, `${figure}.png`),
        clip: { x: 0, y: 0, width, height },
        omitBackground: false
      });
      await page.pdf({
        path: path.join(figureDir, `${figure}.pdf`),
        width: `${width}px`,
        height: `${height}px`,
        printBackground: true,
        margin: { top: '0px', right: '0px', bottom: '0px', left: '0px' },
        pageRanges: '1'
      });
      process.stdout.write(`exported ${figure}.png/.pdf\n`);
    }
  } finally {
    await context.close();
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
