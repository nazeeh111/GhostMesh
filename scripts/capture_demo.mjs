#!/usr/bin/env node
/** Record the actual replay viewer at explicit source timestamps, without authored pose motion. */
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2),
  options = {
    fps: 10,
    seconds: 10,
    start: 1,
    width: 1440,
    height: 1480,
    python: "python",
    ffmpeg: "ffmpeg",
  };
for (let i = 0; i < args.length; i++) {
  const key = args[i].replace(/^--/, "");
  if (key === "reference") {
    options.reference = true;
    continue;
  }
  if (key === "help") {
    console.log(
      "node scripts/capture_demo.mjs --replay FILE --out NEW_DIRECTORY [--python .venv/bin/python] [--playwright-module PATH_OR_PACKAGE] [--browser EXECUTABLE] [--fps 10] [--seconds 10] [--start 1] [--reference]",
    );
    process.exit(0);
  }
  if (!args[i].startsWith("--") || !args[i + 1])
    throw new Error(`Unknown/incomplete option ${args[i]}`);
  options[key] = args[++i];
}
for (const key of ["fps", "seconds", "start", "width", "height"])
  options[key] = Number(options[key]);
if (!options.replay || !options.out)
  throw new Error("--replay and --out are required.");
if (
  !Number.isInteger(options.fps) ||
  options.fps < 1 ||
  options.fps > 30 ||
  options.seconds <= 0 ||
  options.seconds > 60 ||
  options.start < 0 ||
  !Number.isInteger(options.width) ||
  !Number.isInteger(options.height) ||
  options.width % 2 ||
  options.height % 2
)
  throw new Error(
    "Use fps1–30, duration0–60s, nonnegative start and even integer dimensions.",
  );
const replayPath = path.resolve(options.replay),
  out = path.resolve(options.out);
await fs.mkdir(path.dirname(out), { recursive: true });
try {
  await fs.stat(out);
  throw new Error(`Refusing to overwrite ${out}`);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
const replayBytes = await fs.readFile(replayPath),
  replay = JSON.parse(replayBytes);
if (replay.provenance?.kind !== "dataset_inference")
  throw new Error(
    "Release capture requires an actual dataset_inference replay. Use screenshots for synthetic QA.",
  );
const duration = replay.frames.at(-1).time - replay.frames[0].time;
if (options.start + options.seconds > duration + 0.001)
  throw new Error("Selected capture interval exceeds actual replay duration.");
const sha = (buffer) => createHash("sha256").update(buffer).digest("hex");
const sourceHashes = {};
for (const name of ["index.html", "app.js", "replay.js", "style.css"])
  sourceHashes[`ghostmesh/web/${name}`] = sha(
    await fs.readFile(path.join(root, "ghostmesh/web", name)),
  );
const moduleName = options["playwright-module"] || "playwright";
const loaded = await import(
  path.isAbsolute(moduleName) ? pathToFileURL(moduleName).href : moduleName
);
const { chromium } = loaded;
const scratch = await fs.mkdtemp(path.join(tmpdir(), "ghostmesh-demo-"));
let server,
  browser,
  stdout = "",
  serverErrors = "",
  success = false;
function run(binary, argv) {
  return new Promise((resolve, reject) => {
    const process = spawn(binary, argv, {
      cwd: root,
      stdio: ["ignore", "ignore", "pipe"],
    });
    let tail = "";
    process.stderr.on("data", (chunk) => {
      tail = (tail + chunk).slice(-16000);
    });
    process.on("error", reject);
    process.on("exit", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`${binary} exited ${code}: ${tail}`)),
    );
  });
}
try {
  server = spawn(
    options.python,
    ["-u", "-m", "ghostmesh", "serve", "--replay", replayPath, "--port", "0"],
    { cwd: root, stdio: ["ignore", "pipe", "pipe"] },
  );
  server.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  server.stderr.on("data", (chunk) => {
    serverErrors = (serverErrors + chunk).slice(-16000);
  });
  const url = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      clearInterval(interval);
      reject(new Error(`Local viewer startup timed out: ${serverErrors}`));
    }, 30000);
    const interval = setInterval(() => {
      const match = stdout.match(/http:\/\/127\.0\.0\.1:\d+/);
      if (match) {
        clearInterval(interval);
        clearTimeout(timeout);
        resolve(match[0]);
      }
    }, 50);
    server.on("error", (error) => {
      clearInterval(interval);
      clearTimeout(timeout);
      reject(error);
    });
    server.on("exit", (code) => {
      clearInterval(interval);
      clearTimeout(timeout);
      reject(new Error(`Local viewer exited ${code}: ${serverErrors}`));
    });
  });
  browser = await chromium.launch({
    headless: true,
    ...(options.browser ? { executablePath: options.browser } : {}),
  });
  const page = await browser.newPage({
    viewport: { width: options.width, height: options.height },
    deviceScaleFactor: 1,
    reducedMotion: "reduce",
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(() => !document.getElementById("play").disabled);
  if (options.reference) await page.locator("#show-gt").check();
  await page.evaluate(() => document.activeElement.blur());
  // Move away from native controls and let their hover/check paint settle.
  // This changes no replay timestamp or model prediction.
  await page.mouse.move(0, 0);
  await page.waitForTimeout(300);
  const total = Math.round(options.fps * options.seconds),
    sourceTimes = [];
  for (let i = 0; i < total; i++) {
    const target = replay.frames[0].time + options.start + i / options.fps;
    let index = 0;
    while (
      index + 1 < replay.frames.length &&
      replay.frames[index + 1].time <= target + 1e-9
    )
      index++;
    sourceTimes.push(replay.frames[index].time);
    await page.evaluate((index) => {
      const seek = document.getElementById("seek");
      seek.value = String(index);
      seek.dispatchEvent(new Event("input"));
    }, index);
    await page.screenshot({
      path: path.join(scratch, `frame-${String(i).padStart(5, "0")}.png`),
      animations: "disabled",
    });
    if (i === 0 || (i + 1) % 20 === 0)
      console.log(`Captured ${i + 1}/${total} frames`);
  }
  // A backward seek to an already captured instant must reproduce exactly.
  const firstTime = replay.frames[0].time + options.start;
  const firstIndex = replay.frames.findLastIndex(
    (frame) => frame.time <= firstTime + 1e-9,
  );
  await page.evaluate((index) => {
    const seek = document.getElementById("seek");
    seek.value = String(index);
    seek.dispatchEvent(new Event("input"));
  }, firstIndex);
  const repeated = await page.screenshot({ animations: "disabled" });
  await fs.writeFile(path.join(scratch, "repeat-first.png"), repeated);
  const first = await fs.readFile(path.join(scratch, "frame-00000.png"));
  if (sha(repeated) !== sha(first))
    throw new Error("Backward seek did not reproduce identical pixels.");
  if (errors.length)
    throw new Error(`Viewer JavaScript errors: ${errors.join("; ")}`);
  await fs.mkdir(out);
  await fs.writeFile(path.join(out, "desktop.png"), first);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: path.join(out, "mobile.png"),
    fullPage: true,
    animations: "disabled",
  });
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw new Error("Mobile layout overflows.");
  await run(options.ffmpeg, [
    "-n",
    "-hide_banner",
    "-loglevel",
    "error",
    "-framerate",
    String(options.fps),
    "-i",
    path.join(scratch, "frame-%05d.png"),
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    path.join(out, "ghostmesh-demo.mp4"),
  ]);
  await run(options.ffmpeg, [
    "-n",
    "-hide_banner",
    "-loglevel",
    "error",
    "-framerate",
    String(options.fps),
    "-i",
    path.join(scratch, "frame-%05d.png"),
    "-vf",
    "fps=10,scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4",
    "-loop",
    "0",
    path.join(out, "ghostmesh-demo.gif"),
  ]);
  await run(options.ffmpeg, [
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    path.join(out, "ghostmesh-demo.mp4"),
    "-f",
    "null",
    "-",
  ]);
  const record = {
    schema_version: 1,
    scope:
      "Actual app screenshot sequence; timestamp-based seeking, no invented/interpolated pose motion. Silent recording.",
    replay: path.relative(root, replayPath),
    replay_sha256: sha(replayBytes),
    provenance: replay.provenance,
    source_sha256: sourceHashes,
    frames: total,
    fps: options.fps,
    duration_seconds: total / options.fps,
    start_seconds: options.start,
    source_timestamps: sourceTimes,
    reference_visible: Boolean(options.reference),
    viewport: [options.width, options.height],
    backward_seek_identical: true,
    browser_version: browser.version(),
    javascript_errors: errors,
  };
  record.assets = {};
  for (const filename of [
    "desktop.png",
    "mobile.png",
    "ghostmesh-demo.mp4",
    "ghostmesh-demo.gif",
  ])
    record.assets[filename] = sha(await fs.readFile(path.join(out, filename)));
  await fs.writeFile(
    path.join(out, "capture.json"),
    JSON.stringify(record, null, 2) + "\n",
  );
  success = true;
  console.log(`Captured ${total} actual replay frames to ${out}`);
} finally {
  if (browser) await browser.close();
  if (server) server.kill("SIGTERM");
  if (success) await fs.rm(scratch, { recursive: true });
  else console.error(`Incomplete capture; scratch retained at ${scratch}`);
}
