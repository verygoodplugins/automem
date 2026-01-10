#!/usr/bin/env node
/**
 * Dev launcher: start BOTH
 * - iPhone hand-tracking bridge (ws phone + ws/http web)
 * - Graph viewer Vite dev server
 *
 * Goal: remove the “forgot to start the bridge” failure mode.
 */

import { execSync, spawn } from 'node:child_process'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const viewerRoot = join(__dirname, '..')

// Match defaults in hand-tracking-server.js
const PHONE_PORT = Number(process.env.HAND_TRACKING_PHONE_PORT || 8768)
const WEB_PORT = Number(process.env.HAND_TRACKING_WEB_PORT || 8766)

function safeKillPorts() {
  if (process.platform !== 'darwin' && process.platform !== 'linux') return
  try {
    execSync(
      [
        `lsof -ti:${PHONE_PORT} | xargs kill -9 2>/dev/null || true`,
        `lsof -ti:${WEB_PORT} | xargs kill -9 2>/dev/null || true`,
      ].join('; '),
      { stdio: 'ignore', shell: '/bin/bash' }
    )
  } catch {
    // ignore
  }
}

function run(cmd, args, opts) {
  const child = spawn(cmd, args, {
    stdio: 'inherit',
    ...opts,
  })
  child.on('exit', (code) => {
    if (code && code !== 0) {
      process.exitCode = code
    }
  })
  return child
}

console.log('\n🧠 graph-viewer dev:all\n')
console.log(`Using ports: phone=${PHONE_PORT} web=${WEB_PORT}`)
console.log('Clearing any old listeners on those ports...')
safeKillPorts()

console.log('\nStarting iPhone hand-tracking bridge...')
const bridge = run(process.execPath, ['hand-tracking-server.js'], {
  cwd: __dirname,
  env: {
    ...process.env,
    HAND_TRACKING_PHONE_PORT: String(PHONE_PORT),
    HAND_TRACKING_WEB_PORT: String(WEB_PORT),
  },
})

console.log('\nStarting Vite dev server...')
const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm'
const vite = run(npmCmd, ['run', 'dev'], { cwd: viewerRoot, env: process.env })

const shutdown = (signal) => {
  try { bridge.kill(signal) } catch {}
  try { vite.kill(signal) } catch {}
  process.exit(0)
}

process.on('SIGINT', () => shutdown('SIGINT'))
process.on('SIGTERM', () => shutdown('SIGTERM'))
