import http from 'node:http'
import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import path from 'node:path'
import url from 'node:url'

const HOST = process.env.HOST || '0.0.0.0'
const PORT = Number(process.env.PORT || 3000)
const rootDir = path.resolve(process.cwd(), 'dist')

function contentTypeFor(filePath) {
  switch (path.extname(filePath).toLowerCase()) {
    case '.html':
      return 'text/html; charset=utf-8'
    case '.js':
      return 'text/javascript; charset=utf-8'
    case '.css':
      return 'text/css; charset=utf-8'
    case '.json':
      return 'application/json; charset=utf-8'
    case '.svg':
      return 'image/svg+xml'
    case '.png':
      return 'image/png'
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.webp':
      return 'image/webp'
    case '.ico':
      return 'image/x-icon'
    case '.txt':
      return 'text/plain; charset=utf-8'
    default:
      return 'application/octet-stream'
  }
}

function send(res, statusCode, headers) {
  res.writeHead(statusCode, headers)
  res.end()
}

async function tryServeFile(req, res, filePath, { cache } = { cache: 'none' }) {
  try {
    const fileStat = await stat(filePath)
    if (!fileStat.isFile()) return false

    const headers = {
      'Content-Type': contentTypeFor(filePath),
      'Content-Length': String(fileStat.size),
      'Cache-Control':
        cache === 'immutable' ? 'public, max-age=31536000, immutable' : 'no-cache',
    }

    if (req.method === 'HEAD') {
      send(res, 200, headers)
      return true
    }

    res.writeHead(200, headers)
    createReadStream(filePath).pipe(res)
    return true
  } catch {
    return false
  }
}

const server = http.createServer(async (req, res) => {
  if (!req.url) return send(res, 400, { 'Content-Type': 'text/plain; charset=utf-8' })
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    return send(res, 405, { 'Content-Type': 'text/plain; charset=utf-8' })
  }

  const parsed = url.parse(req.url)
  const pathname = decodeURIComponent(parsed.pathname || '/')

  const requestedPath = pathname.replace(/^\/+/, '')
  const candidate = path.resolve(rootDir, requestedPath || 'index.html')
  if (!candidate.startsWith(rootDir + path.sep) && candidate !== rootDir) {
    return send(res, 400, { 'Content-Type': 'text/plain; charset=utf-8' })
  }

  const cache = pathname.startsWith('/assets/') ? 'immutable' : 'none'
  if (await tryServeFile(req, res, candidate, { cache })) return

  // SPA fallback: serve index.html for unknown routes (but keep 404 for missing assets)
  if (!pathname.startsWith('/assets/')) {
    const indexPath = path.join(rootDir, 'index.html')
    if (await tryServeFile(req, res, indexPath, { cache: 'none' })) return
  }

  send(res, 404, { 'Content-Type': 'text/plain; charset=utf-8' })
})

server.listen(PORT, HOST, () => {
  console.log(`graph-viewer: serving ${rootDir} on http://${HOST}:${PORT}`)
})
