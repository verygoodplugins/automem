#!/usr/bin/env node
/**
 * Minimal MCP bridge that proxies JSON-RPC tool calls to the AutoMem HTTP API.
 *
 * Environment variables:
 *   MCP_MEMORY_HTTP_ENDPOINT (default: http://127.0.0.1:8001)
 *   MCP_MEMORY_API_KEY       (optional, sent as Bearer token)
 */

const { URL } = require('url');
const http = require('http');
const https = require('https');

class AutoMemBridge {
  constructor() {
    const endpoint = process.env.MCP_MEMORY_HTTP_ENDPOINT || 'http://127.0.0.1:8001';
    this.baseUrl = new URL(endpoint);
    if (!this.baseUrl.pathname.endsWith('/')) {
      this.baseUrl.pathname += '/';
    }
    this.apiKey = process.env.MCP_MEMORY_API_KEY || null;
  }

  async start() {
    process.stdin.setEncoding('utf8');
    let buffer = '';

    process.stdin.on('data', async chunk => {
      buffer += chunk;
      let index;
      while ((index = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (!line) continue;
        let request;
        try {
          request = JSON.parse(line);
        } catch (error) {
          this.write({
            jsonrpc: '2.0',
            id: null,
            error: { code: -32700, message: 'Parse error' },
          });
          continue;
        }
        try {
          const response = await this.handleRequest(request);
          if (response !== null) {
            this.write(response);
          }
        } catch (error) {
          this.write({
            jsonrpc: '2.0',
            id: request.id ?? null,
            error: { code: -32000, message: error.message },
          });
        }
      }
    });

    process.stdin.on('end', () => process.exit(0));
    process.on('SIGINT', () => process.exit(0));
    process.on('SIGTERM', () => process.exit(0));

    console.error(`AutoMem MCP bridge ready -> ${this.baseUrl.href}`);
  }

  write(payload) {
    process.stdout.write(`${JSON.stringify(payload)}\n`);
  }

  async handleRequest({ method, params = {}, id }) {
    switch (method) {
      case 'initialize':
        return {
          jsonrpc: '2.0',
          id,
          result: {
            protocolVersion: '2024-11-05',
            capabilities: { tools: { listChanged: false } },
            serverInfo: { name: 'automem-http-bridge', version: '0.1.0' },
          },
        };
      case 'notifications/initialized':
        return null;
      case 'tools/list':
        return {
          jsonrpc: '2.0',
          id,
          result: {
            tools: [
              {
                name: 'store_memory',
                description: 'Store a memory in AutoMem',
                inputSchema: {
                  type: 'object',
                  properties: {
                    content: { type: 'string' },
                    tags: { type: 'array', items: { type: 'string' } },
                    importance: { type: 'number', minimum: 0, maximum: 1 },
                    embedding: {
                      type: 'array',
                      items: { type: 'number' },
                      description: 'Optional embedding vector',
                    },
                  },
                  required: ['content'],
                },
              },
              {
                name: 'recall_memory',
                description: 'Recall memories from AutoMem',
                inputSchema: {
                  type: 'object',
                  properties: {
                    query: { type: 'string' },
                    embedding: { type: 'array', items: { type: 'number' } },
                    limit: { type: 'integer', minimum: 1, default: 5 },
                  },
                },
              },
              {
                name: 'check_database_health',
                description: 'Return current AutoMem health status',
                inputSchema: { type: 'object', properties: {} },
              },
            ],
          },
        };
      case 'tools/call':
        return {
          jsonrpc: '2.0',
          id,
          result: await this.callTool(params.name, params.arguments || {}),
        };
      default:
        throw new Error(`Unsupported method: ${method}`);
    }
  }

  async callTool(name, args) {
    switch (name) {
      case 'store_memory':
        return this.storeMemory(args);
      case 'recall_memory':
        return this.recallMemory(args);
      case 'check_database_health':
        return this.checkHealth();
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  async storeMemory(args) {
    const body = {
      content: args.content,
      tags: Array.isArray(args.tags) ? args.tags : [],
    };
    if (typeof args.importance === 'number') body.importance = args.importance;
    if (Array.isArray(args.embedding)) body.embedding = args.embedding;

    const response = await this.request('POST', 'memory', body);
    if (!response.ok) {
      throw new Error(response.error || 'Failed to store memory');
    }
    return {
      success: true,
      memory_id: response.data.memory_id,
      message: response.data.message,
    };
  }

  async recallMemory(args) {
    const query = new URL(this.baseUrl.href);
    query.pathname += 'recall';
    if (args.query) query.searchParams.set('query', args.query);
    if (args.limit) query.searchParams.set('limit', String(args.limit));
    if (Array.isArray(args.embedding)) {
      query.searchParams.set('embedding', args.embedding.join(','));
    }

    const response = await this.fetch(query, { method: 'GET' });
    if (!response.ok) {
      throw new Error(response.error || 'Failed to recall memories');
    }
    return response.data;
  }

  async checkHealth() {
    const response = await this.request('GET', 'health');
    if (!response.ok) {
      throw new Error(response.error || 'Health check failed');
    }
    return response.data;
  }

  async request(method, path, body) {
    const url = new URL(this.baseUrl.href);
    url.pathname += path.replace(/^\//, '');
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (this.apiKey) {
      options.headers.Authorization = `Bearer ${this.apiKey}`;
    }
    if (body !== undefined) {
      options.body = JSON.stringify(body);
    }
    return this.fetch(url, options);
  }

  async fetch(url, options = {}) {
    const method = options.method || 'GET';
    const headers = options.headers || {};
    const body = options.body ? Buffer.from(options.body) : null;

    const requestOptions = {
      method,
      headers,
      timeout: 10000,
    };

    return new Promise(resolve => {
    const transport = url.protocol === 'https:' ? https : http;
    if (url.protocol === 'https:' && process.env.MCP_MEMORY_REJECT_UNAUTHORIZED === 'false') {
      requestOptions.rejectUnauthorized = false;
    }
      const req = transport.request(url, requestOptions, res => {
        const chunks = [];
        res.on('data', chunk => chunks.push(chunk));
        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf8');
          let data;
          try {
            data = raw ? JSON.parse(raw) : {};
          } catch (error) {
            data = { raw };
          }
          const ok = res.statusCode >= 200 && res.statusCode < 300;
          const errorMessage = !ok
            ? data?.message || data?.detail || raw || `HTTP ${res.statusCode}`
            : undefined;
          resolve({ ok, status: res.statusCode, data, error: errorMessage });
        });
      });

      req.on('error', error => {
        resolve({ ok: false, status: 0, error: error.message });
      });

      if (body) {
        req.write(body);
      }
      req.end();
    });
  }
}

if (require.main === module) {
  const bridge = new AutoMemBridge();
  bridge.start().catch(error => {
    console.error(`Failed to start AutoMem bridge: ${error.message}`);
    process.exit(1);
  });
}

module.exports = AutoMemBridge;
