/**
 * AutoMem Search Service Adapter for CORE Benchmark
 *
 * This module provides a search interface compatible with CORE's benchmark
 * evaluation code, allowing direct comparison against AutoMem.
 *
 * Usage:
 *   import { AutoMemSearchService } from './automem-search.js';
 *   const search = new AutoMemSearchService();
 *   const results = await search.search(query, userId, { limit: 20 });
 */

import axios from "axios";

export class AutoMemSearchService {
  constructor(config = {}) {
    this.baseUrl = config.baseUrl || process.env.AUTOMEM_BASE_URL || "http://localhost:8001";
    this.apiToken = config.apiToken || process.env.AUTOMEM_API_TOKEN || "test-token";

    this.axios = axios.create({
      baseURL: this.baseUrl,
      headers: {
        Authorization: `Bearer ${this.apiToken}`,
        "Content-Type": "application/json",
      },
      timeout: 30000,
    });
  }

  /**
   * Search AutoMem for relevant memories.
   *
   * This method mirrors CORE's search.server.js interface so their
   * evaluation code can work with AutoMem.
   *
   * @param {string} query - Search query
   * @param {string} userId - User/conversation ID (used as tag prefix)
   * @param {object} options - Search options
   * @param {number} options.limit - Max results (default: 20)
   * @param {number} options.scoreThreshold - Min score (not used, kept for compat)
   * @returns {object} - { episodes: string[], facts: object[] }
   */
  async search(query, userId, options = {}) {
    const limit = options.limit || 20;

    try {
      const response = await this.axios.get("/recall", {
        params: {
          query: query,
          tags: `conversation:${userId}`,
          tag_match: "exact",
          limit: limit,
          // Enable multi-hop features
          auto_decompose: "true",
          expand_entities: "true",
        },
      });

      const results = response.data.results || [];

      // Convert AutoMem format to CORE format
      const episodes = results.map((r) => {
        const memory = r.memory || {};
        const content = memory.content || "";
        const metadata = memory.metadata || {};
        const sessionDatetime = metadata.session_datetime || "";

        // Include session datetime for temporal context
        if (sessionDatetime) {
          return `[${sessionDatetime}] ${content}`;
        }
        return content;
      });

      // Extract facts from memories that have entity tags
      const facts = results
        .filter((r) => {
          const tags = r.memory?.tags || [];
          return tags.some((t) => t.startsWith("entity:"));
        })
        .map((r) => ({
          fact: r.memory.content,
          validAt: r.memory.metadata?.session_datetime || null,
          source: r.memory.id,
        }));

      return {
        episodes,
        facts,
        // Include raw results for debugging
        _raw: results,
      };
    } catch (error) {
      console.error("AutoMem search error:", error.message);
      return { episodes: [], facts: [] };
    }
  }

  /**
   * Health check
   */
  async healthCheck() {
    try {
      const response = await this.axios.get("/health");
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }
}

export default AutoMemSearchService;

