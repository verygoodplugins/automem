/**
 * CORE-Compatible Evaluation for AutoMem
 *
 * This script runs the same evaluation methodology as CORE's benchmark,
 * allowing direct comparison of results.
 *
 * Methodology (matching CORE):
 * 1. Load LoCoMo conversations
 * 2. For each question:
 *    a. Search AutoMem for relevant context
 *    b. Generate answer using LLM + retrieved context
 *    c. Score answer using LLM-based evaluation
 * 3. Compute accuracy per category
 *
 * Usage:
 *   OPENAI_API_KEY=xxx AUTOMEM_API_TOKEN=xxx node evaluate.js
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
import { AutoMemSearchService } from "./automem-search.js";
import "dotenv/config";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configuration
const CONFIG = {
  dataFile: path.join(__dirname, "../locomo/data/locomo10.json"),
  outputFile: path.join(__dirname, "evaluation_results.json"),
  searchLimit: 20,
  model: process.env.EVAL_MODEL || "gpt-4o-mini",
};

// Initialize services
const searchService = new AutoMemSearchService();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Category names (matching CORE)
const CATEGORY_NAMES = {
  1: "Single Hop",
  2: "Temporal",
  3: "Multi Hop",
  4: "Open Domain",
  5: "Adversarial",
};

/**
 * Generate answer from retrieved context using LLM
 */
async function generateAnswer(question, context, category) {
  const systemPrompt =
    category === 5
      ? `You are answering questions based ONLY on the provided conversation context.
If the information needed to answer is NOT in the context, respond with EXACTLY:
"no information available"
Do NOT make up information.`
      : `You are answering questions based on the provided conversation context.
Answer concisely and directly. Use only information from the context.
If the information is not available, say "no information available".`;

  try {
    const response = await openai.chat.completions.create({
      model: CONFIG.model,
      messages: [
        { role: "system", content: systemPrompt },
        {
          role: "user",
          content: `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`,
        },
      ],
      temperature: 0.0,
      max_tokens: 200,
    });

    return response.choices[0].message.content.trim();
  } catch (error) {
    console.error("LLM generation error:", error.message);
    return "error generating answer";
  }
}

/**
 * Evaluate answer using LLM (matching CORE's evaluateService.js)
 */
async function evaluateAnswer(question, expectedAnswer, generatedAnswer, category) {
  // Category 5: Adversarial - check for "no information" phrase
  if (category === 5) {
    const genLower = generatedAnswer.toLowerCase();
    const isCorrect =
      genLower.includes("no information available") ||
      genLower.includes("not mentioned") ||
      genLower.includes("no information");
    return {
      isCorrect,
      confidence: isCorrect ? 1.0 : 0.0,
      reasoning: isCorrect
        ? "Correctly identified no information available"
        : "Failed to identify that information is not available",
    };
  }

  // For other categories, use LLM-based evaluation
  const prompt = `You are evaluating whether a generated answer matches the expected answer.

Question: ${question}
Expected Answer: ${expectedAnswer}
Generated Answer: ${generatedAnswer}

Consider:
- Semantic equivalence (same meaning, different words)
- Partial matches (answer contains key information)
- Date/time format variations

Respond in JSON format:
{
  "isCorrect": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}`;

  try {
    const response = await openai.chat.completions.create({
      model: CONFIG.model,
      messages: [
        { role: "system", content: "You are a precise answer evaluator." },
        { role: "user", content: prompt },
      ],
      temperature: 0.0,
      max_tokens: 150,
      response_format: { type: "json_object" },
    });

    return JSON.parse(response.choices[0].message.content);
  } catch (error) {
    console.error("Evaluation error:", error.message);
    return { isCorrect: false, confidence: 0.0, reasoning: "Evaluation failed" };
  }
}

/**
 * Main evaluation function
 */
async function runEvaluation() {
  console.log("\n" + "=".repeat(60));
  console.log("🧠 AutoMem CORE-Compatible Evaluation");
  console.log("=".repeat(60));

  // Health check
  console.log("\n🏥 Checking AutoMem health...");
  const healthy = await searchService.healthCheck();
  if (!healthy) {
    throw new Error("AutoMem API is not accessible");
  }
  console.log("✅ AutoMem is healthy");

  // Load data
  console.log(`\n📂 Loading LoCoMo dataset...`);
  const conversations = JSON.parse(fs.readFileSync(CONFIG.dataFile, "utf-8"));
  console.log(`✅ Loaded ${conversations.length} conversations`);

  // Results tracking
  const results = {
    overall: { correct: 0, total: 0 },
    byCategory: {},
    details: [],
  };

  for (let cat = 1; cat <= 5; cat++) {
    results.byCategory[cat] = { correct: 0, total: 0 };
  }

  const startTime = Date.now();

  // Evaluate each conversation
  for (const conv of conversations) {
    const sampleId = conv.sample_id;
    console.log(`\n📝 Evaluating ${sampleId}...`);

    const questions = conv.qa || [];
    let convCorrect = 0;

    for (let i = 0; i < questions.length; i++) {
      const qa = questions[i];
      const question = qa.question;
      const expectedAnswer = qa.answer || "";
      const category = qa.category;

      // Search for context
      const searchResults = await searchService.search(question, sampleId, {
        limit: CONFIG.searchLimit,
      });
      const context = searchResults.episodes.slice(0, 10).join("\n\n");

      // Generate answer
      const generatedAnswer = await generateAnswer(question, context, category);

      // Evaluate
      const evaluation = await evaluateAnswer(
        question,
        expectedAnswer,
        generatedAnswer,
        category
      );

      // Track results
      const isCorrect = evaluation.isCorrect;
      results.overall.total++;
      results.byCategory[category].total++;

      if (isCorrect) {
        results.overall.correct++;
        results.byCategory[category].correct++;
        convCorrect++;
      }

      results.details.push({
        sampleId,
        question,
        expectedAnswer,
        generatedAnswer,
        category,
        isCorrect,
        confidence: evaluation.confidence,
        reasoning: evaluation.reasoning,
        contextLength: context.length,
      });

      // Progress
      if ((i + 1) % 20 === 0) {
        process.stdout.write(`  Processed ${i + 1}/${questions.length}...\r`);
      }
    }

    const convAcc = (convCorrect / questions.length) * 100;
    console.log(`  Accuracy: ${convAcc.toFixed(1)}% (${convCorrect}/${questions.length})`);
  }

  const elapsedTime = (Date.now() - startTime) / 1000;

  // Print results
  console.log("\n" + "=".repeat(60));
  console.log("📊 FINAL RESULTS (CORE-Compatible Methodology)");
  console.log("=".repeat(60));

  const overallAcc = (results.overall.correct / results.overall.total) * 100;
  console.log(`\n🎯 Overall Accuracy: ${overallAcc.toFixed(2)}%`);
  console.log(`   (${results.overall.correct}/${results.overall.total})`);
  console.log(`⏱️  Total Time: ${elapsedTime.toFixed(1)}s`);

  console.log("\n📈 Category Breakdown:");
  for (const [cat, data] of Object.entries(results.byCategory)) {
    if (data.total > 0) {
      const acc = (data.correct / data.total) * 100;
      console.log(
        `  ${CATEGORY_NAMES[cat] || `Cat ${cat}`}: ${acc.toFixed(1)}% (${data.correct}/${data.total})`
      );
    }
  }

  // Comparison with CORE's claimed scores
  console.log("\n🏆 Comparison with CORE (claimed):");
  console.log("  CORE Single Hop: 91%");
  console.log("  CORE Multi Hop: 85%");
  console.log("  CORE Temporal: 88%");
  console.log("  CORE Open Domain: 71%");
  console.log("  CORE Overall: ~85%");
  console.log(`  AutoMem: ${overallAcc.toFixed(2)}%`);

  // Save results
  fs.writeFileSync(CONFIG.outputFile, JSON.stringify(results, null, 2));
  console.log(`\n💾 Results saved to: ${CONFIG.outputFile}`);

  return results;
}

// Run if called directly
runEvaluation().catch(console.error);

