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
  // Use GPT-5.1 for best reasoning
  // GPT-5.1: $1.25/1M input, $10/1M output (best reasoning)
  // GPT-4.1: ~$2/1M input, ~$8/1M output (good balance)
  // gpt-4o-mini: cheapest but weaker reasoning
  model: process.env.EVAL_MODEL || "gpt-5.1",
  evalModel: process.env.EVAL_JUDGE_MODEL || "gpt-5.1",
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
 * Evaluate answer using LLM (matching CORE's EXACT evaluateService.js)
 *
 * CORE's key insight: "you should be generous with your grading -
 * as long as it touches on the same topic as the gold answer,
 * it should be counted as CORRECT"
 */
async function evaluateAnswer(question, expectedAnswer, generatedAnswer, category) {
  // Category 5: Adversarial - check for "no information" phrase
  if (category === 5) {
    const genLower = generatedAnswer.toLowerCase();
    const isCorrect =
      genLower.includes("no information available") ||
      genLower.includes("not mentioned") ||
      genLower.includes("no information") ||
      genLower.includes("cannot find") ||
      genLower.includes("not found") ||
      genLower.includes("don't have") ||
      genLower.includes("do not have");
    return {
      isCorrect,
      confidence: isCorrect ? 1.0 : 0.0,
      reasoning: isCorrect
        ? "Correctly identified no information available"
        : "Failed to identify that information is not available",
    };
  }

  // CORE's EXACT evaluation prompt (from their evaluateService.js)
  const prompt = `Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.

You will be given the following data:
(1) a question (posed by one user to another user)
(2) a 'gold' (ground truth) answer
(3) a generated answer which you will score as CORRECT/WRONG

The point of the question is to ask about something one user should know about the other user based on their prior conversations.

The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: ${question}
Gold answer: ${expectedAnswer}
Generated answer: ${generatedAnswer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Return your response in JSON format: {"label": "CORRECT" or "WRONG", "reasoning": "your explanation"}`;

  try {
    const response = await openai.chat.completions.create({
      model: CONFIG.evalModel,
      messages: [
        {
          role: "system",
          content:
            "You are an evaluator for a question-answering benchmark. Be generous in your grading.",
        },
        { role: "user", content: prompt },
      ],
      temperature: 0.0,
      max_tokens: 200,
      response_format: { type: "json_object" },
    });

    const result = JSON.parse(response.choices[0].message.content);

    // Handle CORE's format (label: "CORRECT"/"WRONG")
    const isCorrect = result.label === "CORRECT" || result.isCorrect === true;
    return {
      isCorrect,
      confidence: isCorrect ? 1.0 : 0.0,
      reasoning: result.reasoning || "",
    };
  } catch (error) {
    console.error("Evaluation error:", error.message);

    // CORE's fallback: 30% word match = CORRECT
    const generatedLower = generatedAnswer.toLowerCase();
    const expectedLower = expectedAnswer.toString().toLowerCase();
    const expectedWords = expectedLower
      .split(/\s+/)
      .filter((word) => word.length > 2);
    const matchingWords = expectedWords.filter((word) =>
      generatedLower.includes(word)
    );
    const matchRatio =
      expectedWords.length > 0 ? matchingWords.length / expectedWords.length : 0;
    const isCorrect = matchRatio > 0.3;

    return {
      isCorrect,
      confidence: matchRatio,
      reasoning: `Fallback: ${matchRatio.toFixed(2)} word match ratio`,
    };
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
runEvaluation().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

