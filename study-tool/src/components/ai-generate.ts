import type { FlashcardData, QuizQuestionData } from './note-store';

const API_KEY_STORAGE = 'uni-notes-anthropic-key';

export function getApiKey(): string {
  return localStorage.getItem(API_KEY_STORAGE) ?? '';
}

export function setApiKey(key: string) {
  localStorage.setItem(API_KEY_STORAGE, key);
}

interface GenerateResult {
  flashcards: FlashcardData[];
  quizQuestions: QuizQuestionData[];
}

export async function generateCards(
  noteContent: string,
  mode: 'flashcards' | 'quiz' | 'both',
): Promise<GenerateResult> {
  const apiKey = getApiKey();
  if (!apiKey) {
    throw new Error('API key not set');
  }

  const prompt = buildPrompt(noteContent, mode);

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      messages: [{ role: 'user', content: prompt }],
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    if (response.status === 401) throw new Error('Invalid API key');
    throw new Error(`API error: ${response.status} — ${err}`);
  }

  const data = await response.json();
  const text = data.content?.[0]?.text ?? '';

  return parseResponse(text, mode);
}

function buildPrompt(content: string, mode: 'flashcards' | 'quiz' | 'both'): string {
  const parts: string[] = [];

  parts.push(`Analyze the following study notes and generate study materials.\n`);

  if (mode === 'flashcards' || mode === 'both') {
    parts.push(`Generate 5-10 flashcards covering the key concepts. Each flashcard should have a concise question/term on the front and a clear answer/definition on the back. Use LaTeX notation (with $ delimiters) for any math.`);
  }

  if (mode === 'quiz' || mode === 'both') {
    parts.push(`Generate 4-6 multiple choice quiz questions testing understanding (not just recall). Each question should have 4 options with exactly one correct answer. Include a brief explanation for each. Use LaTeX notation (with $ delimiters) for any math.`);
  }

  parts.push(`\nRespond with ONLY valid JSON in this exact format (no markdown code fences):`);

  parts.push(`{
  "flashcards": [
    { "front": "question or term", "back": "answer or definition" }
  ],
  "quizQuestions": [
    { "question": "question text", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": "why this is correct" }
  ]
}`);

  if (mode === 'flashcards') {
    parts.push(`\nOnly include the "flashcards" array (set "quizQuestions" to []).`);
  } else if (mode === 'quiz') {
    parts.push(`\nOnly include the "quizQuestions" array (set "flashcards" to []).`);
  }

  parts.push(`\n---\nNOTES:\n${content}`);

  return parts.join('\n');
}

function parseResponse(text: string, _mode: string): GenerateResult {
  // Strip markdown code fences if present
  let cleaned = text.trim();
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '');
  }

  try {
    const parsed = JSON.parse(cleaned);
    return {
      flashcards: Array.isArray(parsed.flashcards)
        ? parsed.flashcards.filter((c: any) => c.front && c.back)
        : [],
      quizQuestions: Array.isArray(parsed.quizQuestions)
        ? parsed.quizQuestions.filter(
            (q: any) => q.question && Array.isArray(q.options) && typeof q.answer === 'number',
          )
        : [],
    };
  } catch {
    throw new Error('Failed to parse AI response. Try again.');
  }
}
