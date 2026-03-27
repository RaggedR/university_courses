export interface FlashcardData {
  front: string;
  back: string;
}

export interface QuizQuestionData {
  question: string;
  options: string[];
  answer: number;
  explanation?: string;
}

export interface SavedNote {
  id: string;
  title: string;
  markdown: string;
  flashcards: FlashcardData[];
  quizQuestions: QuizQuestionData[];
  createdAt: string;
  updatedAt: string;
}

const STORAGE_KEY = 'uni-notes-saved';

function loadAll(): SavedNote[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveAll(notes: SavedNote[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
}

export function getNotes(): SavedNote[] {
  return loadAll();
}

export function getNote(id: string): SavedNote | undefined {
  return loadAll().find(n => n.id === id);
}

export function createNote(title: string, markdown: string): SavedNote {
  const notes = loadAll();
  const note: SavedNote = {
    id: crypto.randomUUID(),
    title,
    markdown,
    flashcards: [],
    quizQuestions: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  notes.push(note);
  saveAll(notes);
  return note;
}

export function updateNote(id: string, updates: Partial<Pick<SavedNote, 'title' | 'markdown' | 'flashcards' | 'quizQuestions'>>): SavedNote | undefined {
  const notes = loadAll();
  const idx = notes.findIndex(n => n.id === id);
  if (idx === -1) return undefined;
  notes[idx] = { ...notes[idx], ...updates, updatedAt: new Date().toISOString() };
  saveAll(notes);
  return notes[idx];
}

export function deleteNote(id: string) {
  const notes = loadAll().filter(n => n.id !== id);
  saveAll(notes);
}

export function exportAsMdx(note: SavedNote): string {
  let mdx = `---\ntitle: "${note.title}"\n---\n\n`;

  const hasFlashcards = note.flashcards.length > 0;
  const hasQuiz = note.quizQuestions.length > 0;

  if (hasFlashcards || hasQuiz) {
    if (hasFlashcards) mdx += `import Flashcard from '../../../components/Flashcard';\n`;
    if (hasQuiz) mdx += `import Quiz from '../../../components/Quiz';\n`;
    mdx += '\n';
  }

  mdx += note.markdown + '\n';

  if (hasFlashcards) {
    mdx += `\n## Flashcards\n\n`;
    mdx += `<Flashcard client:load cards={${JSON.stringify(note.flashcards, null, 2)}} />\n`;
  }

  if (hasQuiz) {
    mdx += `\n## Quiz\n\n`;
    mdx += `<Quiz client:load questions={${JSON.stringify(note.quizQuestions, null, 2)}} />\n`;
  }

  return mdx;
}
