import { useState, useEffect, useCallback, useRef } from 'react';
import {
  getNotes,
  getNote,
  updateNote,
  deleteNote,
  exportAsMdx,
  type SavedNote,
} from './note-store';
import { getApiKey, setApiKey, generateCards } from './ai-generate';
import MarkdownPreview from './MarkdownPreview';
import Flashcard from './Flashcard';
import Quiz from './Quiz';

export default function MyNotesPage({ basePath }: { basePath: string }) {
  const [notes, setNotes] = useState<SavedNote[]>([]);
  const [viewId, setViewId] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [keyInput, setKeyInput] = useState('');
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);
  const flashcardsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setNotes(getNotes());
  }, []);

  const showToast = useCallback((msg: string, type: 'success' | 'error' = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  }, []);

  const currentNote = viewId ? getNote(viewId) : null;

  function handleDelete(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm('Delete this note?')) return;
    deleteNote(id);
    setNotes(getNotes());
    if (viewId === id) setViewId(null);
    showToast('Note deleted');
  }

  function handleExport(note: SavedNote, e: React.MouseEvent) {
    e.stopPropagation();
    const mdx = exportAsMdx(note);
    const blob = new Blob([mdx], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${note.title.toLowerCase().replace(/\s+/g, '-')}.mdx`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Exported .mdx file');
  }

  async function handleGenerate() {
    if (!viewId) return;
    const key = getApiKey();
    if (!key) {
      setShowKeyInput(true);
      setKeyInput('');
      return;
    }
    const note = getNote(viewId);
    if (!note) return;

    setGenerating(true);
    try {
      const result = await generateCards(note.markdown, 'both');
      if (result.flashcards.length > 0) {
        updateNote(viewId, { flashcards: [...note.flashcards, ...result.flashcards] });
      }
      if (result.quizQuestions.length > 0) {
        updateNote(viewId, { quizQuestions: [...note.quizQuestions, ...result.quizQuestions] });
      }
      setNotes(getNotes());
      if (result.flashcards.length === 0 && result.quizQuestions.length === 0) {
        showToast('AI returned empty results', 'error');
      } else {
        showToast(`Generated ${result.flashcards.length} cards + ${result.quizQuestions.length} questions`);
        setTimeout(() => flashcardsRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
      }
    } catch (err: any) {
      if (err.message === 'Invalid API key') {
        setShowKeyInput(true);
        setKeyInput('');
      }
      showToast(err.message || 'Generation failed', 'error');
    } finally {
      setGenerating(false);
    }
  }

  function saveKey() {
    setApiKey(keyInput.trim());
    setShowKeyInput(false);
    handleGenerate();
  }

  // ── Note viewer ──
  if (currentNote) {
    return (
      <>
        <div className="note-header">
          <div>
            <button className="btn btn-secondary btn-sm" onClick={() => setViewId(null)}>
              &larr; Back to Notes
            </button>
            <h1 style={{ marginTop: '0.5rem' }}>{currentNote.title}</h1>
          </div>
          <div className="note-header-actions">
            <button
              className="btn btn-primary btn-sm"
              onClick={handleGenerate}
              disabled={generating}
            >
              {generating ? 'Generating (~60s)...' : 'Generate Flashcards & Quiz'}
            </button>
            <button className="btn btn-secondary btn-sm" onClick={(e) => handleExport(currentNote, e)}>
              Export .mdx
            </button>
          </div>
        </div>

        {showKeyInput && (
          <div className="creator-section" style={{ marginBottom: '1rem' }}>
            <label style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--app-text-muted)' }}>
              Anthropic API Key
            </label>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.4rem' }}>
              <input
                type="password"
                className="title-input"
                placeholder="sk-ant-..."
                value={keyInput}
                onChange={e => setKeyInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && keyInput.trim()) saveKey(); }}
                style={{ marginBottom: 0, flex: 1 }}
              />
              <button className="btn btn-primary btn-sm" onClick={saveKey} disabled={!keyInput.trim()}>
                Save & Generate
              </button>
            </div>
            <span style={{ fontSize: '0.8rem', color: 'var(--app-text-muted)', marginTop: '0.3rem', display: 'block' }}>
              Stored locally in your browser. Only sent to Anthropic's API.
            </span>
          </div>
        )}

        <div className="note-content">
          <MarkdownPreview content={currentNote.markdown} />
        </div>

        <div ref={flashcardsRef}>
          {currentNote.flashcards.length > 0 && (
            <div style={{ marginTop: '2rem' }}>
              <h2>Flashcards</h2>
              <Flashcard cards={currentNote.flashcards} />
            </div>
          )}

          {currentNote.quizQuestions.length > 0 && (
            <div style={{ marginTop: '2rem' }}>
              <h2>Quiz</h2>
              <Quiz questions={currentNote.quizQuestions} />
            </div>
          )}
        </div>

        {toast && <div className={`toast toast-${toast.type}`}>{toast.msg}</div>}
      </>
    );
  }

  // ── Notes list ──
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h1>My Notes</h1>
        <a className="btn btn-primary" href={`${basePath}/upload/`}>
          + Upload Notes
        </a>
      </div>

      {notes.length === 0 ? (
        <div className="notes-empty">
          <p>No notes yet.</p>
          <a className="btn btn-primary" href={`${basePath}/upload/`} style={{ marginTop: '1rem', display: 'inline-flex' }}>
            Upload Your First Note
          </a>
        </div>
      ) : (
        <div className="notes-grid">
          {notes.map(note => (
            <div key={note.id} className="note-card" onClick={() => setViewId(note.id)}>
              <div className="note-card-info">
                <h3>{note.title}</h3>
                <div className="note-card-meta">
                  <span>{new Date(note.updatedAt).toLocaleDateString()}</span>
                  {note.flashcards.length > 0 && <span>{note.flashcards.length} cards</span>}
                  {note.quizQuestions.length > 0 && <span>{note.quizQuestions.length} questions</span>}
                </div>
              </div>
              <div className="note-card-actions">
                <button className="btn btn-secondary btn-sm" onClick={(e) => handleExport(note, e)}>
                  Export
                </button>
                <button className="btn btn-danger btn-sm" onClick={(e) => handleDelete(note.id, e)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {toast && <div className={`toast toast-${toast.type}`}>{toast.msg}</div>}
    </>
  );
}
