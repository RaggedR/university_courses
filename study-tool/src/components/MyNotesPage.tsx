import { useState, useEffect, useCallback } from 'react';
import {
  getNotes,
  getNote,
  updateNote,
  deleteNote,
  exportAsMdx,
  type SavedNote,
  type FlashcardData,
  type QuizQuestionData,
} from './note-store';
import MarkdownPreview from './MarkdownPreview';
import Flashcard from './Flashcard';
import Quiz from './Quiz';
import CardCreator from './CardCreator';

export default function MyNotesPage({ basePath }: { basePath: string }) {
  const [notes, setNotes] = useState<SavedNote[]>([]);
  const [viewId, setViewId] = useState<string | null>(null);
  const [showCreator, setShowCreator] = useState(false);
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);

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

  function handleUpdateFlashcards(cards: FlashcardData[]) {
    if (!viewId) return;
    updateNote(viewId, { flashcards: cards });
    setNotes(getNotes());
  }

  function handleUpdateQuiz(questions: QuizQuestionData[]) {
    if (!viewId) return;
    updateNote(viewId, { quizQuestions: questions });
    setNotes(getNotes());
  }

  // ── Note viewer ──
  if (currentNote) {
    return (
      <>
        <div className="note-header">
          <div>
            <button className="btn btn-secondary btn-sm" onClick={() => { setViewId(null); setShowCreator(false); }}>
              &larr; Back to Notes
            </button>
            <h1 style={{ marginTop: '0.5rem' }}>{currentNote.title}</h1>
          </div>
          <div className="note-header-actions">
            <button
              className={`btn btn-sm ${showCreator ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setShowCreator(!showCreator)}
            >
              {showCreator ? 'Hide Creator' : 'Add Flashcards & Quiz'}
            </button>
            <button className="btn btn-secondary btn-sm" onClick={(e) => handleExport(currentNote, e)}>
              Export .mdx
            </button>
          </div>
        </div>

        <div className="note-content">
          <MarkdownPreview content={currentNote.markdown} />
        </div>

        {showCreator && (
          <CardCreator
            flashcards={currentNote.flashcards}
            quizQuestions={currentNote.quizQuestions}
            noteContent={currentNote.markdown}
            onUpdateFlashcards={handleUpdateFlashcards}
            onUpdateQuiz={handleUpdateQuiz}
          />
        )}

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
