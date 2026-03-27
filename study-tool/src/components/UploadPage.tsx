import { useState, useCallback } from 'react';
import { createNote, type FlashcardData, type QuizQuestionData } from './note-store';
import { getApiKey, setApiKey, generateCards } from './ai-generate';
import MarkdownPreview from './MarkdownPreview';
import Flashcard from './Flashcard';
import Quiz from './Quiz';

export default function UploadPage({ basePath }: { basePath: string }) {
  const [title, setTitle] = useState('');
  const [markdown, setMarkdown] = useState('');
  const [url, setUrl] = useState('');
  const [fetching, setFetching] = useState(false);
  const [editorOpen, setEditorOpen] = useState(true);
  const [flashcards, setFlashcards] = useState<FlashcardData[]>([]);
  const [quizQuestions, setQuizQuestions] = useState<QuizQuestionData[]>([]);
  const [generating, setGenerating] = useState(false);
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [keyInput, setKeyInput] = useState('');
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);

  const showToast = useCallback((msg: string, type: 'success' | 'error' = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  }, []);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      setMarkdown(text);
      if (!title) {
        const name = file.name.replace(/\.(md|mdx|txt)$/, '').replace(/[-_]/g, ' ');
        setTitle(name.charAt(0).toUpperCase() + name.slice(1));
      }
    };
    reader.readAsText(file);
  }

  function toRawGitHubUrl(u: string): string {
    const m = u.match(/^https?:\/\/github\.com\/([^/]+\/[^/]+)\/blob\/(.+)$/);
    return m ? `https://raw.githubusercontent.com/${m[1]}/${m[2]}` : u;
  }

  async function handleFetchUrl() {
    const trimmed = url.trim();
    if (!trimmed) return;
    setFetching(true);
    try {
      const fetchUrl = toRawGitHubUrl(trimmed);
      const res = await fetch(fetchUrl);
      if (!res.ok) {
        showToast(`Fetch failed: HTTP ${res.status}`, 'error');
        return;
      }
      const text = await res.text();
      setMarkdown(text);
      if (!title) {
        const segments = trimmed.split('/');
        const filename = segments[segments.length - 1] || segments[segments.length - 2] || '';
        const name = filename.replace(/\.(md|mdx|txt)$/, '').replace(/[-_]/g, ' ');
        if (name) setTitle(name.charAt(0).toUpperCase() + name.slice(1));
      }
    } catch (err) {
      if (err instanceof TypeError) {
        showToast('CORS blocked — this URL does not allow cross-origin requests', 'error');
      } else {
        showToast('Failed to fetch URL', 'error');
      }
    } finally {
      setFetching(false);
    }
  }

  async function handleGenerate() {
    const key = getApiKey();
    if (!key) {
      setShowKeyInput(true);
      setKeyInput('');
      return;
    }
    setGenerating(true);
    try {
      const result = await generateCards(markdown, 'both');
      if (result.flashcards.length > 0) {
        setFlashcards(prev => [...prev, ...result.flashcards]);
      }
      if (result.quizQuestions.length > 0) {
        setQuizQuestions(prev => [...prev, ...result.quizQuestions]);
      }
      if (result.flashcards.length === 0 && result.quizQuestions.length === 0) {
        showToast('AI returned empty results — try notes with more content', 'error');
      } else {
        showToast(`Generated ${result.flashcards.length} cards + ${result.quizQuestions.length} questions`);
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

  function handleSave() {
    if (!title.trim()) {
      showToast('Please enter a title', 'error');
      return;
    }
    if (!markdown.trim()) {
      showToast('Please enter some content', 'error');
      return;
    }
    createNote(title.trim(), markdown, flashcards, quizQuestions);
    showToast('Note saved!');
    setTitle('');
    setMarkdown('');
    setFlashcards([]);
    setQuizQuestions([]);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.target as HTMLTextAreaElement;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const newValue = markdown.substring(0, start) + '  ' + markdown.substring(end);
      setMarkdown(newValue);
      requestAnimationFrame(() => {
        target.selectionStart = target.selectionEnd = start + 2;
      });
    }
  }

  const actionButtons = (
    <>
      <div className="upload-actions">
        <button className="btn btn-primary" onClick={handleSave}>
          Save Note
        </button>
        <button
          className="btn btn-primary"
          onClick={handleGenerate}
          disabled={generating || !markdown.trim()}
        >
          {generating ? 'Generating (may take ~60s)...' : 'Generate Flashcards & Quiz'}
        </button>
        <a className="btn btn-secondary" href={`${basePath}/my-notes/`}>
          My Notes
        </a>
      </div>

      {showKeyInput && (
        <div className="creator-section" style={{ marginTop: '0.75rem' }}>
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
    </>
  );

  return (
    <>
      <button className="editor-toggle" onClick={() => setEditorOpen(!editorOpen)}>
        <span className={`toggle-arrow ${editorOpen ? 'open' : ''}`}>&#9656;</span>
        {editorOpen ? 'Hide Editor' : 'Show Editor'}
      </button>
      <div className={`upload-layout ${editorOpen ? '' : 'preview-only'}`}>
        {editorOpen && (
          <div className="upload-panel">
            <input
              type="text"
              className="title-input"
              placeholder="Note title..."
              value={title}
              onChange={e => setTitle(e.target.value)}
            />
            <div className="file-upload-row">
              <label className="file-input-label">
                Upload .md file
                <input type="file" accept=".md,.mdx,.txt,.markdown" onChange={handleFile} />
              </label>
            </div>
            <div className="url-fetch-row">
              <input
                type="url"
                className="title-input"
                placeholder="Or paste a URL to a raw markdown file..."
                value={url}
                onChange={e => setUrl(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleFetchUrl(); }}
                style={{ marginBottom: 0, flex: 1 }}
              />
              <button
                className="btn btn-secondary"
                onClick={handleFetchUrl}
                disabled={fetching || !url.trim()}
              >
                {fetching ? 'Fetching...' : 'Fetch'}
              </button>
            </div>
            <textarea
              className="md-editor"
              placeholder={"Paste your markdown here...\n\nSupports LaTeX: $E = mc^2$\nCode blocks: ```python\nAnd all standard markdown."}
              value={markdown}
              onChange={e => setMarkdown(e.target.value)}
              onKeyDown={handleKeyDown}
              spellCheck={false}
            />
          </div>
        )}

        <div className="preview-panel">
          <h2>Preview</h2>
          <div className="preview-box">
            {markdown ? (
              <MarkdownPreview content={markdown} />
            ) : (
              <p style={{ color: 'var(--app-text-muted)', fontStyle: 'italic' }}>
                Start typing to see a live preview...
              </p>
            )}
          </div>
          {actionButtons}
        </div>
      </div>

      {flashcards.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h2>Flashcards</h2>
          <Flashcard cards={flashcards} />
        </div>
      )}

      {quizQuestions.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h2>Quiz</h2>
          <Quiz questions={quizQuestions} />
        </div>
      )}

      {toast && <div className={`toast toast-${toast.type}`}>{toast.msg}</div>}
    </>
  );
}
