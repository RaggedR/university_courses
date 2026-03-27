import { useState, useCallback } from 'react';
import { createNote } from './note-store';
import MarkdownPreview from './MarkdownPreview';

export default function UploadPage({ basePath }: { basePath: string }) {
  const [title, setTitle] = useState('');
  const [markdown, setMarkdown] = useState('');
  const [url, setUrl] = useState('');
  const [fetching, setFetching] = useState(false);
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

  async function handleFetchUrl() {
    const trimmed = url.trim();
    if (!trimmed) return;
    setFetching(true);
    try {
      const res = await fetch(trimmed);
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

  function handleSave() {
    if (!title.trim()) {
      showToast('Please enter a title', 'error');
      return;
    }
    if (!markdown.trim()) {
      showToast('Please enter some content', 'error');
      return;
    }
    createNote(title.trim(), markdown);
    showToast('Note saved!');
    setTitle('');
    setMarkdown('');
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

  return (
    <>
      <div className="upload-layout">
        <div className="upload-panel">
          <h2>Editor</h2>
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
          <div className="upload-actions">
            <button className="btn btn-primary" onClick={handleSave}>
              Save Note
            </button>
            <a className="btn btn-secondary" href={`${basePath}/my-notes/`}>
              My Notes
            </a>
          </div>
        </div>

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
        </div>
      </div>

      {toast && <div className={`toast toast-${toast.type}`}>{toast.msg}</div>}
    </>
  );
}
