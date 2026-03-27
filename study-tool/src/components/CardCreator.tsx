import { useState } from 'react';
import type { FlashcardData, QuizQuestionData } from './note-store';
import { getApiKey, setApiKey, generateCards } from './ai-generate';

interface CardCreatorProps {
  flashcards: FlashcardData[];
  quizQuestions: QuizQuestionData[];
  noteContent: string;
  onUpdateFlashcards: (cards: FlashcardData[]) => void;
  onUpdateQuiz: (questions: QuizQuestionData[]) => void;
}

export default function CardCreator({
  flashcards,
  quizQuestions,
  noteContent,
  onUpdateFlashcards,
  onUpdateQuiz,
}: CardCreatorProps) {
  const [tab, setTab] = useState<'flashcards' | 'quiz'>('flashcards');
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState('');
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [keyInput, setKeyInput] = useState(getApiKey());

  // Flashcard form state
  const [front, setFront] = useState('');
  const [back, setBack] = useState('');

  // Quiz form state
  const [question, setQuestion] = useState('');
  const [options, setOptions] = useState(['', '', '', '']);
  const [answer, setAnswer] = useState(0);
  const [explanation, setExplanation] = useState('');

  async function handleGenerate(mode: 'flashcards' | 'quiz' | 'both') {
    const key = getApiKey();
    if (!key) {
      setShowKeyInput(true);
      return;
    }

    setGenerating(true);
    setGenError('');

    try {
      const result = await generateCards(noteContent, mode);

      if (result.flashcards.length > 0) {
        onUpdateFlashcards([...flashcards, ...result.flashcards]);
      }
      if (result.quizQuestions.length > 0) {
        onUpdateQuiz([...quizQuestions, ...result.quizQuestions]);
      }

      if (result.flashcards.length === 0 && result.quizQuestions.length === 0) {
        setGenError('AI returned empty results. Try notes with more content.');
      }
    } catch (err: any) {
      setGenError(err.message || 'Generation failed');
      if (err.message === 'Invalid API key') {
        setShowKeyInput(true);
      }
    } finally {
      setGenerating(false);
    }
  }

  function saveKey() {
    setApiKey(keyInput.trim());
    setShowKeyInput(false);
    setGenError('');
  }

  function addFlashcard() {
    if (!front.trim() || !back.trim()) return;
    onUpdateFlashcards([...flashcards, { front: front.trim(), back: back.trim() }]);
    setFront('');
    setBack('');
  }

  function removeFlashcard(idx: number) {
    onUpdateFlashcards(flashcards.filter((_, i) => i !== idx));
  }

  function addQuestion() {
    const filledOptions = options.filter(o => o.trim());
    if (!question.trim() || filledOptions.length < 2) return;
    onUpdateQuiz([
      ...quizQuestions,
      {
        question: question.trim(),
        options: filledOptions,
        answer: Math.min(answer, filledOptions.length - 1),
        ...(explanation.trim() ? { explanation: explanation.trim() } : {}),
      },
    ]);
    setQuestion('');
    setOptions(['', '', '', '']);
    setAnswer(0);
    setExplanation('');
  }

  function removeQuestion(idx: number) {
    onUpdateQuiz(quizQuestions.filter((_, i) => i !== idx));
  }

  function updateOption(idx: number, value: string) {
    const next = [...options];
    next[idx] = value;
    setOptions(next);
  }

  return (
    <div>
      {/* ── AI Generate ── */}
      <div className="creator-section" style={{ marginTop: 0, marginBottom: '1rem' }}>
        <h3>Generate with AI</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--app-text-muted)', marginBottom: '0.75rem' }}>
          Uses Claude to create flashcards and quiz questions from your notes.
        </p>

        {showKeyInput && (
          <div className="creator-form" style={{ marginBottom: '0.75rem' }}>
            <label>Anthropic API Key</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="password"
                placeholder="sk-ant-..."
                value={keyInput}
                onChange={e => setKeyInput(e.target.value)}
                style={{ flex: 1 }}
              />
              <button className="btn btn-primary btn-sm" onClick={saveKey} disabled={!keyInput.trim()}>
                Save Key
              </button>
            </div>
            <span style={{ fontSize: '0.8rem', color: 'var(--app-text-muted)' }}>
              Stored locally in your browser. Never sent anywhere except Anthropic's API.
              {' '}<a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--app-accent)' }}>Get an API key</a>
            </span>
          </div>
        )}

        {genError && (
          <div style={{ color: 'var(--app-danger)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            {genError}
          </div>
        )}

        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <button
            className="btn btn-primary btn-sm"
            onClick={() => handleGenerate('both')}
            disabled={generating}
          >
            {generating ? 'Generating...' : 'Generate Both'}
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => handleGenerate('flashcards')}
            disabled={generating}
          >
            Flashcards Only
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => handleGenerate('quiz')}
            disabled={generating}
          >
            Quiz Only
          </button>

          {!showKeyInput && (
            <button
              className="btn btn-sm"
              onClick={() => setShowKeyInput(true)}
              style={{ marginLeft: 'auto', background: 'none', color: 'var(--app-text-muted)', fontSize: '0.8rem' }}
            >
              {getApiKey() ? 'Change API Key' : 'Set API Key'}
            </button>
          )}
        </div>
      </div>

      {/* ── Manual tabs ── */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
        <button
          className={`btn btn-sm ${tab === 'flashcards' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setTab('flashcards')}
        >
          Flashcards ({flashcards.length})
        </button>
        <button
          className={`btn btn-sm ${tab === 'quiz' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setTab('quiz')}
        >
          Quiz Questions ({quizQuestions.length})
        </button>
      </div>

      {tab === 'flashcards' && (
        <div className="creator-section">
          <h3>Add Flashcard</h3>
          <div className="creator-form">
            <label>Front (question/term)</label>
            <textarea
              placeholder="What is the derivative of $x^n$?"
              value={front}
              onChange={e => setFront(e.target.value)}
              rows={2}
            />
            <label>Back (answer/definition)</label>
            <textarea
              placeholder="$nx^{n-1}$ (power rule)"
              value={back}
              onChange={e => setBack(e.target.value)}
              rows={2}
            />
            <button
              className="btn btn-primary btn-sm"
              onClick={addFlashcard}
              disabled={!front.trim() || !back.trim()}
            >
              + Add Card
            </button>
          </div>

          {flashcards.length > 0 && (
            <div className="existing-cards">
              <label>{flashcards.length} card{flashcards.length !== 1 ? 's' : ''}</label>
              {flashcards.map((c, i) => (
                <div key={i} className="existing-card">
                  <div className="existing-card-content">
                    <strong>Q:</strong> {c.front}<br />
                    <strong>A:</strong> {c.back}
                  </div>
                  <button className="delete-btn" onClick={() => removeFlashcard(i)} title="Remove">
                    &times;
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {tab === 'quiz' && (
        <div className="creator-section">
          <h3>Add Quiz Question</h3>
          <div className="creator-form">
            <label>Question</label>
            <textarea
              placeholder="What is the integral of $\cos(x)$?"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              rows={2}
            />

            <label>Options (at least 2, select the correct one)</label>
            <div className="options-grid">
              {options.map((opt, i) => (
                <div key={i} className="option-row">
                  <input
                    type="radio"
                    name="correct-answer"
                    checked={answer === i}
                    onChange={() => setAnswer(i)}
                  />
                  <input
                    type="text"
                    placeholder={`Option ${i + 1}`}
                    value={opt}
                    onChange={e => updateOption(i, e.target.value)}
                  />
                </div>
              ))}
            </div>

            <label>Explanation (optional)</label>
            <textarea
              placeholder="Explain why the answer is correct..."
              value={explanation}
              onChange={e => setExplanation(e.target.value)}
              rows={2}
            />

            <button
              className="btn btn-primary btn-sm"
              onClick={addQuestion}
              disabled={!question.trim() || options.filter(o => o.trim()).length < 2}
            >
              + Add Question
            </button>
          </div>

          {quizQuestions.length > 0 && (
            <div className="existing-cards">
              <label>{quizQuestions.length} question{quizQuestions.length !== 1 ? 's' : ''}</label>
              {quizQuestions.map((q, i) => (
                <div key={i} className="existing-card">
                  <div className="existing-card-content">
                    <strong>Q:</strong> {q.question}<br />
                    Options: {q.options.join(' | ')} (correct: {q.options[q.answer]})
                  </div>
                  <button className="delete-btn" onClick={() => removeQuestion(i)} title="Remove">
                    &times;
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
