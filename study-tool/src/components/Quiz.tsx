import { useState } from 'react';
import { renderMath } from './math-renderer';

interface Question {
  question: string;
  options: string[];
  answer: number; // 0-indexed
  explanation?: string;
}

interface QuizProps {
  questions: Question[];
  title?: string;
}

export default function Quiz({ questions, title }: QuizProps) {
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [answers, setAnswers] = useState<(number | null)[]>(
    new Array(questions.length).fill(null)
  );
  const [submitted, setSubmitted] = useState(false);

  const q = questions[current];
  const isAnswered = selected !== null;
  const isCorrect = selected === q.answer;

  function select(idx: number) {
    if (answers[current] !== null) return; // already locked in
    setSelected(idx);
  }

  function confirm() {
    if (selected === null) return;
    const newAnswers = [...answers];
    newAnswers[current] = selected;
    setAnswers(newAnswers);
  }

  function next() {
    if (current < questions.length - 1) {
      setCurrent(current + 1);
      setSelected(answers[current + 1]);
    }
  }

  function prev() {
    if (current > 0) {
      setCurrent(current - 1);
      setSelected(answers[current - 1]);
    }
  }

  function finish() {
    setSubmitted(true);
  }

  function reset() {
    setCurrent(0);
    setSelected(null);
    setAnswers(new Array(questions.length).fill(null));
    setSubmitted(false);
  }

  const score = answers.filter((a, i) => a === questions[i].answer).length;
  const allAnswered = answers.every(a => a !== null);
  const locked = answers[current] !== null;

  if (submitted) {
    return (
      <div className="quiz-container">
        {title && <div className="quiz-title">{title}</div>}
        <div className="quiz-results">
          <div className="quiz-score">
            {score} / {questions.length}
          </div>
          <div className="quiz-score-label">
            {score === questions.length
              ? 'Perfect!'
              : score >= questions.length * 0.7
                ? 'Well done!'
                : 'Keep studying!'}
          </div>

          <div className="quiz-review">
            {questions.map((q, i) => (
              <div
                key={i}
                className={`quiz-review-item ${answers[i] === q.answer ? 'correct' : 'incorrect'}`}
              >
                <div
                  className="quiz-review-question"
                  dangerouslySetInnerHTML={{ __html: renderMath(q.question) }}
                />
                <div className="quiz-review-answer">
                  {answers[i] === q.answer ? 'Correct' : `Your answer: ${q.options[answers[i]!]} — Correct: ${q.options[q.answer]}`}
                </div>
                {q.explanation && (
                  <div
                    className="quiz-review-explanation"
                    dangerouslySetInnerHTML={{ __html: renderMath(q.explanation) }}
                  />
                )}
              </div>
            ))}
          </div>

          <button className="quiz-btn" onClick={reset}>Try Again</button>
        </div>
      </div>
    );
  }

  return (
    <div className="quiz-container">
      {title && <div className="quiz-title">{title}</div>}

      <div className="quiz-progress">
        Question {current + 1} of {questions.length}
        <div className="quiz-progress-bar">
          <div
            className="quiz-progress-fill"
            style={{ width: `${(answers.filter(a => a !== null).length / questions.length) * 100}%` }}
          />
        </div>
      </div>

      <div
        className="quiz-question"
        dangerouslySetInnerHTML={{ __html: renderMath(q.question) }}
      />

      <div className="quiz-options">
        {q.options.map((opt, i) => {
          let cls = 'quiz-option';
          if (locked) {
            if (i === q.answer) cls += ' correct';
            else if (i === answers[current] && i !== q.answer) cls += ' incorrect';
          } else if (i === selected) {
            cls += ' selected';
          }

          return (
            <button
              key={i}
              className={cls}
              onClick={() => select(i)}
              disabled={locked}
              dangerouslySetInnerHTML={{ __html: renderMath(opt) }}
            />
          );
        })}
      </div>

      {locked && q.explanation && (
        <div
          className="quiz-explanation"
          dangerouslySetInnerHTML={{ __html: renderMath(q.explanation) }}
        />
      )}

      <div className="quiz-controls">
        <button className="quiz-btn" onClick={prev} disabled={current === 0}>
          &larr; Prev
        </button>

        {!locked && (
          <button className="quiz-btn primary" onClick={confirm} disabled={selected === null}>
            Confirm
          </button>
        )}

        {locked && current < questions.length - 1 && (
          <button className="quiz-btn primary" onClick={next}>
            Next &rarr;
          </button>
        )}

        {locked && current === questions.length - 1 && allAnswered && (
          <button className="quiz-btn primary" onClick={finish}>
            See Results
          </button>
        )}

        {!locked && current < questions.length - 1 && (
          <button className="quiz-btn" onClick={next}>
            Skip &rarr;
          </button>
        )}
      </div>
    </div>
  );
}
