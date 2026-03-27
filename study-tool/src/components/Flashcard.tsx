import { useState, useRef, useEffect, useCallback } from 'react';
import { renderMath } from './math-renderer';

interface Card {
  front: string;
  back: string;
}

interface FlashcardProps {
  cards: Card[];
  title?: string;
}

export default function Flashcard({ cards, title }: FlashcardProps) {
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [shuffle, setShuffle] = useState(false);
  const [order, setOrder] = useState<number[]>(cards.map((_, i) => i));
  const cardRef = useRef<HTMLDivElement>(null);

  const currentCard = cards[order[index]];

  const doShuffle = useCallback(() => {
    const shuffled = [...order];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    setOrder(shuffled);
    setIndex(0);
    setFlipped(false);
  }, [order]);

  useEffect(() => {
    if (shuffle) doShuffle();
  }, [shuffle]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        setFlipped(f => !f);
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        next();
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        prev();
      }
    };

    const el = cardRef.current;
    el?.addEventListener('keydown', handleKey);
    return () => el?.removeEventListener('keydown', handleKey);
  }, [index, cards.length]);

  function next() {
    setFlipped(false);
    setIndex(i => (i + 1) % cards.length);
  }

  function prev() {
    setFlipped(false);
    setIndex(i => (i - 1 + cards.length) % cards.length);
  }

  return (
    <div className="flashcard-container">
      {title && <div className="flashcard-title">{title}</div>}

      <div
        ref={cardRef}
        className={`flashcard ${flipped ? 'flipped' : ''}`}
        onClick={() => setFlipped(f => !f)}
        tabIndex={0}
        role="button"
        aria-label={flipped ? 'Answer side. Click to see question.' : 'Question side. Click to see answer.'}
      >
        <div className="flashcard-inner">
          <div
            className="flashcard-face flashcard-front"
            dangerouslySetInnerHTML={{ __html: renderMath(currentCard.front) }}
          />
          <div
            className="flashcard-face flashcard-back"
            dangerouslySetInnerHTML={{ __html: renderMath(currentCard.back) }}
          />
        </div>
      </div>

      <div className="flashcard-controls">
        <button onClick={prev} aria-label="Previous card">&larr; Prev</button>
        <span className="flashcard-counter">
          {index + 1} / {cards.length}
        </span>
        <button onClick={next} aria-label="Next card">Next &rarr;</button>
      </div>

      <div className="flashcard-controls">
        <label className="flashcard-shuffle">
          <input
            type="checkbox"
            checked={shuffle}
            onChange={e => setShuffle(e.target.checked)}
          />
          Shuffle
        </label>
      </div>
    </div>
  );
}
