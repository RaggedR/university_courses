# Study Tool

A static website for university notes with LaTeX rendering, syntax-highlighted code, interactive flashcards, and auto-generated quizzes.

**Live site:** https://raggedr.github.io/university_courses/

## Features

- **Markdown notes** with full LaTeX math support ($inline$ and $$display$$)
- **Code blocks** with syntax highlighting (all languages via Shiki)
- **Interactive flashcards** — click to flip, keyboard navigation, shuffle mode
- **Multiple-choice quizzes** — with scoring, explanations, and review
- **AI generation** — generate flashcards and quiz questions from your notes using Claude
- **Upload notes** — paste markdown or upload .md files, saved to browser localStorage
- **Export to .mdx** — download notes with flashcards/quizzes as ready-to-commit files
- **Full-text search** (Cmd+K) via Pagefind
- **Dark/light mode**
- **Mobile responsive**

## Stack

- [Astro](https://astro.build/) + [Starlight](https://starlight.astro.build/) — static site generation with docs theme
- [React](https://react.dev/) — interactive components (flashcards, quizzes, upload UI)
- [KaTeX](https://katex.org/) — LaTeX math rendering
- [Shiki](https://shiki.style/) — code syntax highlighting
- [Pagefind](https://pagefind.app/) — static search
- GitHub Pages — hosting via GitHub Actions

## How this was built

This project was built in a single session as a demo of [Claude Code](https://claude.ai/code) — Anthropic's agentic coding tool. The goal was to show what Claude Code is capable of when given a real task end-to-end: from initial architecture decisions through to a deployed site with interactive features.

Everything — the project scaffolding, component code, CSS, GitHub Actions workflow, and this README — was written by Claude Code with human direction on requirements.

## Development

```bash
npm install
npm run build && npm run preview   # Production preview at localhost:4321
```

## Adding notes

Drop `.mdx` files into `src/content/docs/` and add a sidebar entry in `astro.config.mjs`:

```js
sidebar: [
  {
    label: 'My Course',
    autogenerate: { directory: 'my-course' },
  },
],
```

See `src/content/docs/examples/demo.mdx` for an example with flashcards and quizzes.

## License

MIT
