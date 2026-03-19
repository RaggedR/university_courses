# How to Use Claude Code

*A guide for people who've never had an AI build things for them.*

---

## The One-Sentence Version

You have the ideas. Claude has the implementation skills. Your job is to imagine clearly and steer decisively.

---

## Part 1: What Claude Code Actually Is

Claude Code is a command-line tool. It's not a chatbot in a browser — it lives in your terminal, right next to `git` and `npm` and `python`. You type `claude` in a directory and start a conversation, but unlike ChatGPT, Claude Code can *do things*: read your files, edit your code, run commands, search the web, and build entire projects.

Think of it as a junior developer sitting next to you who can type at 1,000 words per minute, has read most of the internet, but has no idea what you're trying to build until you tell it.

### Starting a Session

```bash
cd ~/my-project     # navigate to your project directory
claude               # start Claude Code
```

That's it. Claude opens in your terminal, ready to work. Everything it does will be relative to the directory you started in.

### It's a Terminal, Not a Chatbox

Claude Code shows you what it's doing. When it reads a file, you see which file. When it edits code, you see the diff. When it runs a command, you see the command and its output. This transparency is the point — you're not blindly trusting an AI, you're watching a collaborator work and approving each step.

---

## Part 2: Imagination Is the Bottleneck

This is the most important section in this document.

Claude can write any code you can describe. It can build frontends, backends, databases, APIs, mobile apps, data pipelines, browser extensions, CLI tools. It knows most programming languages, most frameworks, most design patterns.

**The limiting factor is not Claude's ability. It's your ability to imagine what you want.**

If you sit down and type "build me something cool," you'll get something generic and forgettable. If you sit down and type "I want an app where I type a word and it shows me that word translated into 30 languages, displayed on a rotating globe," you'll get something remarkable.

The people who get the most out of Claude Code are the ones who:
- Have a clear picture of what they want to build
- Can describe it in concrete, specific terms
- Know when the output isn't right and can articulate *why*

You don't need to know *how* to build it. You need to know *what* you want. The how is Claude's job.

### What If You Don't Know What You Want?

That's fine — Claude can help you figure it out. Start a conversation:

> "I want to build something with the Spotify API. I'm interested in music discovery. What could we build?"

Claude will brainstorm with you. But the best ideas will come from your own frustrations, curiosities, and interests. Claude can build anything — the question is what's worth building.

### If You Don't Understand Something, Just Ask

This sounds obvious but it changes everything. When you hit a concept you don't understand — CORS, middleware, WebSockets, Docker, whatever — your first instinct might be to open a browser tab and start Googling. Don't. Just ask Claude:

> "I keep seeing the word 'middleware' in Express code. Give me a clear and simple explanation with diagrams and examples."

Claude will explain it right there, in context, tailored to your level, with examples that relate to what you're actually building. No scrolling through Stack Overflow. No watching a 40-minute YouTube video to extract 2 minutes of useful information.

This works for anything:
- "What's the difference between SQL and NoSQL? Explain it simply."
- "I don't understand async/await. Walk me through it with a diagram."
- "Why does my frontend need to talk to my backend? Why can't it call Google directly?"
- "What actually happens when I type npm install?"

You have a patient, knowledgeable tutor sitting in your terminal. Use it. The people who learn fastest are the ones who ask the most questions.

---

## Part 3: Permissions — Trust Gradually

When you first start Claude Code, it asks your permission before doing almost anything: reading files, editing code, running commands. This is good. **You should start restrictive and open up gradually.**

### The Permission Levels

Claude Code has three main permission modes, from cautious to autonomous:

**Ask every time** (default for new users): Claude asks before every file read, every edit, every command. Tedious, but you see everything. Start here.

**Allow some things automatically**: As you get comfortable, you grant blanket permissions for safe operations:
- Reading files — always safe, let Claude read freely
- Editing files — safe if you're using git (you can always revert)
- Running commands — this is where you should be more careful

**YOLO mode** (the community's name, not the official one): Claude runs everything without asking. Only use this if you trust your setup and have good version control habits. Not recommended for beginners.

### The Principle: Progressive Trust

Think of it like handing someone the keys to your house:

1. **Week 1**: Watch everything. Read every diff. Approve every command. Learn what Claude typically does.
2. **Week 2**: Auto-approve file reads and edits. You're using git, so any edit is reversible. Still approve shell commands.
3. **Week 3**: Auto-approve common safe commands (`npm install`, `npm run dev`, test commands). Still manually approve anything destructive.
4. **Later**: You'll develop a feel for what's safe. Some people go full auto. Some never do. Both are fine.

The important thing: **you can always say no.** If Claude proposes something you don't understand, reject it and ask what it's trying to do. Claude will explain and suggest alternatives.

### What Could Go Wrong?

The real risks are:
- **Running destructive commands**: `rm -rf`, `git push --force`, dropping database tables. Claude is trained to be cautious about these, but you're the last line of defence.
- **Exposing secrets**: Claude won't intentionally commit your `.env` file, but mistakes happen. Always check `git diff` before committing.
- **Runaway costs**: If Claude is calling paid APIs (OpenAI, etc.) in a loop, it can rack up a bill. Watch for this.

None of these are likely if you're paying attention. They're just worth knowing about.

---

## Part 4: Directories Matter

This sounds mundane but it's one of the most important practical lessons.

### Always Start Claude in the Right Directory

Claude Code operates relative to where you launched it. If you're working on your translation app but you accidentally started Claude in your home directory, it won't know about your project files, won't find your `package.json`, and will be confused.

```bash
# Wrong — Claude doesn't know which project you mean
cd ~
claude

# Right — Claude immediately sees your project files
cd ~/projects/hello-world-translator
claude
```

### One Project, One Directory, One Session

Keep your projects in separate directories. Don't dump multiple projects into a single folder. When Claude reads your files to understand the codebase, a clean directory structure means it understands faster and makes fewer mistakes.

```
~/projects/
├── hello-world-translator/     # start claude here for this project
├── portfolio-website/          # start claude here for this project
└── data-analysis-scripts/      # start claude here for this project
```

### New Topic? New Session.

If you've been working on your backend and want to switch to something completely unrelated — writing an essay, exploring a new API, learning a concept — start a fresh session. Claude carries the full conversation in its context window, and leftover context from a previous topic can cause confusion.

In the terminal: `/clear` starts a fresh conversation within the same session. Or just quit (`/exit` or Ctrl+C) and start a new `claude` session.

---

## Part 5: The Bulletin Board — CLAUDE.md

This is one of the most powerful features and the one least obvious to new users.

### The Problem

Claude Code starts every session with amnesia. It doesn't remember what you talked about yesterday, what conventions your project uses, or that you prefer tabs over spaces. Every new session is a blank slate.

### The Solution: CLAUDE.md

`CLAUDE.md` is a file that Claude reads at the start of every session. It's a shared bulletin board — persistent instructions that survive across conversations. Whatever you write in `CLAUDE.md`, Claude will follow.

There are three levels:

**Global** (`~/.claude/CLAUDE.md`): Instructions that apply to everything you do with Claude, across all projects. Your personal preferences and working style.

```markdown
# My Preferences
- I prefer Python for backend code
- Always use type hints
- Run tests after making changes
- I'm a beginner — explain your changes
```

**Project** (`CLAUDE.md` in your project root): Instructions specific to this project. Technical details, architecture decisions, conventions.

```markdown
# Hello World Translator
- Backend: Express on port 3000
- Frontend: React + Vite on port 5173
- API key is in .env as GOOGLE_TRANSLATE_API_KEY
- To run: npm start in /backend, npm run dev in /frontend
```

**Sub-directory** (`CLAUDE.md` in any sub-folder): Instructions specific to that part of the project. Useful for large projects with different conventions in different areas.

### Why This Matters

Without `CLAUDE.md`, you'd repeat yourself every session: "Remember, we're using Express, the API key is in .env, don't use semicolons in our JavaScript..." With it, Claude already knows.

### Setting It Up

Run `/init` inside Claude Code. It will look at your project and draft a `CLAUDE.md` for you. Review it, edit it, and commit it to your repository. Everyone who works on the project (including future Claude sessions) benefits.

**Pro tip**: Run `/init` periodically, especially before the conversation gets too long. Claude will update `CLAUDE.md` with things it learned during the session — conventions, architecture decisions, gotchas. This is how you build up institutional memory.

---

## Part 6: Working Style — How to Get the Best Results

### Be Specific, Not Vague

```
Bad:  "Make the app better"
Good: "Add error handling to the /api/translate endpoint — if Google's
       API returns an error, send a clear error message back to the
       frontend instead of crashing"
```

The more concrete your request, the better the output. You don't need to know *how* to implement it — just describe *what* you want to see happen.

### Push Back When It's Wrong

Claude's first attempt isn't always right. That's normal. The people who get the best results are the ones who say:

- "That's not what I meant — I wanted X, not Y"
- "This is too complicated. Simplify it."
- "Why did you add all those extra files? I just wanted one function."
- "The backend doesn't just talk to databases — it also calls external APIs" ← actual correction that improved a document significantly

Don't accept output you're not happy with. Claude responds well to correction and will adjust.

### Iterate, Don't Specify Everything Upfront

You don't need a perfect spec before you start. The best workflow is:

1. Describe roughly what you want
2. Look at what Claude produces
3. React: "this part is good, this part needs to change"
4. Repeat

This is faster and produces better results than trying to write a comprehensive brief. You'll discover what you actually want by seeing drafts.

### Give Claude Hard Problems

Claude works better on interesting, challenging problems than on trivial ones. This sounds anthropomorphic but it's practically true — the model engages more deeply with complex tasks. If you find yourself giving Claude a series of boring, repetitive tasks, consider whether you can frame it as one interesting problem instead.

```
Boring:  "Add a button. Now add another button. Now add a form."
Better:  "I need a translation interface that lets users type text,
          select target languages, and see results appear in real time.
          Show a loading state while waiting."
```

### Know When to Start Fresh

If a conversation has gone on too long and Claude seems confused — repeating mistakes, forgetting context, going in circles — start a new session. Long conversations accumulate noise. A fresh start with a clear description of where you are and what you need is often faster than continuing.

---

## Part 7: Explanation Mode

Claude Code has different output styles. **Explanation mode** is the one worth knowing about.

In normal mode, Claude is terse — it does the work, shows you the changes, and moves on. Efficient, but you might not learn anything.

In explanation mode, Claude explains *why* it's making each decision. It teaches you as it works. This is invaluable when you're learning a new technology, framework, or codebase.

You'll see things like:

> *★ Insight: We're using `async/await` here because the Google API call takes ~200ms. Without `await`, the function would return before the response arrives and you'd get `undefined` instead of a translation.*

**When to use explanation mode**: when you're learning, exploring a new codebase, or working with unfamiliar technology.

**When to turn it off**: when you know what you're doing and just want Claude to work fast.

---

## Part 8: Practical Habits

### Use Git

Always, always work in a git repository. This is your safety net:
- Every change Claude makes can be reviewed with `git diff`
- Every change can be reverted with `git checkout`
- You can experiment freely knowing nothing is permanent

If you're not using git, you're working without a net. Claude can make mistakes. Git means those mistakes are always reversible.

### Read What Claude Writes

Don't just approve everything blindly. Read the code. Read the diffs. You don't need to understand every line, but you should understand the *shape* of what changed. "It added a new file for the API route" — that's enough to know it's on the right track. "It deleted half my frontend and replaced it with something completely different" — that's a sign to stop and ask questions.

### Ask Claude to Explain Your Own Code

One of the most underrated uses: open Claude in an existing project and ask it to explain what the code does. It will read the files and give you a walkthrough. This is incredibly useful for:
- Codebases you inherited from someone else
- Code you wrote six months ago and forgot
- Open-source projects you want to understand

### Use Claude for Non-Code Tasks

Claude Code isn't just for writing code. It can:
- Write documentation and teaching materials
- Create Anki flashcards from material you're studying
- Draft emails, reports, and proposals
- Analyse data and create visualisations
- Research topics by searching the web
- Review and explain academic papers

If it involves text, Claude can probably help.

---

## Part 9: What Claude Code Costs

Claude Code is included with Anthropic's subscription plans. You pay a monthly subscription (Pro, Team, or Enterprise), and Claude Code usage comes out of your plan's allowance. It's not billed per-token like the API — you get a usage quota that resets each month.

If you use it heavily, you may hit your quota before the month ends. When that happens, you'll either need to wait for it to reset or upgrade your plan.

Things that use more of your quota:
- Long conversations (more context for Claude to process each turn)
- Reading large files (all that text is part of the conversation)
- Using the most capable models (Opus uses more quota than Sonnet)

Things that keep usage down:
- Starting fresh sessions for new topics (less accumulated context)
- Being specific (fewer back-and-forth rounds)
- Using `/compact` to compress the conversation when it gets long

---

## Part 10: The Mental Model

Think of Claude Code as a collaboration:

```
┌───────────────────────────────────────────────────┐
│                                                   │
│   YOU                        CLAUDE               │
│                                                   │
│   • Imagination              • Architecture       │
│   • Direction                • Implementation     │
│   • Quality control          • Speed              │
│   • Domain knowledge         • Breadth of         │
│   • The final decision         knowledge          │
│                              • "Here are three    │
│                                 ways to do that"  │
│                                                   │
│              CLAUDE.md                            │
│          (shared memory)                          │
│                                                   │
│              Git                                  │
│          (safety net)                             │
│                                                   │
└───────────────────────────────────────────────────┘
```

Programming with Claude Code is a conversation. Not architect and builder — more like two people at a whiteboard. You say "I want a translation app." Claude says "here are three ways to structure that, with trade-offs." You say "the second one, but simpler." Claude builds it. You look at it and say "this part's not right." Claude adjusts.

You don't need to know the answer before you start. You need to be willing to react honestly to what you see, push back when it's wrong, and say "yes" when it's right. The final decision is always yours. But getting there is a dialogue.

The better you get at imagining clearly, describing precisely, and steering decisively, the more extraordinary things you can build.

Now go imagine something.
