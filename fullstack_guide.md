# Full-Stack Web Development for Computer Scientists

*Everything they didn't teach you in your algorithms course.*

You can write a red-black tree from scratch but you've never deployed a website. This document is for you.

### What We're Building

By the end of this document, you'll understand every layer of a full-stack web application. The running example is a **Hello World translator** — an app with a single button that translates "Hello, World!" into eight languages at once:

```
┌──────────────────────────────────────┐
│  Hello World Translator              │
│                                      │
│  [Translate Hello World!]            │
│                                      │
│  French:     Bonjour le monde !      │
│  Spanish:    ¡Hola, mundo!           │
│  Japanese:   こんにちは世界！          │
│  Arabic:     !مرحبا بالعالم          │
│  Swahili:    Salamu, Dunia!          │
│  Chinese:    你好世界！               │
│  Hindi:      नमस्ते दुनिया!            │
│  German:     Hallo Welt!             │
└──────────────────────────────────────┘
```

Simple enough to understand in one glance. But it's *not* a static page — when you click that button, a chain of real things happens: the browser talks to your server, your server talks to Google Translate (using a secret API key), Google sends back translations, and your server passes them to the browser. Every concept in this document exists to make that chain work.

---

## Part 1: The Network Layer

### HTTP — The Language of the Web

You already know about protocols from your networking course. HTTP (HyperText Transfer Protocol) is the protocol that browsers use to talk to servers. Every time you visit a website, your browser sends an HTTP **request** and gets back an HTTP **response**.

An HTTP request has:
- A **method** (what you want to do — more on this below)
- A **URL** (where you want to do it)
- **Headers** (metadata — think of them as key-value pairs stapled to the envelope)
- A **body** (optional — the actual data you're sending)

An HTTP response has:
- A **status code** (200 = OK, 404 = not found, 500 = server broke)
- **Headers** (metadata coming back)
- A **body** (the actual content — HTML, JSON, an image, whatever)

This is just text over TCP. There is no magic. You could type HTTP requests by hand into a raw TCP socket and it would work.

### HTTP Methods — The Verbs

HTTP defines several methods. The main ones:

| Method | Purpose | Has a body? | Example |
|--------|---------|-------------|---------|
| `GET` | Retrieve data | No | Loading a webpage, fetching user profiles |
| `POST` | Send new data | Yes | Submitting a form, creating an account |
| `PUT` | Replace existing data | Yes | Updating your entire profile |
| `PATCH` | Partially update data | Yes | Changing just your email |
| `DELETE` | Remove data | Sometimes | Deleting a comment |

When you type a URL into your browser and hit Enter, that's a GET request. When you submit a login form, that's typically a POST request.

### URLs and Ports

You know what a URL is. But consider this one:

```
http://localhost:3000/api/users/42
```

Breaking it down:
- `http://` — the protocol
- `localhost` — the hostname (in this case, your own machine)
- `:3000` — the **port**
- `/api/users/42` — the **path**

**Ports** are how one machine runs multiple network services simultaneously. Your computer has 65,535 available ports. Think of the hostname as a building's street address and the port as the apartment number. Common conventions:
- Port 80: HTTP (default, so browsers hide it)
- Port 443: HTTPS (encrypted HTTP)
- Port 3000: A popular default for Node.js backend servers
- Port 5173: Vite's default for frontend dev servers
- Port 5432: PostgreSQL database

During development, your frontend might run on `localhost:5173` and your backend on `localhost:3000`. Same machine, different ports, two separate programs.

---

## Part 2: APIs

### What Is an API?

API stands for Application Programming Interface. In the broadest sense, it's any defined interface that lets one piece of software talk to another. The C standard library is an API. A Python class's public methods are an API.

But in web development, "API" almost always means a **web API**: a server that accepts HTTP requests and returns data (usually JSON). Instead of returning HTML pages meant for humans to look at, it returns structured data meant for programs to consume.

For example, if you visit `https://api.github.com/users/torvalds` in your browser, you won't get a webpage — you'll get raw JSON data about Linus Torvalds's GitHub profile. That's a web API.

### What Is REST?

REST (Representational State Transfer) is an architectural style for designing web APIs. It's not a protocol or a standard — it's a set of conventions. A "RESTful" API typically:

1. **Uses URLs to represent resources**: `/users`, `/users/42`, `/users/42/posts`
2. **Uses HTTP methods as verbs**: GET to read, POST to create, PUT/PATCH to update, DELETE to remove
3. **Is stateless**: each request contains all the information needed to process it (the server doesn't remember your previous requests)

So a REST API for a blog might look like:

```
GET    /api/posts          → list all posts
GET    /api/posts/7        → get post #7
POST   /api/posts          → create a new post (data in request body)
PUT    /api/posts/7        → update post #7 (data in request body)
DELETE /api/posts/7        → delete post #7
```

This is just a convention. Nobody enforces it. Plenty of APIs break these rules. But it's the dominant pattern you'll encounter.

### What Is JSON?

JSON (JavaScript Object Notation) is a text format for structured data. It looks like this:

```json
{
  "name": "Ada Lovelace",
  "age": 36,
  "languages": ["English", "French", "Mathematics"],
  "address": {
    "city": "London",
    "country": "England"
  },
  "alive": false
}
```

If you know Python dictionaries, you already know 90% of JSON. The key differences from Python:
- Strings must use double quotes (not single)
- Booleans are `true`/`false` (not `True`/`False`)
- No trailing commas allowed
- `null` instead of `None`
- No comments allowed

JSON has won. It's the default format for almost all web APIs. XML lost. YAML is used for configuration files but not for API communication. When your frontend sends data to your backend, it's almost certainly JSON. When the backend responds, it's almost certainly JSON.

### What Is an API Key?

Many APIs require authentication. An API key is the simplest form: a long random string that identifies you. It's like a password for your application.

```
sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234
```

When you sign up for a service (OpenAI, Stripe, Google Maps, etc.), they give you an API key. You include it in your HTTP requests so the service knows:
1. **Who you are** (for billing and rate limiting)
2. **That you're authorised** (not just anyone can use the API)

You typically send it in an HTTP header:

```
Authorization: Bearer sk-proj-abc123def456...
```

### Where Does the Key Go? The .env File

**Never put API keys directly in your code.** If your code goes to GitHub, so does the key. Bots scan GitHub continuously for accidentally exposed keys. Within minutes, someone will find yours and start making API calls on your account.

Instead, you put secrets in a file called `.env` (short for "environment"):

```bash
# .env
OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234
DATABASE_URL=postgres://user:password@localhost:5432/mydb
SECRET_SAUCE=extra-spicy
```

This file sits in your project root and is **listed in `.gitignore`** so Git never tracks it. Your code reads from it at runtime:

```python
# Python
import os
api_key = os.environ["OPENAI_API_KEY"]
```

```javascript
// JavaScript (Node.js)
const apiKey = process.env.OPENAI_API_KEY;
```

A library like `dotenv` loads the `.env` file into your process's environment variables when the app starts. The key exists only in memory at runtime, never in your source code.

**Rule of thumb**: if it's a secret, it goes in `.env`. If `.env` is not in your `.gitignore`, stop everything and fix that first.

### APIs Cost Money

This is the thing nobody tells you upfront: most useful APIs are **paid services**. When your backend calls the OpenAI API to generate text, or the Google Translate API to translate a sentence, you are making an HTTP request to someone else's server — and they charge you for it.

You typically pay **per request** or **per unit of usage** (per token, per character, per image, per second of audio). The costs vary enormously:

**AI / Machine Learning APIs** (the expensive ones):

| Service | What it does | Approximate cost |
|---------|-------------|-----------------|
| OpenAI GPT-4o | Text generation (flagship) | ~$2.50 / 1M input tokens, ~$10 / 1M output tokens |
| OpenAI GPT-4o-mini | Text generation (cheap, fast) | ~$0.15 / 1M input tokens, ~$0.60 / 1M output tokens |
| **Anthropic Claude Sonnet** | **Text generation** | **~$3 / 1M input, ~$15 / 1M output** |
| **Anthropic Claude Opus** | **Text generation (flagship)** | **~$15 / 1M input, ~$75 / 1M output** |
| OpenAI DALL-E 3 | Image generation | ~$0.04–$0.08 per image |
| OpenAI Whisper | Speech-to-text | ~$0.006 per minute of audio |
| OpenAI text-embedding-3-small | Convert text to vectors | ~$0.02 / 1M tokens |
| Google Text-to-Speech | Generate spoken audio | ~$4–$16 per 1M characters |

> **What's a token?** Roughly ¾ of a word. "Hello, how are you?" is about 6 tokens. A page of text is ~400 tokens. So generating a full page of text with Claude Opus costs about 3 cents — which sounds cheap until your app does it 10,000 times a day and you owe Anthropic $300.

**Utility APIs** (cheaper, sometimes free):

| Service | What it does | Approximate cost |
|---------|-------------|-----------------|
| Google Translate | Text translation | $20 per 1M characters |
| Google Maps | Maps, geocoding, directions | $5–$7 per 1,000 requests |
| Stripe | Payment processing | 2.9% + $0.30 per transaction |
| Twilio | Send SMS messages | ~$0.0079 per SMS |
| SendGrid | Send emails | Free up to 100/day, then tiered |

**Free APIs** (they exist!):

| Service | What it does | Limits |
|---------|-------------|--------|
| GitHub API | Repository data, issues, PRs | 5,000 requests/hour (authenticated) |
| Open-Meteo | Weather data | Unlimited, no key needed |
| REST Countries | Country data | Unlimited, no key needed |

**How billing works in practice**: you go to the provider's website (e.g., platform.openai.com), create an account, enter your credit card, and they give you an API key. Every API call your app makes is logged against that key. At the end of the month, they charge your card. Most services let you set spending limits so you don't get a surprise bill.

**The critical mistake new developers make**: putting an AI API key in frontend code. If your API key is in JavaScript that runs in the browser, anyone can open DevTools, find the key, and make API calls on your account. This is why expensive API calls should *always* go through your backend — the key lives in your server's `.env` file, never in the browser.

```
  Browser                    Your Backend              OpenAI
 ┌──────────┐              ┌─────────────┐          ┌──────────┐
 │ User     │  "translate  │  Express    │  API     │          │
 │ clicks   │──this text"─→│  server     │──call──→ │  GPT-4o  │
 │ button   │              │  (has key   │          │          │
 │          │←─"resultado"─│   in .env)  │←─reply──│          │
 └──────────┘              └─────────────┘          └──────────┘
                                  ↑
                           Key stays here.
                           Never in the browser.
```

---

## Part 3: Frontend and Backend

### The Architecture

A modern web application is two separate programs:

```
┌─────────────────┐         HTTP / JSON         ┌─────────────────┐
│                 │  ←─────────────────────────→ │                 │
│    FRONTEND     │    GET /api/users            │    BACKEND      │
│                 │    POST /api/login           │                 │
│  Runs in the    │    { "email": "a@b.com" }    │  Runs on a      │
│  user's browser │                              │  server          │
│                 │  ←── { "token": "xyz..." } ──│                 │
│  (JavaScript)   │                              │  (Any language)  │
└─────────────────┘                              └────────┬────────┘
                                                          │
                                                          │ SQL queries
                                                          │
                                                    ┌─────┴─────┐
                                                    │ DATABASE  │
                                                    └───────────┘
```

**Frontend** (also called "client-side"): the code that runs in the user's browser. It handles what things look like, what happens when you click a button, and making requests to the backend. Written in HTML, CSS, and JavaScript.

**Backend** (also called "server-side"): the code that runs on a server (your machine during development, a cloud server in production). It handles business logic, database access, authentication, and anything that requires secrets or trust. Can be written in any language — JavaScript (Node.js), Python, Go, Java, Ruby, etc.

**Why separate them?** Because the frontend runs on the user's machine, which means the user can see and modify *everything* in the frontend. You cannot trust the frontend. Any validation, authentication, or secret-keeping must happen on the backend, which you control.

### How They Communicate

During development, you run both on your machine:

```
Frontend:  http://localhost:5173    (Vite dev server)
Backend:   http://localhost:3000    (Express server)
```

The frontend makes HTTP requests to the backend using JavaScript's `fetch` API:

```javascript
// Frontend code (runs in the browser)
const response = await fetch('http://localhost:3000/api/translate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Hello, World!',
    languages: ['fr', 'es', 'ja']
  })
});

const data = await response.json();  // parse the JSON response
console.log(data.translations);       // [{language: "fr", translation: "Bonjour..."}]
```

That's it. The frontend sends an HTTP request containing JSON. The backend receives it, does its work (in our case, calls Google Translate), and sends back a JSON response. It's just text over TCP, formatted as HTTP, carrying JSON payloads.

### CORS — The Thing That Will Block You

When your frontend at `localhost:5173` tries to call your backend at `localhost:3000`, the browser will block it by default. This is **CORS** (Cross-Origin Resource Sharing) — a security policy that prevents a website from making requests to a different domain/port.

You fix this by telling your backend to explicitly allow requests from your frontend's origin:

```javascript
// Backend: "yes, localhost:5173 is allowed to talk to me"
app.use(cors({ origin: 'http://localhost:5173' }));
```

Every new full-stack developer hits this wall. When you see a CORS error in your browser console, it's not a bug — it's a security feature you need to configure.

### But Why Do I Need a Backend At All?

Fair question. If the frontend is a program that runs in the browser, and APIs are just HTTP requests, why can't the frontend call the OpenAI API directly? Why bother with a backend?

Three reasons:

**1. Secrets.** The frontend is public. Anyone can right-click your webpage, open DevTools, and read every line of JavaScript. If your OpenAI API key is in that JavaScript, it's no longer a secret. The backend is the only place you can keep a secret, because your server code never reaches the user's browser.

**2. Trust.** Suppose you're building a translation app with a rate limit: each user gets 10 translations per day (so you don't go bankrupt on API costs). If that logic is in the frontend, any user can open DevTools and remove the limit. The backend is the only place you can enforce rules that users can't bypass.

**3. Orchestration.** Many features require coordinating multiple things: check the user is logged in, look up their usage in a database, call an external API, save the result, return it. The backend is where you compose these steps.

Here's a concrete example — the app we're going to build in this document: a **Hello World translator**. You click a button and it translates "Hello, World!" into French, Spanish, Japanese, Arabic, Swahili, Chinese, Hindi, and German, all at once.

```
What the user sees:
┌──────────────────────────────────────┐
│  Hello World Translator              │
│                                      │
│  [Translate Hello World!]            │
│                                      │
│  • fr: Bonjour le monde !            │
│  • es: ¡Hola, mundo!                 │
│  • ja: こんにちは世界！                │
│  • ar: !مرحبا بالعالم                │
│  • sw: Salamu, Dunia!                │
│  • zh: 你好世界！                     │
│  • hi: नमस्ते दुनिया!                  │
│  • de: Hallo Welt!                   │
└──────────────────────────────────────┘

What actually happens:

  Browser                     Your Backend                  Google API
 ┌──────────┐               ┌──────────────┐            ┌──────────────┐
 │          │  POST         │              │  8 API     │              │
 │  React   │─/api/translate→  Express     │──calls───→ │  Google      │
 │  app     │  {text,langs} │  server      │  (with     │  Translate   │
 │          │               │              │  API key   │              │
 │          │←─{results}────│  holds key   │  from      │              │
 └──────────┘               │  in .env     │  .env)     │              │
                            │              │←─results──│              │
                            └──────────────┘            └──────────────┘
```

The frontend has no API key. It doesn't know how to talk to Google. It just sends a request to *your* backend saying "translate this into these languages." Your backend holds the key, calls Google eight times, collects the results, and sends them all back. The user never sees or touches the API key.

Without the backend, this is just a static HTML page with a button that does nothing. *With* the backend, it's a real application that talks to the outside world.

---

## Part 4: The JavaScript Ecosystem

Before we look at the code for our translator, you need to understand the tools involved. There are a lot of names and it's easy to get lost, so here's the map:

```
┌─────────────────────────────────────────────────────────┐
│  Node.js     — runs JavaScript outside the browser      │
│  npm         — installs libraries (like pip for Python)  │
│                                                         │
│  ┌─── Backend ───────────┐  ┌─── Frontend ────────────┐ │
│  │  Express              │  │  React (builds the UI)   │ │
│  │  (handles HTTP        │  │  Vite  (dev server +     │ │
│  │   requests)           │  │         build tool)      │ │
│  └───────────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

Node.js and npm are the foundation. Express is a backend library. React and Vite are frontend tools. They're not alternatives to each other — they're layers in a stack.

### Node.js — JavaScript Outside the Browser

JavaScript was originally trapped inside web browsers. That was its entire world. In 2009, Node.js took Chrome's JavaScript engine (called V8) and wrapped it in a standalone program you can run from the terminal, just like you run `python3 script.py`:

```bash
# Python:
python3 my_script.py

# JavaScript with Node.js:
node my_script.js
```

That's what Node.js is: a program called `node` that executes JavaScript files. When you run `node server.js` to start your backend, you're using Node.js. When Vite builds your frontend, it's running on Node.js under the hood.

This is why JavaScript dominates full-stack development: you can use one language for both frontend and backend. You don't *have* to — plenty of backends are Python, Go, Java, etc. — but the option is there, and the ecosystem is enormous.

### npm — The Package Manager

npm (Node Package Manager) is to JavaScript what pip is to Python. It downloads and installs third-party libraries ("packages") from a central registry.

```bash
npm init              # creates package.json (like requirements.txt)
npm install express   # downloads express and adds it to package.json
npm install           # installs everything listed in package.json
```

**Why do you need this?** Because you don't want to write an HTTP server from scratch, or a JSON parser, or a CORS handler. Other people have written these things. npm lets you pull them in with one command.

`package.json` is the file that defines your project. It lists your dependencies (what libraries you need) and scripts (shortcut commands):

```json
{
  "name": "hello-world-translator-backend",
  "type": "module",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5"
  }
}
```

Key things here:

- **`"type": "module"`** enables modern `import` syntax (like Python's `import`). Without this, Node.js uses an older system called `require()`. You'll see both in the wild, but `import` is the modern standard.
- **`"scripts"`** defines shortcut commands. `npm start` runs `node server.js`. You could also define `"dev"`, `"build"`, `"test"`, etc. `npm run dev` runs whatever is under `"dev"`.
- **`"dependencies"`** lists the libraries your project needs. The `^4.18.2` means "version 4.18.2 or any compatible newer version."

**`node_modules/`** is the directory where npm puts the actual downloaded code. It's often enormous (hundreds of megabytes — this is a running joke in the JavaScript community). It goes in `.gitignore` — never commit it. Anyone who clones your project runs `npm install` to recreate it from `package.json`.

**`package-lock.json`** records the *exact* version of every package installed (including dependencies of dependencies). This ensures that `npm install` produces identical results on any machine. Commit this file — it's not like `node_modules`.

### async/await — How JavaScript Handles Waiting

Before we look at the Express and React code, one JavaScript concept you need: **async/await**.

When your backend calls the Google Translate API, it takes time — maybe 200 milliseconds. JavaScript is single-threaded, so it can't just freeze and wait (that would block every other request). Instead, it uses **asynchronous** programming: start the request, go do other things, come back when the response arrives.

```javascript
// This function is "async" — it can use "await" to pause for slow operations
async function translateWord(text, lang) {
  // "await" means: send this request, then pause THIS function until
  // the response comes back. Other code can run in the meantime.
  const response = await fetch('https://translation.googleapis.com/...');

  // This also waits — parsing the response body takes time too
  const data = await response.json();

  return data.translations[0].translatedText;
}
```

If you know Python, `async/await` works almost identically to Python's `asyncio`. The `await` keyword means "pause here until this thing finishes." The `async` keyword marks a function as one that can use `await`.

The key insight: **every HTTP request in your code — whether the frontend calling your backend, or your backend calling Google — uses `await`**. Network calls are slow. You always wait for the response before continuing.

### Express — The Backend Framework

**Express** is a minimal backend framework for Node.js. It handles incoming HTTP requests and lets you define what to do with them. It's the most popular Node.js backend framework and has been since roughly 2012.

Here's the backend for our Hello World translator. Its entire job is to receive a request from the browser, call the Google Translate API (using the secret key from `.env`), and return the result:

```javascript
// server.js
import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());            // allow the frontend to talk to us
app.use(express.json());    // parse JSON request bodies

// POST /api/translate
// The frontend sends: { "text": "Hello, World!", "languages": ["fr", "es", "ja"] }
// We call Google Translate for each language and send back the results.
app.post('/api/translate', async (req, res) => {
  const { text, languages } = req.body;

  // The API key lives in .env — the frontend never sees it
  const apiKey = process.env.GOOGLE_TRANSLATE_API_KEY;

  // For each language, call the Google Translate API
  // (This is our backend making its OWN HTTP request to Google's server)
  const translations = [];
  for (const lang of languages) {
    const googleResponse = await fetch(
      `https://translation.googleapis.com/language/translate/v2?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: text, target: lang })
      }
    );
    const data = await googleResponse.json();
    translations.push({
      language: lang,
      translation: data.data.translations[0].translatedText
    });
  }

  // Send the results back to the frontend as JSON
  res.json({ original: text, translations });
});

// Start listening on port 3000 (or whatever port Cloud Run assigns)
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
```

Let's unpack what's happening here:

- `app.post('/api/translate', ...)` registers a **route handler**: when an HTTP POST request arrives at the path `/api/translate`, this function runs.
- The function receives `req` (the incoming request — including the JSON body the frontend sent) and `res` (the response object we use to send data back).
- Inside the handler, our server makes *its own* HTTP requests to Google's API. This is the key insight: **the backend is itself a client**. It receives a request from the browser, then turns around and makes a request to Google. The browser never talks to Google directly.
- `process.env.GOOGLE_TRANSLATE_API_KEY` reads the API key from the environment (loaded from `.env` at startup). This line is why the backend exists.

Run it with `node server.js`. It listens on port 3000 and waits.

### React — The Frontend Framework

**React** (by Meta/Facebook) is a JavaScript library for building user interfaces. Instead of manually manipulating HTML elements with `document.getElementById()` and friends, you write **components** — self-contained pieces of UI that manage their own state.

Here's the frontend for our translator. It's a button that says "Translate Hello World", and when you click it, it calls our backend and displays the results:

```jsx
// App.jsx
import { useState } from 'react';

function App() {
  const [translations, setTranslations] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleClick() {
    setLoading(true);

    // POST to our backend — NOT to Google directly!
    const response = await fetch('http://localhost:3000/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: 'Hello, World!',
        languages: ['fr', 'es', 'ja', 'ar', 'sw', 'zh', 'hi', 'de']
      })
    });

    const data = await response.json();   // parse JSON response
    setTranslations(data.translations);   // store in state → triggers re-render
    setLoading(false);
  }

  return (
    <div>
      <h1>Hello World Translator</h1>
      <button onClick={handleClick} disabled={loading}>
        {loading ? 'Translating...' : 'Translate Hello World!'}
      </button>

      {translations && (
        <ul>
          {translations.map(t => (
            <li key={t.language}>
              <strong>{t.language}:</strong> {t.translation}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

Key concepts:

- **JSX**: the HTML-like syntax inside JavaScript (`<div>`, `<ul>`, `<li>`, etc.). It looks like HTML but it's actually JavaScript that creates React elements. Gets compiled to normal JS before running in the browser.
- **Components**: functions that return JSX. `App` is a component. In a larger app, you'd break it into smaller components — a `TranslateButton`, a `TranslationList`, etc.
- **State** (`useState`): data that, when changed, causes the component to re-render automatically. When `setTranslations(data)` is called, React re-draws the UI to show the new list. You don't manually update the DOM — you update the state and React figures out what changed.
- **Event handlers**: `onClick={handleClick}` means "when this button is clicked, call `handleClick`." That function calls `fetch()` to talk to our backend.
- **Props**: inputs to a component, passed like HTML attributes: `<TranslationItem language="fr" text="Bonjour" />`. This is how parent components pass data to child components.

The frontend knows *nothing* about Google Translate. It doesn't have an API key. It just sends a request to `/api/translate` on our backend and displays whatever comes back.

React doesn't give you a full framework. It's just the UI layer. You'll often see it paired with:
- **Vite**: a build tool and dev server (the thing running on port 5173)
- **React Router**: for handling URL-based navigation
- **Tailwind CSS** or similar: for styling

### Vite — The Build Tool

Browsers can't directly run React's JSX syntax, TypeScript, or modern JavaScript features. **Vite** (French for "fast") is a build tool that:

1. In development: runs a dev server that transforms your code on-the-fly as the browser requests it. Hot Module Replacement (HMR) means changes appear instantly without a full page reload.
2. For production: bundles all your JavaScript, CSS, and assets into optimised files for deployment.

```bash
npm create vite@latest my-app -- --template react   # scaffold a new project
cd my-app
npm install
npm run dev      # start dev server on localhost:5173
```

You edit your React components, save the file, and the changes appear in the browser within milliseconds. This is the development experience.

---

## Part 5: Putting It All Together

### A Complete Full-Stack Request

Here's exactly what happens when the user clicks "Translate Hello World!":

```
1. User clicks the button
   │
2. React calls handleClick(), which calls:
   │  fetch('http://localhost:3000/api/translate', {
   │    method: 'POST',
   │    body: JSON.stringify({
   │      text: "Hello, World!",
   │      languages: ["fr", "es", "ja", "ar", "sw", "zh", "hi", "de"]
   │    })
   │  })
   │
3. The browser turns this into a raw HTTP request and sends it:
   │  POST /api/translate HTTP/1.1
   │  Host: localhost:3000
   │  Content-Type: application/json
   │
   │  {"text":"Hello, World!","languages":["fr","es","ja","ar","sw","zh","hi","de"]}
   │
4. Express receives it on port 3000
   │  → app.post('/api/translate', handler) matches the path
   │  → handler reads GOOGLE_TRANSLATE_API_KEY from process.env
   │  → handler calls Google's API 8 times (one per language)
   │     — each of these is ANOTHER HTTP request, from our server to Google's server
   │  → Google responds 8 times with translations
   │
5. Express sends the collected results back:
   │  HTTP/1.1 200 OK
   │  Content-Type: application/json
   │
   │  {"original":"Hello, World!","translations":[
   │    {"language":"fr","translation":"Bonjour le monde !"},
   │    {"language":"es","translation":"¡Hola, mundo!"},
   │    {"language":"ja","translation":"こんにちは世界！"},
   │    ...
   │  ]}
   │
6. The browser receives the response
   │  → fetch() resolves
   │  → response.json() parses the JSON into a JavaScript object
   │
7. React updates state:
   │  setTranslations(data.translations)
   │
8. React re-renders — the list of translations appears on screen
```

No magic. Just text over TCP, all the way down.

Notice that there are *two layers* of HTTP requests here. The browser makes one request to your backend. Your backend then makes eight requests to Google. The browser has no idea Google is involved — it just asked your server for translations and got them back. Your backend is a **middleman** that holds the secret key.

### Project Structure

Our Hello World translator project looks like this:

```
hello-world-translator/
├── frontend/
│   ├── src/
│   │   ├── App.jsx        the translate button + results list
│   │   └── main.jsx       entry point (renders App into the page)
│   ├── package.json       frontend dependencies (react, vite)
│   ├── vite.config.js     Vite configuration
│   └── index.html         the single HTML file React mounts into
│
├── backend/
│   ├── server.js          Express app (the /api/translate endpoint)
│   ├── package.json       backend dependencies (express, cors)
│   └── .env               GOOGLE_TRANSLATE_API_KEY=... (NEVER commit this)
│
├── .gitignore             must include: node_modules/, .env
└── README.md
```

Two directories, two programs, two `package.json` files. The frontend is served by Vite on port 5173. The backend is run by Node on port 3000. They talk over HTTP.

---

## Part 6: Things You'll Need Soon

### Databases

Most real apps need to store data permanently — user accounts, posts, settings. This means a **database**. The main options are PostgreSQL (relational, the default choice), SQLite (a database in a single file), and MongoDB (stores JSON-like documents). This is a big enough subject that we'll cover it separately. For now, just know that the backend is the only part of your app that talks to the database — the frontend never connects directly.

### Authentication

How do you know who's making a request? Common patterns:

- **Session-based**: server stores login state; sends a cookie to the browser; browser sends cookie with every request.
- **Token-based (JWT)**: server generates a signed token on login; frontend stores it and sends it in the `Authorization` header. The server verifies the signature without needing to store anything.
- **OAuth**: "Sign in with Google/GitHub." Delegates authentication to a trusted third party.

### Middleware

In Express, middleware is a function that runs *before* your route handler. It's how you add cross-cutting concerns:

```javascript
// This function runs on EVERY request before the route handler
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`);  // logging
  next();  // pass control to the next middleware or route
});

// Or apply to specific routes
app.use('/api/admin', requireAuth);  // only authenticated users
```

Common middleware: logging, authentication checks, CORS headers, rate limiting, request body parsing.

### TypeScript

TypeScript is JavaScript with static types. It catches bugs at compile time that JavaScript would only catch at runtime (or never). Most professional codebases use it.

```typescript
// JavaScript: hope for the best
function add(a, b) { return a + b; }
add("hello", 5);  // "hello5" — no error, just wrong

// TypeScript: catch it at compile time
function add(a: number, b: number): number { return a + b; }
add("hello", 5);  // ERROR: Argument of type 'string' is not assignable
```

You can adopt it gradually. `.js` files become `.ts` (or `.tsx` for React). Vite and most modern tools support it out of the box.

### Version Control Conventions

You know Git. But you might not know the workflow:

- **Feature branches**: never commit directly to `main`. Create a branch (`feat/add-login`), work there, then open a pull request.
- **Pull requests (PRs)**: propose merging your branch into `main`. Others review the code before it's merged.
- **CI/CD** (Continuous Integration/Deployment): automated tests run on every PR. If tests pass, the code can be automatically deployed.

### Testing

- **Unit tests**: test individual functions. (Jest, Vitest, pytest)
- **Integration tests**: test that components work together. (Supertest for APIs)
- **End-to-end (E2E) tests**: simulate a real user clicking through the app. (Playwright, Cypress)

---

## Part 7: Deployment — Getting Your App on the Internet

### What Does "Deploy" Actually Mean?

When you run `npm run dev`, your app lives on `localhost` — your own machine. Nobody else can reach it. **Deploying** means putting your code on someone else's computer (a server in a data centre) that's connected to the internet 24/7 and has a public URL.

The two halves of your app deploy differently because they *are* different:

- **The frontend** is just files. After you run `npm run build`, Vite produces a folder of static HTML, CSS, and JavaScript. These don't need a running process — any web server can hand them to browsers. This is called **static hosting**.

- **The backend** is a running program. Your Express server needs to be actively executing, listening on a port, waiting for requests. This needs **server hosting** — a machine (real or virtual) that runs your code.

### The Google Cloud Approach

A beginner-friendly combination:

**Firebase Hosting** serves your frontend. It's a static file host with a global CDN (Content Delivery Network — copies of your files cached in data centres worldwide so they load fast for everyone). It's run by Google and has a generous free tier.

**Cloud Run** runs your backend. You give it your code, it packages it into a **container** (an isolated, reproducible environment — like a lightweight virtual machine), and runs it on Google's infrastructure. It gives you an HTTPS URL, handles scaling, and shuts down when idle so you pay nothing during quiet periods.

```
   Browser                          Google Cloud
 ┌──────────┐                    ┌─────────────────────────────────────┐
 │  User at  │   HTTPS           │                                     │
 │  home     │ ──────────────→   │  Firebase Hosting                   │
 │           │ ←────────────── │  (serves your static React files)    │
 │           │                   │  your-app-name.web.app              │
 │           │                   │                                     │
 │  React    │   HTTPS           │  Cloud Run                          │
 │  in the   │ ──────────────→   │  (runs your Express server)         │
 │  browser  │ ←────────────── │  your-backend-abc123.a.run.app      │
 └──────────┘                    └─────────────────────────────────────┘
```

### Setting Up: Accounts and Tools

Before you can deploy anything, you need to create accounts and install command-line tools. This is the tedious-but-necessary bureaucracy:

1. **Google Cloud account**: go to cloud.google.com and sign up. You'll need a Google account. **Google will ask for a credit card** — this is normal. They need it to bill you if you exceed the free tier. New accounts get $300 in free credits that last 90 days. For a hello-world app, you won't spend any of it.

2. **Create a project**: Google Cloud organises everything into "projects." One project = one app. All your resources (hosting, backend, databases) live inside a project. You create one from the Google Cloud Console (the web dashboard) or the command line. You pick a globally unique project ID — something like `my-cool-app-12345`.

3. **Install command-line tools**: you'll need the `gcloud` CLI (for Cloud Run) and the `firebase` CLI (for Firebase Hosting). An AI coding assistant like Claude Code can walk you through the exact install commands for your operating system.

4. **Log in from the terminal**: both `gcloud` and `firebase` have login commands that open your browser and ask you to authenticate. After that, your terminal can deploy to your Google Cloud account.

### How Deployment Actually Works

**Deploying the backend to Cloud Run:**

You point the `gcloud` CLI at your backend directory and tell it to deploy. Cloud Run looks at your code, figures out it's a Node.js app (from `package.json`), builds a container image in the cloud, and starts running it. The whole process takes 1–2 minutes. When it's done, it gives you a URL like `https://your-backend-abc123-uc.a.run.app`. That URL is your backend. It's live. Anyone in the world can hit it.

You don't need to set up a server, install an operating system, configure Nginx, open firewall ports, or obtain an SSL certificate. Cloud Run handles all of that.

**Deploying the frontend to Firebase:**

You run `npm run build` to produce the static files, then use the `firebase` CLI to upload them. Firebase gives you a URL like `https://your-app-name.web.app`. Done.

The one thing you need to wire up: tell your frontend code the URL of your Cloud Run backend. During development it was `localhost:3000`; in production it's that `a.run.app` URL. This is typically done with an environment variable that Vite bakes into the built files.

### What Does Hosting Cost?

For a personal project or learning exercise: **nothing**.

| Service | Free Tier |
|---------|-----------|
| Firebase Hosting | 10 GB storage, 360 MB/day transfer |
| Cloud Run | 2 million requests/month, 360,000 GB-seconds of compute |

Cloud Run **scales to zero** — when nobody is using your app, it shuts down completely and you aren't billed. When someone visits again, it cold-starts in a couple of seconds.

You'd need hundreds of thousands of users before hitting a real bill. But it's still good practice to **set a billing alert** in the Google Cloud Console (e.g., "email me if spending exceeds $5/month") so you're never surprised.

### Docker — What Is It and Do I Need It?

You'll hear the word "Docker" a lot. Here's the concept:

Your app works on your machine. But your machine has a specific operating system, specific versions of Node.js, specific libraries installed. If you copy your code to a different machine, it might break.

**Docker** solves this by letting you define, in a file called a `Dockerfile`, the *exact* environment your app needs: which operating system, which version of Node.js, which system libraries. Docker builds this into a **container image** — a frozen snapshot of a complete, isolated environment. You can run that image on any machine and it will behave identically.

Cloud Run runs containers. When you deploy with `gcloud run deploy --source .`, Cloud Run *automatically* builds a container for you behind the scenes — you don't need to write a Dockerfile or install Docker yourself. But for larger projects, you'll eventually want to write your own Dockerfile for more control.

For now: know that Docker exists, know that it solves the "works on my machine" problem, and know that Cloud Run uses it under the hood.

### Other Deployment Options

Firebase + Cloud Run isn't the only path. The landscape:

| Service | What it hosts | Good for | Cost |
|---------|--------------|----------|------|
| **Vercel** | Frontend + serverless functions | React/Next.js apps | Generous free tier |
| **Netlify** | Frontend + serverless functions | Static sites, Jamstack | Generous free tier |
| **Railway** | Full backend servers | Any backend | $5/month hobby plan |
| **Render** | Full backend servers | Any backend | Free tier available |
| **Fly.io** | Full backend servers | Low-latency global apps | Free tier available |
| **AWS / Azure** | Everything | Enterprise scale | Complex pricing, overkill for beginners |

Vercel is particularly popular for React apps and is arguably even easier than Firebase for frontend-only deployments. But Firebase + Cloud Run is a solid combo when you need a proper backend server.

---

## Part 8: The Mental Model

If you take one thing from this document, make it this:

**A web application is a frontend program that makes HTTP requests to a backend program. Both programs speak JSON. The backend holds the secrets and does the things the frontend can't be trusted to do — calling paid APIs, enforcing rules, storing data. Everything else is details.**

```
                                                           External APIs
                                                        ┌──────────────────┐
                                                  HTTP  │ Google Translate  │
                                               ┌──────→ │ OpenAI / Claude   │
   Browser                    Your Server      │  JSON  │ Stripe / Twilio   │
 ┌──────────┐  HTTP + JSON  ┌───────────┐     │  +key  │ ...anything with  │
 │  React   │ ────────────→ │  Express  │─────┘        │    an API key     │
 │  (JS)    │ ←──────────── │  (Node)   │              └──────────────────┘
 └──────────┘               │           │
    Port 5173               │  .env has │                  Database
                            │  all the  │     SQL       ┌──────────────────┐
                            │  secrets  │─────────────→ │ PostgreSQL       │
                            └───────────┘  (direct TCP  │ (stores data     │
                              Port 3000     connection)  │  permanently)    │
                                                        └──────────────────┘
                                                           Port 5432
```

The frontend is a program that paints pixels and sends requests. The backend is a program that accepts those requests and does the privileged work. It talks to two kinds of external services:

- **Web APIs** (Google Translate, OpenAI, etc.) — over HTTP, using JSON, authenticated with an API key
- **Databases** (PostgreSQL, etc.) — over a direct TCP connection, using SQL queries, authenticated with a username/password

Both sets of credentials live in `.env`. The frontend never sees either. Ports keep everything from colliding on the same machine.

That's the whole thing. Everything else — React, Express, REST, CORS, JWT, Docker, Kubernetes — is just detail layered on top of this basic architecture.

Now go build something.
