# Software Engineering Principles

*Everything between "it works on my machine" and "it works in production, at scale, maintained by a team."*

You can write code. You've shipped side projects, maybe even something with users. But there's a gap between writing a script that works and building software that survives -- survives other developers reading it, survives requirements changing, survives 3am on-call pages, survives you leaving and someone else taking over.

That gap is software engineering. It's not about knowing more languages or frameworks. It's about understanding *why* professional codebases look the way they do -- why there are so many files, so many tests, so many rules that feel like overhead until the day they save you.

Three sittings. One coffee each. By the end, you'll understand the principles that every senior engineer has internalized but rarely explains from scratch.

---

## Sitting 1: Complexity and How to Fight It

### Software Engineering Is Not Programming

Programming is the act of writing code that makes a computer do something. Software engineering is the act of writing code that a *team* can maintain and evolve *over years*.

The distinction matters because the hard part of software isn't getting the computer to do the right thing once. The hard part is:

- What happens when requirements change next quarter?
- What happens when someone who didn't write this code needs to fix a bug in it?
- What happens when this needs to handle 100x more users?
- What happens when you need to change one part without breaking five others?

A 200-line script doesn't need software engineering. A system with 50 files, 3 developers, and paying customers does. The principles in this document are answers to the question: *how do you keep a growing codebase from collapsing under its own weight?*

### Complexity: The Only Real Enemy

If there's one thing this entire document is about, it's this: **complexity is the single greatest threat to software systems.** Not bugs. Not performance. Not hackers. Complexity.

Fred Brooks, in his 1986 essay "No Silver Bullet," drew a distinction that still holds:

```
┌──────────────────────────────────────────────────────────┐
│                    Complexity                              │
│                                                           │
│   Essential                    Accidental                 │
│   ──────────                   ──────────                 │
│   The problem is               We made this hard          │
│   genuinely hard.              for no good reason.        │
│                                                           │
│   A tax system has             Our config is split        │
│   complex rules because        across 4 files because     │
│   tax law is complex.          "that's how it evolved."   │
│                                                           │
│   Pathfinding in a             The deploy takes 45 min    │
│   graph is O(V + E).           because nobody optimized   │
│   That's just math.            the Docker build.          │
│                                                           │
│   You can't eliminate           You can and should         │
│   essential complexity.         eliminate accidental       │
│                                 complexity.                │
└──────────────────────────────────────────────────────────┘
```

Most of the principles in this document are strategies for minimizing accidental complexity. The essential complexity of your problem is irreducible -- but you'd be amazed how much of the pain in real codebases is accidental.

Signs you're drowning in accidental complexity:
- Changing one feature requires touching 8 files
- New developers take weeks to become productive
- Nobody is confident that a change won't break something unexpected
- There are parts of the codebase that "nobody touches because it works"
- The test suite takes 40 minutes and still misses real bugs

### Abstraction: The One Idea

Every principle in this document is, at some level, a form of **abstraction** -- hiding complexity behind a simpler interface.

When you call `requests.get("https://api.example.com/data")`, you don't think about TCP handshakes, DNS resolution, TLS certificates, HTTP headers, socket buffers, or packet retransmission. The `requests` library *abstracts* all of that behind a single function call.

This is the most powerful idea in all of computing:

```
What you see                         What's actually happening
─────────                            ──────────────────────
requests.get(url)                    DNS lookup → IP resolution
                                     TCP 3-way handshake
                                     TLS negotiation (if HTTPS)
                                     HTTP request serialization
                                     Socket write
                                     Wait for response
                                     HTTP response parsing
                                     Decompression (if gzipped)
                                     Connection pooling / keep-alive
                                     Retry logic
                                     Timeout handling
```

Abstraction lets you think at the level of your problem ("fetch this data") instead of at the level of the implementation ("negotiate a TLS session with cipher suite selection"). Without it, every program would be millions of lines long and nobody could understand anything.

But abstractions aren't free. Joel Spolsky called this the **Law of Leaky Abstractions**: all non-trivial abstractions, to some degree, are leaky. The TCP connection *does* sometimes drop. The database query *does* sometimes time out. The garbage collector *does* sometimes pause your program. When the abstraction leaks, you need to understand the layer beneath.

This is why the best engineers aren't just good at using abstractions -- they're good at knowing when and how abstractions fail. And that's why the CS foundations matter even when you're working at a high level.

### Modularity: Divide and Conquer

Abstraction is the idea. **Modularity** is how you apply it to your codebase.

A module is any unit of code with a defined boundary -- a function, a class, a file, a package, a service. The point of modular design is simple: **you should be able to understand, change, and test one piece without understanding the whole system.**

Consider a monolithic 5,000-line file versus five well-structured 1,000-line modules:

```
Monolith (5,000 lines)               Modular (5 × 1,000 lines)
──────────────────────                ───────────────────────────

┌────────────────────┐                ┌──────┐  ┌──────┐  ┌──────┐
│                    │                │ Auth │  │ Users│  │ Posts│
│  Everything is     │                │      │──│      │──│      │
│  connected to      │                └──────┘  └──────┘  └──────┘
│  everything.       │                    │                    │
│                    │                ┌──────┐            ┌──────┐
│  Change one line,  │                │  DB  │            │Search│
│  break anything.   │                │      │            │      │
│                    │                └──────┘            └──────┘
└────────────────────┘
                                      Change Auth → only Auth
Change anything → test everything     changes. Test Auth alone.
```

The modular version has the same total code. But each piece can be understood in isolation. The connections between modules are explicit and limited. When something breaks in search, you look at the Search module and its connections -- not the entire codebase.

### Coupling and Cohesion

Two modules interact. How tightly they're connected is called **coupling**. How well the code inside each module relates to a single purpose is called **cohesion**.

**You want low coupling and high cohesion.** That's the whole rule.

```
                    Cohesion
                    (within a module)

      Low                              High
  ┌──────────────┐              ┌──────────────┐
  │ send_email() │              │ send_email() │
  │ parse_csv()  │              │ format_body()│
  │ resize_img() │              │ add_headers()│
  │ calc_tax()   │              │ check_spam() │
  └──────────────┘              └──────────────┘
  "utils.py" — the junk drawer     "email.py" — one clear job


                    Coupling
                    (between modules)

      Tight                            Loose
  ┌───────┐   ┌───────┐        ┌───────┐   ┌───────┐
  │ Auth  │──→│ Users │        │ Auth  │   │ Users │
  │       │←──│       │        │       │──→│       │
  │ knows │   │ knows │        │ calls │   │ exposes│
  │ User  │   │ Auth  │        │ one   │   │ one   │
  │ fields│   │ logic │        │ method│   │ method│
  └───────┘   └───────┘        └───────┘   └───────┘
  Both know each other's          Auth knows Users' interface,
  internals. Change one,          not its internals.
  break both.                     Change internals freely.
```

**Tight coupling** means module A knows the internal details of module B. If B changes its database schema, A breaks. If B renames a private method, A breaks. They can't be deployed, tested, or understood independently.

**Loose coupling** means module A only knows B's public interface -- the contract B promises to uphold. B can rewrite its internals completely, and as long as the interface holds, A never notices.

**High cohesion** means everything in a module is related to a single responsibility. Low cohesion means a module is a grab-bag of unrelated functions (we've all written a `utils.py` that became a dumping ground).

Here's the practical test: if you need to change how emails are sent, how many files do you need to open? If the answer is one, you have good cohesion. If the answer is seven, your email logic is scattered.

### Information Hiding

David Parnas wrote the seminal paper on this in 1972: **each module should hide a design decision behind its interface.** If you later change that decision, only the module changes -- the rest of the system is unaffected.

```
Module: DatabaseStore
──────────────────────
  Public interface:
    save(key, value)
    load(key) → value
    delete(key)

  Hidden decision: are we using PostgreSQL, Redis, or a flat file?

  → The rest of the system doesn't know and doesn't care.
  → You can swap PostgreSQL for Redis without changing any caller.
```

This is why experienced engineers are almost allergic to leaking internal details. Every piece of internal structure that leaks through the interface is a piece you can never change without breaking callers.

Real example: if your API returns database row IDs as sequential integers, clients will start assuming they're sequential (sorting by ID, checking if one record is "newer" by comparing IDs). You've leaked an internal database detail, and now you can never switch to UUIDs without breaking clients. Hide the decision.

---

## Sitting 2: The Discipline

### The SOLID Principles

These five principles (Robert C. Martin, early 2000s) come up in every software engineering interview and textbook. Some are more useful than others. Here's what each one actually means in practice:

**S — Single Responsibility Principle**
A class/module should have one reason to change.

This is the most important one. Not "a class should do one thing" (too vague) -- a class should have *one stakeholder* whose requirements could cause it to change. If your `UserService` class handles authentication, profile rendering, *and* billing, then a change to billing logic, auth rules, or the profile UI all force you to modify the same file. Three reasons to change. Three opportunities to accidentally break the other two.

**O — Open/Closed Principle**
Software should be open for extension, closed for modification.

You should be able to add new behavior without changing existing code. In practice, this means using interfaces and polymorphism:

```python
# Closed for modification — you never edit this function:
def process_payment(processor: PaymentProcessor, amount: float):
    processor.charge(amount)

# Open for extension — add new processors without changing existing code:
class StripeProcessor(PaymentProcessor):
    def charge(self, amount): ...

class PayPalProcessor(PaymentProcessor):
    def charge(self, amount): ...
```

**L — Liskov Substitution Principle**
If S is a subtype of T, you should be able to use S anywhere you use T without breaking things.

The classic violation: a `Square` class that extends `Rectangle`. If you set a rectangle's width, the height shouldn't change. But for a square, setting the width *must* change the height. Code that expects `Rectangle` behavior breaks when handed a `Square`. The inheritance hierarchy lied.

**I — Interface Segregation Principle**
Don't force clients to depend on methods they don't use.

If your `Animal` interface requires `fly()`, `swim()`, and `run()`, then `Dog` has to implement `fly()` (and presumably throw an error). Better: separate `Flyable`, `Swimmable`, and `Runnable` interfaces. Each client depends only on what it needs.

**D — Dependency Inversion Principle**
High-level modules shouldn't depend on low-level modules. Both should depend on abstractions.

Your business logic shouldn't import `psycopg2` directly. It should depend on a `Database` interface. The PostgreSQL implementation is injected at startup. This is why "dependency injection" is such a big deal in enterprise code -- it's the mechanical implementation of this principle.

### DRY, KISS, YAGNI

Three acronyms that every engineer knows and sometimes misapplies:

**DRY — Don't Repeat Yourself.** If the same logic exists in two places, it will eventually diverge. Extract it into a single source of truth. But be careful: two pieces of code that *look* the same aren't always *conceptually* the same. Billing tax calculation and shipping tax calculation might share logic today but diverge tomorrow. Premature DRY creates coupling between things that should be independent.

**KISS — Keep It Simple, Stupid.** The simplest solution that works is usually the best one. This is harder than it sounds because engineers love elegant abstractions. The discipline is in resisting abstraction until it's earned.

**YAGNI — You Aren't Gonna Need It.** Don't build features or abstractions for hypothetical future requirements. You think you'll need a plugin system someday? You won't. Build for what you need today. If you're wrong, refactoring is cheaper than maintaining unused abstraction.

When these conflict (and they will), YAGNI usually wins. The cost of building something you don't need is paid immediately and forever. The cost of not having something you later need is paid once, when you build it.

### Testing: Why, What, and How Much

You test software for one reason: **confidence to change things.** A comprehensive test suite means you can refactor, add features, and fix bugs without fear. Without tests, every change is a gamble.

The **testing pyramid** describes how many of each type you should have:

```
                    ╱╲
                   ╱  ╲
                  ╱ E2E╲         Few: slow, expensive, brittle
                 ╱──────╲        Tests the whole system end-to-end
                ╱        ╲       "Click the button, check the page"
               ╱Integration╲    Some: medium speed, medium cost
              ╱─────────────╲   Tests modules working together
             ╱               ╲  "API call returns correct JSON"
            ╱   Unit Tests    ╲  Many: fast, cheap, focused
           ╱───────────────────╲ Tests one function/class in isolation
          ╱                     ╲ "This function returns 7 for input [3,4]"
         ╱───────────────────────╲
```

**Unit tests** test a single function or class, mocking out dependencies. They run in milliseconds and you should have hundreds. They catch logic errors immediately.

**Integration tests** test how modules work together -- your API handler calling a real database, your payment module talking to a (sandbox) Stripe API. They're slower but catch the bugs that unit tests miss: miscommunications between modules.

**End-to-end (E2E) tests** test the entire system from the user's perspective -- spinning up a browser, clicking buttons, verifying page content. They're slow, flaky, and expensive. Have a few for critical paths (login, checkout, the main happy path). Don't try to cover everything with E2E tests.

**What not to test:** Trivial getters/setters, third-party library internals, implementation details (test behavior, not how it's implemented). If a test breaks every time you refactor but the behavior hasn't changed, the test is testing the wrong thing.

**Test-Driven Development (TDD):** Write the test first, watch it fail, write the minimum code to pass it, refactor. It sounds backwards but it forces you to think about the interface before the implementation. Valuable for well-defined problems. Less valuable for exploratory work where you don't yet know what you're building.

### Code Review: The Most Underrated Practice

Code review -- having another person read your code before it's merged -- is probably the highest-value practice in professional software engineering. It catches bugs, yes, but that's not the main benefit. The main benefits are:

1. **Knowledge sharing.** After review, at least two people understand every change. When one person goes on vacation, the team doesn't stop.
2. **Consistency.** The codebase stays cohesive instead of developing five different styles.
3. **Design feedback.** A reviewer sees the change with fresh eyes and often spots better approaches.
4. **Accountability.** Knowing someone will read your code makes you write it more carefully.

Good code review is not about gatekeeping or proving you're smart. It's a conversation. The goal is to make the code better, not to make the author feel bad.

### CI/CD: Automated Confidence

**Continuous Integration (CI)** means every code change is automatically built and tested. When you push a commit, a server pulls your code, installs dependencies, runs the test suite, runs the linter, and reports pass/fail. No human has to remember to run the tests. No "it works on my machine" -- it either passes CI or it doesn't.

**Continuous Deployment (CD)** extends this: if CI passes, the code is automatically deployed to production. Not every team does this (some have a manual approval step), but the principle is the same -- minimize the time between writing code and getting it to users.

```
Developer pushes code
        │
        ▼
┌────────────────┐     ┌──────────────┐     ┌────────────────┐
│  Build & Test  │────→│  Code Review │────→│    Deploy       │
│   (automated)  │     │   (human)    │     │  (automated)    │
│                │     │              │     │                 │
│  Lint? ✓       │     │  Looks good? │     │  Staging first  │
│  Types? ✓      │     │  Approved ✓  │     │  then prod      │
│  Tests? ✓      │     │              │     │                 │
└────────────────┘     └──────────────┘     └────────────────┘
```

CI/CD is the infrastructure that makes all the other practices sustainable. Tests are only useful if they run every time. Linting rules only matter if they're enforced. CI makes the rules automatic.

### Debugging Like an Engineer

Amateur debugging: stare at the code, add print statements randomly, change things until it works, move on without understanding why.

Professional debugging: apply the scientific method.

```
1. REPRODUCE       Can you reliably trigger the bug?
       │           If not, gathering more information is step one.
       ▼
2. HYPOTHESIZE     Based on the symptoms, what could cause this?
       │           Form a specific, testable prediction.
       ▼
3. TEST            Design an experiment that distinguishes your
       │           hypothesis from alternatives.
       ▼
4. CONCLUDE        Was your hypothesis right?
       │           If yes → fix the root cause.
       │           If no  → back to step 2 with new information.
       ▼
5. VERIFY          Does your fix actually resolve the bug?
       │           Did you introduce any new bugs?
       ▼
6. PREVENT         Add a test that catches this bug.
                   If it broke silently, add monitoring.
```

Key disciplines:
- **Change one thing at a time.** If you change three things and the bug disappears, you don't know which change fixed it (or if you just masked it).
- **Read the error message.** Really read it. The line number, the stack trace, the actual words. Most errors tell you exactly what's wrong if you slow down and read.
- **Reproduce before fixing.** If you can't trigger the bug on demand, you can't verify your fix.
- **Bisect.** `git bisect` is a binary search through your commit history to find exactly which commit introduced the bug. It's criminally underused.

### Technical Debt

Ward Cunningham coined this metaphor: technical debt is like financial debt. You take a shortcut now (ship a hack, skip tests, copy-paste instead of abstracting) and it makes you faster today. But it accrues interest -- every future change in that area is harder, slower, and riskier.

```
                  ┌─────────────────────────────────────────┐
                  │              Code Quality                │
                  │                                         │
   Ship fast ←────┤                                         ├────→ Ship right
                  │  ┌───────────────────────────────────┐  │
                  │  │  "Move fast and break things"      │  │
                  │  │                                    │  │
                  │  │  ─→ works until you have users     │  │
                  │  │     who care about "things"        │  │
                  │  │     breaking                       │  │
                  │  └───────────────────────────────────┘  │
                  │                                         │
                  │  The skill is knowing where on this     │
                  │  spectrum to be, and when.              │
                  └─────────────────────────────────────────┘
```

Not all technical debt is bad. Sometimes shipping a quick hack today and cleaning it up next sprint is the right business decision. The danger is *unacknowledged* debt -- hacks that nobody tracks, that pile up until the codebase is a minefield.

**The Rewrite Trap:** When a codebase accumulates too much debt, the temptation is to rewrite it from scratch. This almost always goes badly. Joel Spolsky called it "the single worst strategic mistake that any software company can make." The old codebase, ugly as it is, contains years of bug fixes and edge case handling. The rewrite loses all of it and takes twice as long as estimated. Usually the answer is incremental refactoring, not a Big Rewrite.

---

## Sitting 3: Designing Systems

### Architecture Is About Boundaries

System architecture is the art of deciding where to draw boundaries. Which things are grouped together? Which things communicate over a network? Which things share a database? These are the decisions that are hardest to change later.

```
Monolith                          Microservices
─────────                         ─────────────

┌────────────────────┐            ┌───────┐ ┌───────┐ ┌───────┐
│                    │            │ Auth  │ │ Users │ │ Posts │
│  Auth + Users +   │            │Service│ │Service│ │Service│
│  Posts + Search + │            └───┬───┘ └───┬───┘ └───┬───┘
│  Billing + Email  │                │         │         │
│                   │            ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
│  One process.     │            │ AuthDB│ │UserDB │ │PostDB │
│  One database.    │            └───────┘ └───────┘ └───────┘
│  Simple.          │
└────────┬──────────┘            Each service: own process,
         │                       own database, own deploy.
    ┌────┴─────┐                 Complex but independent.
    │    DB    │
    └──────────┘
```

**Start with a monolith.** This is the near-universal advice from people who've done both. A monolith is simpler to build, deploy, debug, and reason about. Microservices solve scaling and team-independence problems that you don't have yet. The cost of splitting a monolith later is much lower than the cost of managing premature microservices now.

Martin Fowler: "Almost all the successful microservice stories have started with a monolith that got too big and was broken up."

### Databases: State Is the Hard Part

Stateless code is easy. You have inputs, you produce outputs, done. The hard part of any real system is **state** -- the data that persists between requests. That's the database.

The two main families:

**Relational databases** (PostgreSQL, MySQL, SQLite): Data lives in tables with rows and columns. Relationships are explicit (foreign keys). You query with SQL. They enforce schemas and consistency. They've been the right choice for 90% of applications for 50 years. If you're not sure what database to use, use PostgreSQL.

**Document databases** (MongoDB, DynamoDB): Data lives as flexible JSON-like documents. No schema enforced by the database. Easier to get started, harder to maintain as your data model evolves. Useful when your data is genuinely unstructured or when you need massive horizontal scale.

```
Relational (PostgreSQL)              Document (MongoDB)
───────────────────────              ──────────────────

users table:                         users collection:
┌────┬──────────┬────────┐           {
│ id │ name     │ email  │             "_id": "abc123",
├────┼──────────┼────────┤             "name": "Ada",
│ 1  │ Ada      │ a@x.io │             "email": "a@x.io",
│ 2  │ Grace    │ g@y.io │             "addresses": [
└────┴──────────┴────────┘               {"city": "London"},
                                         {"city": "Paris"}
addresses table:                       ]
┌────┬─────────┬────────┐           }
│ id │ user_id │ city   │
├────┼─────────┼────────┤           Everything about one user
│ 1  │ 1       │ London │           is in one document. Flexible,
│ 2  │ 1       │ Paris  │           but harder to query across
└────┴─────────┴────────┘           documents.

Normalized — no duplicate data.
Joins connect tables.
```

**Migrations:** Your database schema will change. You'll add columns, rename tables, change types. Database migrations are scripts that evolve your schema incrementally, like version control for your database structure. Every real project needs them. Tools like Alembic (Python), Flyway (Java), or Prisma (Node.js) manage this.

### Caching: Trading Freshness for Speed

A cache stores the result of an expensive operation so you don't have to repeat it. It's the single most effective performance optimization in software.

```
Without cache:                        With cache:

Browser → Server → Database           Browser → Server → Cache hit?
                                                  │          │
         Response: 200ms                          │     Yes: 5ms
                                                  │
                                              No: Server → Database
                                                          200ms
                                              (store result in cache)
```

Caching seems simple but introduces one of the hardest problems in computer science:

> "There are only two hard things in Computer Science: cache invalidation and naming things." — Phil Karlton

When the underlying data changes, the cache becomes stale. When do you invalidate it? Options:
- **Time-based (TTL):** Cache expires after N seconds. Simple but you serve stale data for up to N seconds.
- **Event-based:** When data changes, explicitly delete the cached version. Accurate but complex to implement correctly.
- **Never:** Some data (like a user's profile photo URL) changes so rarely that a long TTL is fine.

Common caching layers: browser cache (HTTP headers), CDN (Cloudflare, CloudFront), application cache (Redis, Memcached), database query cache, CPU cache (hardware). Each layer trades freshness for speed.

### Scaling: Vertical vs. Horizontal

Your app is slow and your server is overwhelmed. You have two options:

```
Vertical scaling                     Horizontal scaling
(bigger machine)                     (more machines)

┌──────────────────┐                 ┌────────┐ ┌────────┐ ┌────────┐
│                  │                 │Server 1│ │Server 2│ │Server 3│
│   BIGGER         │                 └────┬───┘ └────┬───┘ └────┬───┘
│   SERVER         │                      │         │         │
│                  │                 ┌────┴─────────┴─────────┴───┐
│   More CPU,      │                 │        Load Balancer        │
│   More RAM,      │                 │    (distributes requests)   │
│   Faster disk    │                 └────────────────────────────┘
│                  │
└──────────────────┘                 Same code on each server.
                                     State must be externalized
Simple. No code changes.             (database, Redis — not in
But there's a ceiling.               memory on the server).
```

**Vertical first.** Modern servers are powerful. A single well-optimized server can handle surprisingly high traffic. Vertical scaling requires zero code changes -- you just give your server more resources. Only go horizontal when vertical scaling hits its limits or when you need redundancy (if one server dies, others keep running).

Horizontal scaling requires that your application is **stateless** -- each request can be handled by any server. This means session data, file uploads, and any user-specific state must live in an external store (database, Redis, S3), not in memory on the server.

### The CAP Theorem (Briefly)

In a distributed system, you can have at most two of three properties:

- **Consistency:** Every read gets the most recent write
- **Availability:** Every request gets a response (even if it's stale)
- **Partition tolerance:** The system keeps working even if network links between nodes fail

Network partitions *will* happen (this is the real world), so partition tolerance isn't optional. That leaves you choosing between consistency and availability during a failure.

Your bank chooses consistency -- it would rather refuse your request than show you a wrong balance. A social media feed chooses availability -- it would rather show you slightly stale posts than show nothing.

In practice, most applications don't need to think about CAP until they're distributed across multiple servers. If you have one database server, it's not a distributed system and CAP doesn't apply. Don't over-engineer for problems you don't have.

### Security: The Basics Everyone Must Know

Security isn't a feature you add later. It's a property of the system that you either maintain from the start or spend enormous effort retrofitting.

The essentials (the OWASP Top 10 in plain language):

**1. Never trust user input.** Every piece of data from a user -- form fields, URL parameters, headers, cookies, file uploads -- is potentially malicious. Validate and sanitize everything at the boundary.

**2. SQL injection.** If you build SQL queries by string concatenation, an attacker can inject arbitrary SQL. Always use parameterized queries:
```python
# WRONG — injectable:
cursor.execute(f"SELECT * FROM users WHERE id = {user_input}")

# RIGHT — parameterized:
cursor.execute("SELECT * FROM users WHERE id = %s", (user_input,))
```

**3. Authentication ≠ Authorization.** Authentication: "who are you?" Authorization: "are you allowed to do this?" Verifying a user's identity (login) doesn't mean they can access any resource. Always check permissions on every request.

**4. Secrets belong in environment variables.** API keys, database passwords, signing keys -- never in source code, never in git. Use `.env` files locally and secret managers in production.

**5. HTTPS everywhere.** Unencrypted HTTP means anyone on the network can read (and modify) the traffic. In 2026, there is no excuse for not using HTTPS. Let's Encrypt makes it free.

**6. Hash passwords.** Never store passwords in plaintext. Use bcrypt, argon2, or scrypt. Not MD5. Not SHA-256. Those are fast hashes -- you want a *slow* hash for passwords, because slow means expensive for attackers to brute-force.

---

## Sitting 4: The Human Side

### Estimation and Why It's Hard

Software estimation is notoriously unreliable. This isn't because engineers are bad at it -- it's because software development is *exploration*. You're building something that hasn't been built before (if it had, you'd use the existing one). The unknowns are genuinely unknown.

Hofstadter's Law: "It always takes longer than you expect, even when you take into account Hofstadter's Law."

Practical strategies:
- **Break it down.** Estimate small pieces, not the whole project. The error on each piece is smaller.
- **Multiply by π.** Only half-joking. Most individual estimates are 2-4x too optimistic. The unknown unknowns dominate.
- **Track actuals.** Compare your estimates to how long things actually took. Over time, you calibrate.
- **Use ranges.** "2-5 days" is more honest than "3 days" and lets stakeholders plan for the realistic case.

### Documentation as Communication

Code tells you *what* the system does. Documentation tells you *why* and *how to use it*.

The most valuable documentation (in order):
1. **A good README.** How to set up, how to run, where things live. One page. Keep it current.
2. **Architecture decision records (ADRs).** When you make a significant design decision, write down what you decided, what alternatives you considered, and why you chose this one. Future developers (including future you) will wonder "why is it like this?" -- ADRs answer that question.
3. **API documentation.** If other people (or other teams) call your code, document the interface. What are the endpoints? What are the parameters? What does it return? What errors are possible?
4. **Inline comments for *why*, not *what*.** `x += 1  # increment x` is useless. `x += 1  # account for the off-by-one in the upstream API` is valuable.

Documentation that describes what the code does line-by-line is almost always a waste -- the code already says that. Documentation that explains *why* the code exists, what problem it solves, and what trade-offs were made is invaluable.

### Incident Response: When Things Break in Production

Things will break. Servers will go down, databases will run out of space, deploys will introduce bugs. How you respond matters more than preventing every possible failure.

```
1. DETECT         Monitoring alerts you (or a user reports it).
       │
       ▼
2. TRIAGE         How bad is it? Who's affected? What's the blast radius?
       │
       ▼
3. MITIGATE       Stop the bleeding. Rollback the deploy, scale up,
       │          redirect traffic. Fix the symptom first.
       ▼
4. ROOT CAUSE     Now figure out why. Was it a code bug? A config
       │          change? A dependency failure? A traffic spike?
       ▼
5. FIX            Fix the underlying cause, not just the symptom.
       │
       ▼
6. POSTMORTEM     Write up what happened, why, and how to prevent it.
                  Blameless. The goal is learning, not punishment.
```

The most important discipline: **mitigate first, investigate second.** When the site is down, roll back the deploy *before* you spend 30 minutes finding the bug. Restore service, then figure out why it broke.

**Blameless postmortems** are critical. If people are punished for incidents, they hide information. If incidents are treated as learning opportunities, the team gets better. Every major tech company does blameless postmortems. The question is never "who screwed up?" but "what in our process allowed this to happen?"

### The Myth of the 10x Developer

The idea of a "10x developer" -- an individual who is ten times more productive than average -- is mostly a myth in the way people think about it. Nobody writes code 10x faster.

What does exist: developers who make their *team* dramatically more productive. They do this not by typing faster but by:
- Making good architectural decisions that prevent months of rework
- Writing code that others can understand and build on
- Reviewing code carefully and mentoring junior developers
- Identifying the right problem to solve (not just solving problems fast)
- Saying "we shouldn't build this" when appropriate

The highest-leverage thing you can do as an engineer isn't writing more code. It's reducing the amount of code that needs to be written.

---

## A Map of Where You Stand

Now you can place things on the map. When someone says:

- **"We need to refactor the billing module"** -- they mean the billing code has high coupling to other modules or low cohesion internally. Changes to billing are breaking unrelated features, or billing logic is scattered across the codebase. The fix is better module boundaries.

- **"Let's add a caching layer"** -- they want to put Redis (or similar) between the application and the database to avoid repeating expensive queries. You'll need to think about cache invalidation -- when do you clear the cache so users don't see stale data?

- **"CI is red"** -- the automated build/test pipeline is failing. Nobody should merge code until it's green. Go look at the build log, find the failing test, fix it.

- **"That's a lot of technical debt"** -- the codebase has accumulated shortcuts that are making future work harder. It's not a crisis, but if unaddressed, velocity will keep dropping.

- **"We need better observability"** -- when something goes wrong, the team can't figure out what happened. They need structured logging, metrics, and dashboards so they can diagnose production issues.

- **"This should be behind a feature flag"** -- deploy the code but don't enable it for users yet. A configuration flag controls whether the new feature is active. Lets you decouple deployment from release.

- **"Let's write an ADR for this"** -- the decision is significant enough that future developers will wonder why it was made. Write it down now while the context is fresh.

---

## Further Reading

These are the best resources for going deeper. They're the books and essays that shaped how the industry thinks about software engineering.

**The essentials:**
- *A Philosophy of Software Design* by John Ousterhout -- the best modern book on managing complexity. Short, opinionated, practical. If you read one book from this list, make it this one.
- *The Pragmatic Programmer* by Hunt & Thomas -- wide-ranging advice on the craft of software development. Full of principles that stick with you.

**Design and architecture:**
- *Clean Architecture* by Robert C. Martin -- expands on SOLID and presents a layered architecture philosophy. Take the dogma with a grain of salt, but the core ideas are sound.
- *Designing Data-Intensive Applications* by Martin Kleppmann -- the definitive guide to databases, distributed systems, and data infrastructure. Dense but transformative.
- *Design Patterns* by the "Gang of Four" -- the original patterns book. Academically important. Skim the catalog; don't try to memorize it.

**The classics:**
- *The Mythical Man-Month* by Fred Brooks (1975) -- "adding people to a late project makes it later." Still true. Still being ignored.
- "No Silver Bullet" by Fred Brooks (1986 essay) -- the essential/accidental complexity distinction. 10 pages. Read it.
- *Out of the Tar Pit* by Moseley & Marks (2006 paper) -- argues that state management is the primary source of complexity. Influenced functional programming's rise.

**Testing and reliability:**
- *Working Effectively with Legacy Code* by Michael Feathers -- how to add tests and make changes to code that was never designed for it. Invaluable if you inherit a codebase.
- *Site Reliability Engineering* by Google -- how Google runs production systems. Free online. The chapters on postmortems, monitoring, and on-call are gold.

**The human side:**
- *An Elegant Puzzle* by Will Larson -- systems thinking applied to engineering management. Useful even if you're not a manager, because it explains why organizations make the decisions they do.
- *Team Topologies* by Skelton & Pais -- how team structure shapes (and is shaped by) software architecture. Conway's Law made practical.

---

*That's the territory. You don't need to have mastered all of this -- you need to know these forces exist and how they push on each other. The next time someone proposes a "quick hack," you'll understand the debt. The next time a deploy breaks production, you'll know how to respond. The next time the codebase feels overwhelming, you'll know where to draw the boundaries. That's the difference between writing code and engineering software.*
