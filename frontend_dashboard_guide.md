# Frontend Design and Dashboards

*How to think about, describe, and build the thing between your data and your users.*

You have data. Maybe it's in a PostgreSQL database, maybe it's scraped from the web, maybe it's coming from an API. And you want a human being to be able to look at it, filter it, click on things, and make decisions. You need a dashboard -- a "user-friendly interface to some data."

But what should it look like? How do you describe what you want? How do you choose between Streamlit and React and Flutter and a single HTML file? And once you've chosen, how do you make it not look terrible?

This document is a framework for thinking about all of that. Not a tutorial for any specific tool -- a way to think about the *design* problem so that when you sit down to build (or to describe what you want built), you know what you're actually asking for.

Four sittings. One coffee each.

---

## Sitting 1: What Are You Actually Building?

### The Dashboard Spectrum

Not all dashboards are the same thing. They sit on a spectrum from "just show me the data" to "this is a full application":

```
Simple                                                          Complex
───────────────────────────────────────────────────────────────────────

 Data table       Explorer         Monitor         Workflow        App
 with filters     with charts      with alerts     with actions    with
                                                                   state

 "Let me see      "Let me slice    "Tell me when   "Let me do      Full
  my data"         and dice"        something's     things to       CRUD,
                                    wrong"          the data"       auth,
                                                                   roles

 ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 │  Table   │    │  Filters │    │  Status  │    │  Kanban  │    │  Multi-  │
 │  + sort  │    │  + chart │    │  + graph │    │  + forms │    │  screen  │
 │  + page  │    │  + drill │    │  + alert │    │  + drag  │    │  + auth  │
 └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘

 Streamlit         Streamlit        Vanilla JS       React            Flutter
 Flask template    Flask + Chart    + API polling    Next.js          full app
 single HTML       React                            Flask + htmx
```

Where you sit on this spectrum determines everything -- your tech stack, your complexity budget, and how you think about the design.

### Five Dashboard Archetypes

Almost every dashboard you'll build fits one of these patterns:

**1. The Report**
Static or near-static. Shows data, maybe with filters. Read-only. Your audience opens it, looks at the numbers, closes it. Think: analytics dashboard, monthly metrics page, leaderboard.

**2. The Explorer**
Interactive data exploration. The user doesn't know what they're looking for -- they want to slice, filter, drill down, and discover patterns. Think: a dataset browser, a search interface, a faceted catalog.

**3. The Monitor**
Watches something in real-time (or near-real-time) and surfaces what's important. Status indicators, timelines, alerts. Think: server health dashboard, activity feed, CI/CD pipeline status.

**4. The Workflow**
The user doesn't just view data -- they *act* on it. Change statuses, assign things, create entries, drag cards between columns. Think: kanban board, CMS, admin panel, issue tracker.

**5. The Control Panel**
A settings/configuration interface. Forms, toggles, dropdowns. The user is configuring a system. Think: admin settings page, feature flag manager, user management panel.

### Your Projects, Mapped

Here's how your existing dashboards map to these archetypes:

```
Project                    Archetype        Stack              Why that stack?
─────────────────────────────────────────────────────────────────────────────

AU AI/ML Scout             Explorer         Streamlit          Data exploration,
                                                               rapid prototyping

Alexar                     Explorer         Streamlit          Same: explore data,
                                                               charts, filters

Bills                      Workflow         Flask              CRUD + AI categorize,
                                                               server-side logic

Lyra-Claudius Dashboard    Monitor          Vanilla JS         Real-time GitHub API,
                                                               zero dependencies

Focus Dashboard            Monitor          Vanilla JS         Same: team activity
                                                               feed from GitHub

CMS (Collaboration Tools)  Workflow         Flutter            Kanban, graph, CRUD,
                                                               cross-platform, auth

Book Friend Finder         Explorer         Flask              Recommendations +
                                                               browsing

Gateway                    Workflow + App   Next.js + React    Web3, auth, complex
                                                               state, marketplace

Accountability             Monitor          Express + HTML     Team progress
                                                               tracking

Math Research Tools        Explorer         FastAPI            Knowledge graph,
                                                               PDF processing
```

Notice the pattern: **the archetype drove the stack choice** (whether you knew it at the time or not). Explorers gravitate to Streamlit. Monitors to vanilla JS with API polling. Workflows to React or Flutter. This isn't a coincidence -- different archetypes have different needs.

### Know Your Archetype Before You Start

Before writing any code, ask yourself: *which archetype am I building?* This single question resolves an enormous number of downstream decisions:

```
                        Report   Explorer   Monitor   Workflow   Control
                        ──────   ────────   ───────   ────────   ───────
Needs real-time?         No      No         YES       Maybe      No
User creates data?       No      No         No        YES        YES
Complex interactions?    No      Some       No        YES        Some
Needs auth?              Maybe   Maybe      Maybe     YES        YES
Needs a backend?         Maybe   Maybe      Maybe     YES        YES
Mobile-friendly?         Nice    Nice       YES       Depends    No
Offline support?         No      No         No        Nice       No
```

An Explorer that you're building as a Workflow is over-engineered. A Workflow that you're building as a Report is missing features. Get the archetype right and the rest follows.

---

## Sitting 2: Describing What You Want

### The Problem: "I'll Know It When I See It"

The hardest part of building a dashboard isn't the code. It's figuring out *what you're building*. Most people have a vague picture in their head -- "a nice dashboard with the data" -- and then they start coding and discover, mid-implementation, that they don't know what "nice" means or which data matters.

You need to make the invisible visible *before* you code. There are three levels of describing a UI, and you should go through them in order.

### Level 1: The Data Inventory

Before you draw a single box, list your data. What entities exist? What properties do they have? What are the relationships? This is the foundation everything else rests on.

```
Example: A startup scouting dashboard

ENTITIES:
  Company
    - name (text)
    - founded_year (number)
    - sector (enum: AI, Biotech, Fintech, Climate, ...)
    - stage (enum: Seed, Series A, Series B, ...)
    - funding_total (currency)
    - description (long text)
    - website (URL)
    - location (city, country)

  Investor
    - name (text)
    - type (enum: VC, Angel, Accelerator, Government)
    - portfolio_size (number)

  Funding Round
    - company → Company
    - investor → Investor
    - amount (currency)
    - date (date)
    - round_type (enum: Seed, Series A, ...)

KEY QUESTIONS:
  - How many companies? (~500) → table with pagination
  - How many investors? (~200) → secondary entity
  - What do users want to DO? → Browse, filter, compare, export
  - What's the primary entry point? → Company list, then drill into details
```

The data inventory tells you what the dashboard *can* show. It's the raw material. Everything else is about choosing what to show, where, and how.

### Level 2: The Information Hierarchy

Not all data is equally important. The information hierarchy is your decision about what matters most. It answers: **when someone opens this dashboard, what should they see first?**

```
Primary     What the user sees immediately.        The main table/list/grid.
            This is the "home screen."              The summary numbers.
            Maximum 3-5 pieces of information.      The key chart.

Secondary   What appears when they interact.        Filters, detail panels,
            One click away from primary.            expanded rows, tabs.

Tertiary    What appears on demand.                 Modals, settings, export
            Available but not visible by default.   options, advanced filters,
                                                    historical data.
```

Applied to the AU AI/ML Scout:

```
PRIMARY:    Company table (name, sector, stage, funding)
            Summary stats bar (total companies, total funding, avg round size)

SECONDARY:  Filter sidebar (sector, stage, location, funding range)
            Company detail panel (full description, funding history, investors)
            Sector breakdown chart

TERTIARY:   Export to CSV
            Investor network graph
            Advanced search (regex, date ranges)
```

This hierarchy prevents the most common dashboard mistake: **showing everything at once.** A dashboard that shows all data with equal prominence is a spreadsheet. The whole point is *curation* -- deciding what matters and making it easy to find.

### Level 3: The Wireframe

A wireframe is a rough sketch of the layout. It doesn't need to be pretty. It needs to show *where things go.* You can draw this on paper, in a notes app, or in ASCII:

```
┌──────────────────────────────────────────────────────────┐
│  Logo    [Search...........................] [Export] [⚙] │
├──────────┬───────────────────────────────────────────────┤
│          │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐            │
│ Filters  │  │ 523 │ │ $2B │ │  47 │ │  12 │            │
│          │  │total│ │funds│ │ seed│ │ IPO │            │
│ Sector   │  └─────┘ └─────┘ └─────┘ └─────┘            │
│ □ AI     │                                              │
│ □ Bio    │  ┌──────────────────────────────────────────┐ │
│ □ Fin    │  │  Name    Sector   Stage   Funding   ▼   │ │
│          │  ├──────────────────────────────────────────┤ │
│ Stage    │  │  Acme    AI       Seed    $2.1M         │ │
│ □ Seed   │  │  Beta    Bio      Ser.A   $15M          │ │
│ □ Ser.A  │  │  Gamma   Fin      Ser.B   $45M          │ │
│ □ Ser.B  │  │  Delta   AI       Seed    $800K         │ │
│          │  │  ...     ...      ...     ...           │ │
│ Funding  │  ├──────────────────────────────────────────┤ │
│ $0-$5M   │  │  < 1  2  3  4  5 ... 26 >              │ │
│ $5M-$50M │  └──────────────────────────────────────────┘ │
│ $50M+    │                                              │
│          │  ┌──────────────────────────────────────────┐ │
│ Location │  │  [Funding by Sector bar chart]           │ │
│ ▼ Select │  │  █████████ AI: $800M                     │ │
│          │  │  ██████ Bio: $500M                       │ │
│ [Clear]  │  │  ████ Fin: $350M                        │ │
│          │  └──────────────────────────────────────────┘ │
└──────────┴───────────────────────────────────────────────┘
```

This wireframe tells you (or anyone you're working with) exactly what to build. It answers:
- What's in the sidebar? Filters.
- What's the main content? A table with summary cards above it.
- What's below the table? A chart.
- What are the interaction points? Filter checkboxes, table sorting, pagination, search, export.

You can draw this in 5 minutes. It saves hours of "that's not what I meant" during implementation.

### Describing UI to an AI (or a Developer)

When you're asking Claude (or a teammate) to build a dashboard, the quality of the result depends almost entirely on the quality of the description. Here's what works:

**Bad:** "Build me a dashboard for my startup data."

**Better:** "Build me an explorer dashboard for ~500 Australian AI/ML startups. Main view is a filterable, sortable table showing company name, sector, stage, and total funding. Left sidebar has checkbox filters for sector and stage, and a slider for funding range. Clicking a row opens a detail panel on the right showing the full company profile and funding history."

**Best:** The wireframe above + the data inventory + a note about the archetype.

The pattern is: **archetype + data + layout + interactions.**

```
Template for describing a dashboard:

1. ARCHETYPE:  "This is a [report/explorer/monitor/workflow/control panel]"
2. DATA:       "The main entities are [X, Y, Z] with these properties: [...]"
3. LAYOUT:     "The main screen shows [primary content]. [Sidebar/topbar] has
                [filters/navigation]. Clicking [X] shows [detail/modal/panel]."
4. ACTIONS:    "The user can [filter, sort, search, create, edit, delete, export,
                drag, toggle, ...]"
5. MOOD:       "It should feel [dense/spacious, professional/playful,
                data-heavy/visual, minimal/feature-rich]"
```

That fifth point -- mood -- is surprisingly important. "Dense and data-heavy like a Bloomberg terminal" produces a very different result than "clean and spacious like a Notion page."

---

## Sitting 3: The Data-to-Pixels Pipeline

### Data Shape Determines UI Shape

This is the single most important insight in dashboard design: **the shape of your data suggests the shape of your UI.** You don't choose UI patterns arbitrarily -- the data tells you what works.

```
Data shape                    Natural UI pattern          Example
──────────────────────────────────────────────────────────────────────

List of items with            Table with sort/filter      Company list,
  properties                                              user admin panel

Items with a status           Kanban board                Task tracker,
  enum (stages)               (columns = statuses)        pipeline view

Time series                   Line/area chart             Revenue over time,
  (value over time)                                       server metrics

Categories with               Bar chart, pie chart,       Funding by sector,
  quantities                  treemap                     traffic by source

Tree / hierarchy              Tree view, breadcrumbs,     File system,
                              nested list                 org chart

Graph / network               Force-directed graph,       Knowledge graph,
                              adjacency matrix            social network

Single item with              Detail page / card          User profile,
  many properties             with sections               company page

Key-value pairs               Definition list,            Settings page,
                              form fields                 config panel

Geospatial                    Map with markers/           Store locator,
                              heatmap                     regional data
```

When you're stuck on "what should this look like?", look at your data. If you have 500 companies with properties, the answer is almost certainly "a table." If those companies have stages, consider adding a kanban view. If they have locations, consider a map. The data speaks.

### Master-Detail: The Pattern You'll Use Most

The single most common dashboard pattern is **master-detail**: a list of items (master) on one side, and the details of the selected item (detail) on the other.

```
┌─────────────────────┬────────────────────────────────┐
│  MASTER (list)      │  DETAIL (selected item)        │
│                     │                                │
│  ▸ Company Alpha    │  Company Beta                  │
│  ▶ Company Beta  ←──│──                              │
│  ▸ Company Gamma    │  Sector: AI/ML                 │
│  ▸ Company Delta    │  Stage: Series A               │
│  ▸ Company Epsilon  │  Founded: 2023                 │
│                     │  Funding: $15M                 │
│                     │                                │
│                     │  Description:                  │
│                     │  Lorem ipsum dolor sit amet... │
│                     │                                │
│                     │  Funding History:              │
│                     │  ┌────────┬────────┬────────┐  │
│                     │  │ Seed   │ $2M    │ 2023   │  │
│                     │  │ Ser. A │ $13M   │ 2024   │  │
│                     │  └────────┴────────┴────────┘  │
└─────────────────────┴────────────────────────────────┘
```

Your CMS uses this pattern: task list on the left, task detail on the right. The Lyra-Claudius dashboard uses a variant: activity feed (master) with expanded event details. The AU Scout uses it: company table (master) with company detail on click.

The master-detail pattern works because it matches how humans process information: scan a list to find something interesting, then dive into the details. The master provides *context* (where am I in the set?) while the detail provides *depth*.

Variations:
- **Side-by-side** (desktop): master left, detail right. Best when you switch between items frequently.
- **Drill-down** (mobile or simple): master fills the screen, clicking an item navigates to a detail page. Better for mobile or when details are complex.
- **Modal/overlay**: master stays visible, detail appears as a panel or popup. Good for quick edits.

### The Read Path: How Data Becomes Pixels

Every dashboard has this pipeline, whether you make it explicit or not:

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Data   │────→│  API    │────→│  State  │────→│  View   │────→│ Pixels  │
│  Store  │     │  Layer  │     │  Layer  │     │  Layer  │     │         │
│         │     │         │     │         │     │         │     │         │
│ Postgres│     │ REST    │     │ React   │     │ JSX/    │     │ Browser │
│ SQLite  │     │ GraphQL │     │ state   │     │ HTML    │     │ renders │
│ JSON    │     │ direct  │     │ Riverpod│     │ Widgets │     │ Flutter │
│ CSV     │     │ file    │     │ Streamlit│    │ st.write│     │ paints  │
│         │     │ read    │     │ session │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
```

Different stacks collapse different layers:

- **Streamlit** collapses API + State + View into one: `st.dataframe(df)`. You go straight from data to pixels. This is why it's fast to build but hard to customize.
- **Flask templates** collapse State + View: the server renders HTML directly from database queries. No client-side state management.
- **Vanilla JS** makes the API layer explicit (fetch calls) but handles state informally (variables, DOM state).
- **React/Flutter** make every layer explicit: API client → state management → component tree → render. More code, but each layer is testable and replaceable.

The question is: how much control do you need at each layer? If the answer is "not much," use Streamlit. If the answer is "total control over every pixel and interaction," use React or Flutter.

### Where Filtering Happens

Filtering is the core interaction in most dashboards. But *where* you filter matters:

```
Database-side filtering              Client-side filtering
──────────────────────               ─────────────────────

SELECT * FROM companies              fetch('/api/companies')
WHERE sector = 'AI'                    → get ALL companies
AND funding > 1000000                  → filter in JavaScript
ORDER BY funding DESC                  → sort in JavaScript
LIMIT 50 OFFSET 100                    → paginate in JavaScript

✓ Handles millions of rows           ✓ Instant interaction (no network)
✓ Only transfers what you need       ✓ Simpler backend
✗ Every filter change = API call     ✗ Must load all data upfront
✗ Slower interactions                ✗ Breaks at ~10,000+ rows
```

**Rule of thumb:**
- Under 1,000 items: load everything, filter client-side. Instant and simple.
- 1,000 - 10,000 items: either works. Client-side with some care, or server-side with debounced API calls.
- Over 10,000 items: server-side filtering is mandatory. You can't send 10,000 items to the browser on every page load.

Your AU Scout (~500 companies) can load everything and filter client-side. Your CMS with thousands of tasks probably needs server-side filtering. The choice has real UX implications: client-side filtering feels instant; server-side filtering has a loading state.

### Real-Time vs. Polling vs. Static

How fresh does the data need to be?

```
Static                 Polling                 Real-time
──────                 ───────                 ─────────

Load once.             Fetch every N           Server pushes updates
User refreshes         seconds.                as they happen.
to see changes.
                       setInterval(() => {     WebSocket or SSE
                         fetch('/api/data')    connection stays open.
                       }, 30000)

Reports,               Activity feeds,         Chat, live metrics,
historical data,       monitoring dashboards,  collaborative editing,
batch analytics        status pages            stock tickers

Your Scout dashboard   Your Lyra-Claudius      Your CMS kanban
                       dashboard               (if multi-user)
```

Polling is the pragmatic middle ground. A 30-second polling interval covers 90% of "I want it to feel live" use cases without the complexity of WebSockets. Your Lyra-Claudius dashboard polls the GitHub API and it feels current enough.

Real-time (WebSockets) is worth the complexity only when latency matters -- chat, collaborative editing, live game state. For most dashboards, polling is fine.

---

## Sitting 4: Making It Not Ugly

### Visual Hierarchy: What the Eye Sees First

Your eye doesn't read a dashboard like a book. It scans. It's drawn to certain things first:

```
What draws the eye (strongest → weakest):

1. SIZE          Large elements are seen first
2. COLOR         Especially saturated colors on a neutral background
3. CONTRAST      Light on dark, dark on light
4. POSITION      Top-left in LTR cultures, center of screen
5. WHITESPACE    Isolated elements stand out
6. MOTION        Anything that moves (use sparingly!)
```

The implication: **make important things big, colorful, or isolated. Make unimportant things small, gray, or grouped with other things.**

```
Good hierarchy                       Bad hierarchy (everything equal)
──────────────                       ───────────────────────────────

┌────────────────────────┐           ┌────────────────────────┐
│  523 Companies         │           │ Companies: 523         │
│  ████████████████████  │           │ Total Funding: $2.1B   │
│                        │           │ Average Round: $4.2M   │
│  $2.1B Total Funding   │           │ Seed Stage: 47         │
│                        │           │ Series A: 31           │
│  47 at Seed Stage      │           │ Series B: 18           │
└────────────────────────┘           │ IPO: 12                │
                                     │ Active: 489            │
The big number (523) is the          │ Inactive: 34           │
headline. Funding is secondary.      └────────────────────────┘
Seed count is a detail.
                                     Everything is the same size.
                                     Nothing stands out.
                                     The eye doesn't know where
                                     to start.
```

### The Layout Patterns

There are really only a handful of dashboard layouts. Nearly every dashboard you've seen is one of these:

**1. Sidebar + Content**
```
┌────────┬───────────────────────────┐
│        │                           │
│  Nav   │     Main content          │
│  or    │                           │
│  Filters│                          │
│        │                           │
└────────┴───────────────────────────┘
```
Used by: almost everything. Your CMS, your Scout dashboard. The sidebar holds navigation or filters. Content is the main stage. This works because it gives persistent context (you always see where you are) without eating too much horizontal space.

**2. Top bar + Content**
```
┌─────────────────────────────────────┐
│  Filters / tabs / search            │
├─────────────────────────────────────┤
│                                     │
│         Main content                │
│                                     │
└─────────────────────────────────────┘
```
Used by: simpler dashboards, mobile-first designs. Good when you have few navigation items or filters. Streamlit defaults to this.

**3. Dashboard Grid**
```
┌────────┬────────┬────────┬────────┐
│ KPI 1  │ KPI 2  │ KPI 3  │ KPI 4  │
├────────┴────┬───┴────────┴────────┤
│             │                     │
│  Chart 1    │     Chart 2         │
│             │                     │
├─────────────┴─────────────────────┤
│           Table / List            │
└───────────────────────────────────┘
```
Used by: analytics/reporting dashboards. KPI cards on top, charts in the middle, detail table at the bottom. This is the "executive dashboard" layout. Grafana, Metabase, and Google Analytics all use variants of this.

**4. Kanban**
```
┌──────────┬──────────┬──────────┬──────────┐
│ Backlog  │ In Prog  │ Review   │ Done     │
├──────────┼──────────┼──────────┼──────────┤
│ ┌──────┐ │ ┌──────┐ │ ┌──────┐ │          │
│ │Card 1│ │ │Card 3│ │ │Card 5│ │          │
│ └──────┘ │ └──────┘ │ └──────┘ │          │
│ ┌──────┐ │ ┌──────┐ │          │          │
│ │Card 2│ │ │Card 4│ │          │          │
│ └──────┘ │ └──────┘ │          │          │
└──────────┴──────────┴──────────┴──────────┘
```
Used by: workflow dashboards where items have stages. Your CMS task board. Trello, Jira, Linear. Only use this when your data has a natural status progression.

### Color With Purpose

Color in dashboards serves exactly three purposes:

**1. Status signaling.** Red = bad/blocked. Yellow = warning/in-progress. Green = good/done. Blue = information/neutral. This is so deeply ingrained that violating it (green for errors, red for success) genuinely confuses users.

**2. Categorical distinction.** Different colors for different categories (sectors, types, teams). Keep it subtle -- muted tones, not saturated primaries. If you need more than 6-7 categories, color alone won't work; add icons or labels.

**3. Visual hierarchy.** A single accent color draws the eye to actions (buttons, links, selected items). Everything else is neutral (grays, white, off-white).

```
Good color usage                      Bad color usage
────────────────                      ───────────────

┌─────────────────────────┐           ┌─────────────────────────┐
│  Tasks                  │           │  Tasks                  │
│                         │           │                         │
│  ● Deploy API    Done   │ ← green   │  ● Deploy API    Done   │ ← green
│  ● Fix login     Stuck  │ ← red     │  ● Fix login     Stuck  │ ← purple
│  ● Write docs   Active  │ ← blue    │  ● Write docs   Active  │ ← orange
│                         │           │  ● Add tests    Todo    │ ← teal
│  [+ New Task]           │ ← accent  │  ● Review PR    Active  │ ← magenta
└─────────────────────────┘           └─────────────────────────┘

Three colors: status meaning is          Five colors: decorative,
instantly clear.                         no consistent meaning.
```

**The 60-30-10 rule:** 60% neutral (backgrounds, text), 30% secondary (cards, borders, subtle fills), 10% accent (buttons, highlights, status). This ratio keeps a dashboard calm and scannable. The moment you have too many colors competing for attention, the dashboard feels chaotic.

### Typography and Density

Two variables control how a dashboard *feels*:

**Information density:** How much data per square centimeter of screen?

```
Low density (spacious)               High density (compact)
──────────────────────               ─────────────────────

┌────────────────────┐               ┌────────────────────────────────┐
│                    │               │ Name     Sector  Stage  Fund   │
│  Company Alpha     │               │ Alpha    AI      Seed   $2.1M  │
│                    │               │ Beta     Bio     SerA   $15M   │
│  Sector: AI        │               │ Gamma    Fin     SerB   $45M   │
│  Stage:  Seed      │               │ Delta    AI      Seed   $800K  │
│  Funding: $2.1M    │               │ Epsilon  Cli     SerA   $22M   │
│                    │               │ Zeta     AI      SerB   $38M   │
└────────────────────┘               │ Eta      Bio     Seed   $1.2M  │
                                     │ Theta    Fin     SerA   $9M    │
Shows 1 company in the               └────────────────────────────────┘
same space.
                                     Shows 8 companies in the
Good for detail views.               same space.
Bad for scanning.
                                     Good for scanning.
                                     Bad for detail views.
```

Match density to the task. Explorers need high density (scan lots of items). Detail views need low density (read and understand one item). The master-detail pattern solves this by having both: high-density list + low-density detail.

**Font sizing:** Use a clear typographic scale. Not ten different sizes -- three or four:

```
Title       24-32px    Page headings, hero numbers
Heading     18-20px    Section headings, card titles
Body        14-16px    Table content, descriptions, labels
Caption     12px       Timestamps, secondary metadata, footnotes
```

Pick a system font (Inter, system-ui, -apple-system) and stick with it. Custom fonts are a design decision you don't need to make for a dashboard.

### Whitespace Is Not Wasted Space

The most common mistake in dashboard design is cramming too much in. Whitespace (empty space between elements) is what makes a layout *readable.* It groups related things and separates unrelated things.

```
No whitespace                        With whitespace
─────────────                        ───────────────

┌──────────────────────┐             ┌──────────────────────────┐
│Name:Alpha            │             │                          │
│Sector:AI             │             │  Name: Alpha             │
│Stage:Seed            │             │  Sector: AI              │
│Funding:$2.1M         │             │  Stage: Seed             │
│Founded:2023          │             │  Funding: $2.1M          │
│Description:Lorem     │             │  Founded: 2023           │
│ipsum dolor sit amet  │             │                          │
│consectetur           │             │  Description:            │
│Team:12people         │             │  Lorem ipsum dolor sit   │
│Status:Active         │             │  amet consectetur        │
│Website:alpha.com     │             │                          │
└──────────────────────┘             │  Team: 12 people         │
                                     │  Status: Active          │
Everything runs together.            │  Website: alpha.com      │
Hard to scan.                        │                          │
                                     └──────────────────────────┘

                                     Grouped by type.
                                     Breathing room.
                                     Eye finds what it needs.
```

The rule: **when in doubt, add more space.** You can always tighten later. Starting tight and trying to add space is much harder because elements have already been positioned relative to each other.

### Responsive Design: The 80% Solution

"Responsive design" means your layout adapts to different screen sizes. For dashboards, there's a pragmatic approach:

```
Desktop (1200px+)          Tablet (768-1199px)       Mobile (< 768px)
─────────────────          ─────────────────────     ──────────────────

┌────┬───────────┐         ┌──────────────────┐      ┌──────────────┐
│    │           │         │  [≡] Title       │      │ [≡] Title    │
│Side│  Content  │         ├──────────────────┤      ├──────────────┤
│bar │           │         │                  │      │              │
│    │           │         │  Content         │      │   Content    │
│    │           │         │  (full width)    │      │  (stacked)   │
│    │           │         │                  │      │              │
└────┴───────────┘         └──────────────────┘      └──────────────┘

Sidebar visible.           Sidebar collapses to       Sidebar is a
Side-by-side content.      hamburger menu.             hamburger menu.
                           Content goes full width.    Everything stacks
                                                       vertically.
```

If your dashboard is primarily desktop (most internal tools are), do desktop-first and make mobile "acceptable" (stacked layout, hamburger nav). If it's user-facing or you know people will use it on phones, design mobile-first and expand for desktop.

Your vanilla JS dashboards (Lyra-Claudius, Focus) are naturally responsive because they're simple. Your CMS handles this with NavigationRail on desktop and BottomNavigationBar on mobile. The key insight: responsive isn't about making the same layout shrink -- it's about having *different layouts* for different sizes.

---

## Sitting 5: Picking Your Stack

### The Decision Matrix

Now the practical question. You've identified your archetype, sketched a wireframe, mapped your data. What do you build it with?

```
                 Speed to   Custom    Complex      Multi-     Long-term
                 prototype  design    interaction  platform   maintenance
                 ─────────  ──────    ──────────   ────────   ───────────
Streamlit        ★★★★★      ★☆☆☆☆    ★★☆☆☆       Web only   ★★★☆☆
Gradio           ★★★★☆      ★☆☆☆☆    ★★☆☆☆       Web only   ★★☆☆☆
Flask + HTML     ★★★★☆      ★★★☆☆    ★★☆☆☆       Web only   ★★★☆☆
Vanilla JS       ★★★☆☆      ★★★★★    ★★★☆☆       Web only   ★★☆☆☆
React / Next.js  ★★☆☆☆      ★★★★★    ★★★★★       Web only   ★★★★★
Flutter          ★★☆☆☆      ★★★★★    ★★★★★       All        ★★★★★
```

### When to Use Each

**Streamlit** — when you want to explore data *right now.* You have a DataFrame, you want filters and charts. You don't care about custom design. Prototype in an hour, decide later if it needs to be "real." Best for: data science, internal tools, quick demos. Your AU Scout lives here correctly.

**Gradio** — when your interface is "input → model → output." File upload, sliders, text fields in; generated content out. Your Voice-to-Image pipeline is the perfect Gradio use case.

**Flask + templates** — when you want a real web app but interactions are mostly page-based (click link, submit form, see result). Server-side rendering is simple and works everywhere. Good for CRUD apps where you don't need real-time client-side interaction. Your Bills dashboard fits here.

**Single HTML + vanilla JS** — when you want zero dependencies, fast loading, and full control. Surprisingly powerful for read-mostly dashboards that pull from an API. Your Lyra-Claudius dashboard is a perfect example: one file, no build step, fetches from GitHub API, renders a beautiful activity feed. Best for: monitors, simple explorers, anything that needs to "just work" anywhere.

**React / Next.js** — when you need complex client-side interactions: drag-and-drop, optimistic updates, complex forms, real-time collaboration, multi-page apps with shared state. The setup cost is high but the payoff is real for Workflow and App-level dashboards. Your Gateway marketplace needs this.

**Flutter** — when you need cross-platform (web + iOS + Android) from a single codebase, or when you want the most sophisticated widget system. Your CMS is the right call for Flutter: kanban with drag-and-drop, knowledge graph visualization, multiple platforms, complex state management with Riverpod. The cost is a heavier toolchain and slower iteration on styling compared to CSS.

### The "Should I Use a Framework?" Flowchart

```
Does the user create, edit, or delete data?
├── No → Does it need complex charts or interactions?
│        ├── No → Single HTML file + vanilla JS
│        └── Yes → Streamlit (data) or vanilla JS (custom)
│
└── Yes → Does it need auth, roles, or multi-user?
         ├── No → Flask + templates (simple CRUD)
         └── Yes → Does it need real-time or drag-and-drop?
                  ├── No → Flask + htmx (enhanced server-rendering)
                  └── Yes → React or Flutter
                           ├── Web only → React / Next.js
                           └── Multi-platform → Flutter
```

### CSS: The Styling Question

The dashboard *works.* Now it needs to not look like a 1998 intranet page. Your options:

**Tailwind CSS** — utility classes directly in your HTML. `<div class="bg-white rounded-lg shadow p-4 flex gap-2">`. No separate CSS file, no naming conventions, just description-as-class. Fast to build, easy to customize, surprisingly maintainable. The dominant choice for modern web dashboards.

**Component libraries** — pre-built components that look good out of the box. shadcn/ui (React + Tailwind), Material UI (React), Vuetify (Vue), Chakra UI. You get buttons, tables, cards, modals, and forms that look professional with zero design effort. The trade-off: they all look similar, and deep customization requires fighting the library.

**Plain CSS** — when you want total control or your project is simple enough that a framework is overkill. Your vanilla JS dashboards use plain CSS and look great. For a single-file dashboard, CSS custom properties (variables) give you theming in ~20 lines:

```css
:root {
  --bg: #ffffff;
  --text: #1a1a2e;
  --accent: #4361ee;
  --border: #e0e0e0;
  --card: #f8f9fa;
  --success: #2d6a4f;
  --warning: #e76f51;
  --radius: 8px;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
```

Change those 9 variables and your entire dashboard re-themes. This is what your single-file dashboards should use.

**Flutter's material/cupertino** — Flutter has its own widget system independent of CSS. Material Design gives you a complete component library. Your CMS uses AppTheme for consistent styling across the app. The advantage of Flutter's approach: styling is type-safe (no typos in class names) and widgets handle their own layout (no CSS box model surprises).

---

## Putting It All Together

Here's the process, start to finish:

```
1. IDENTIFY ARCHETYPE        "This is an explorer dashboard."
         │
         ▼
2. DATA INVENTORY             "500 companies with sector, stage, funding.
         │                     200 investors. Funding rounds connect them."
         ▼
3. INFORMATION HIERARCHY      "Primary: company table + summary stats.
         │                     Secondary: filters + detail panel.
         │                     Tertiary: export + investor graph."
         ▼
4. WIREFRAME                  Draw the layout. Boxes and labels.
         │                     5 minutes. Paper or ASCII.
         ▼
5. PICK STACK                 "~500 items, explorer, read-only →
         │                     Streamlit for prototype,
         │                     React if it needs to be polished."
         ▼
6. BUILD THE SKELETON         Layout first, no real data.
         │                     Get the structure right.
         ▼
7. CONNECT DATA               Wire up the API / database.
         │                     See real data in the skeleton.
         ▼
8. ADD INTERACTIONS           Filters, sorting, clicking, navigation.
         │                     One interaction at a time.
         ▼
9. POLISH                     Color, typography, whitespace, loading
                              states, empty states, error states.
```

Steps 1-4 take an afternoon. They save you from the most expensive kind of waste: building the wrong thing.

The biggest mistake people make is jumping straight to step 6. They open their editor, start a React project, and build *something* -- but without a wireframe or information hierarchy, they end up rearranging the UI five times before it feels right. Those five rearrangements are free if they happen in a wireframe. They cost days if they happen in code.

---

## Further Reading

**Design fundamentals:**
- *Refactoring UI* by Adam Wathan & Steve Schoger -- the single best resource for developers who want to make things look good without becoming designers. Practical, visual, no theory. If you read one book from this list, make it this one.
- *The Design of Everyday Things* by Don Norman -- why some interfaces feel intuitive and others don't. Not about software specifically, but the principles apply directly.

**Dashboard-specific:**
- *Information Dashboard Design* by Stephen Few -- the definitive guide to dashboard visual design. Heavy on data visualization principles and anti-patterns.
- *Storytelling with Data* by Cole Nussbaumer Knaflic -- how to choose the right chart and strip away visual noise. Transforms how you present data.

**Frontend development:**
- The Tailwind CSS docs (tailwindcss.com) -- even if you don't use Tailwind, their "utility-first" philosophy and their design system documentation teach you how to think about styling systematically.
- Josh Comeau's blog (joshwcomeau.com) -- deep, visual explanations of CSS layout, animations, and the mental models behind web rendering.
- The Flutter Widget Catalog (flutter.dev/docs/development/ui/widgets) -- comprehensive visual catalog of every built-in Flutter widget.

**Data visualization:**
- *The Visual Display of Quantitative Information* by Edward Tufte -- the classic. Dense, opinionated, and beautiful. Tufte's core idea: maximize the data-ink ratio (every drop of ink on the page should represent data, not decoration).
- Observable (observablehq.com) -- interactive data visualization notebooks. Great for prototyping charts before committing to a framework.

---

*That's the framework. You don't need to follow it rigidly -- you need to have the vocabulary. The next time you sit down to build a dashboard, spend 20 minutes on the archetype, the data inventory, and a wireframe before you touch code. You'll build the right thing faster, and when you describe what you want to someone else (or to Claude), the result will actually match the picture in your head.*
