# Computer Science Foundations for Startup Founders

*Everything that's actually running under your `pip install` and `git push`.*

You're building a company that trains a robot hand to do sign language. You use GitHub, you run training jobs, you `pip install` things, and it all basically works. But you don't really know what's happening under the hood -- and that's starting to bite. What is a "process"? Why does your training job die when you close your laptop? What does it mean when someone says "just SSH into the box"?

This document is a map of the territory. Not a textbook. Not a course. Just the mental models you need so that the systems you depend on stop feeling like magic.

Three sittings. Maybe a coffee each.

---

## Sitting 1: The Machine Under You

### What is an Operating System?

Your laptop has hardware: a CPU, RAM, a disk, a screen, a network card. But you can't just talk to the hardware directly. You need software that manages all of it -- decides which program gets the CPU right now, which chunk of RAM belongs to which app, and makes sure two programs don't write to the same file at the same time.

That software is the **operating system**. And it's really just three things:

```
┌─────────────────────────────────────────────────────┐
│                  Operating System                    │
│                                                     │
│   ┌─────────────┐                                   │
│   │   Shell      │  ← how you talk to the machine   │
│   └──────┬──────┘                                   │
│          │                                          │
│   ┌──────┴──────┐                                   │
│   │ System Tools │  ← ls, cp, ps, grep, pip, git    │
│   └──────┬──────┘                                   │
│          │                                          │
│   ┌──────┴──────┐                                   │
│   │   Kernel     │  ← the actual boss               │
│   └──────┬──────┘                                   │
│          │                                          │
├──────────┴──────────────────────────────────────────┤
│              Hardware (CPU, RAM, disk, GPU)          │
└─────────────────────────────────────────────────────┘
```

**The kernel** is the core -- it runs with full hardware access and manages everything. **The shell** is how you (the human) issue commands. **System tools** are all the little programs that come pre-installed (`ls`, `cp`, `ps`, `grep`, `chmod`, etc.).

macOS, Linux, and Windows are all operating systems. macOS and Linux are both descendants of **Unix**, which is why they feel so similar at the terminal. Windows took a different path, which is why WSL (Windows Subsystem for Linux) exists -- it's literally a Linux kernel running inside Windows so developers can use Unix tools.

### Unix and Why It Won

In 1969, Ken Thompson and Dennis Ritchie at Bell Labs built an operating system called Unix. It had a few ideas that turned out to be so good that almost everything still uses them today:

**1. Everything is a file.** Your keyboard? File. Your screen? File. A network connection? File. A USB device? File. This sounds weird, but it's genius -- it means the same tools that read and write files can also read from the network, talk to devices, and pipe data between programs. One interface for everything.

**2. Small programs that do one thing well.** Instead of one giant program that does everything, Unix gives you tiny tools: `ls` lists files, `grep` searches text, `sort` sorts lines, `wc` counts words. You combine them.

**3. Pipes.** The `|` character lets you chain small programs together, feeding the output of one into the input of the next:

```bash
# Count how many Python files you have
ls *.py | wc -l

# Find which process is using port 8080
lsof -i :8080 | grep LISTEN

# See your 10 largest files
du -sh * | sort -rh | head -10
```

This is the Unix philosophy. And it's why, in 2026, every cloud server, every Android phone, every Mac, every supercomputer, your robot's embedded computer, and the majority of the internet runs on some descendant of Unix. Windows is the main exception, and even it needed to bolt on a Unix compatibility layer (WSL) because developers kept demanding it.

When you open Terminal on your Mac, you're using a Unix system. When you SSH into a GPU server to run training jobs, you're using a Unix system. The commands are the same everywhere.

### The Kernel: Hardware's Bodyguard

The kernel is the one program on your computer that actually touches the hardware. Every other program -- Chrome, Python, Slack, your training script -- has to *ask* the kernel for permission whenever it wants to do anything real.

Want to read a file? Ask the kernel. Want to send data over the network? Ask the kernel. Want to allocate memory? Ask the kernel.

These requests are called **system calls** (syscalls). There are a few hundred of them, and they're the entire interface between your programs and the hardware:

```
┌─────────────────────────────┐
│  Your Python training script │
│                              │
│  model = load("weights.pt") │  ← "I need to read a file"
│                              │
└──────────────┬───────────────┘
               │
        open() system call
               │
               ▼
┌──────────────────────────────┐
│          Kernel               │
│                               │
│  - Checks: does this process  │
│    have permission to read    │
│    this file?                 │
│  - Finds the file on disk     │
│  - Reads the bytes            │
│  - Hands them back            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Hard Drive / SSD       │
└──────────────────────────────┘
```

Why not let programs talk to hardware directly? Because chaos. If your training script and Slack both tried to write to the same spot on the disk at the same time, you'd get corrupted data. The kernel is the traffic cop. It serializes access, enforces permissions, and keeps programs from stepping on each other.

The kernel also handles:
- **Memory management**: each program thinks it has all the RAM to itself (the kernel creates this illusion using virtual memory)
- **Process scheduling**: deciding which program gets the CPU right now (your machine has maybe 8-16 cores but hundreds of running programs -- the kernel time-slices between them)
- **Device drivers**: talking to your GPU, your WiFi card, your keyboard
- **Networking**: managing TCP connections, routing packets

When people say "Linux," they technically mean the kernel. The kernel is the one piece Linus Torvalds actually wrote. Everything else -- the shell, the tools, the desktop environment -- comes from other projects (mostly the GNU project, which is why purists say "GNU/Linux").

### The Shell: How You Talk to the Machine

The shell is a program that reads text commands you type, figures out what program to run, and shows you the output. That's it. It's a text-based interface to the operating system.

When you open Terminal on your Mac, you're running a shell called **zsh** (Z shell). On most Linux servers, it's **bash** (Bourne Again Shell). They're 95% identical for everyday use.

```
┌──────────────────────────────────┐
│  Terminal.app (or iTerm2)        │  ← the window (just a container)
│  ┌────────────────────────────┐  │
│  │  zsh                       │  │  ← the shell (interprets commands)
│  │                            │  │
│  │  $ ls                      │  │  ← you type a command
│  │  model.py  train.sh  data/ │  │  ← the shell ran `ls` and showed output
│  │  $                         │  │  ← ready for next command
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

Important distinction: **the terminal and the shell are different things.** The terminal (Terminal.app, iTerm2, Windows Terminal) is just the window -- it handles displaying text and capturing your keystrokes. The shell is the program running inside it that actually interprets your commands. You can run different shells in the same terminal. If you type `bash` in your zsh terminal, you've just started a bash shell inside your zsh shell.

When you type `python train.py`, here's what the shell actually does:

1. Reads your input: `python train.py`
2. Splits it into: command = `python`, argument = `train.py`
3. Searches for a program called `python` (using the PATH -- more on this shortly)
4. Finds it at `/usr/bin/python3`
5. Asks the kernel to create a new process running that program
6. Waits for the process to finish (or gives you back control if you use `&`)
7. Shows the prompt again

The shell also gives you scripting -- loops, conditionals, variables. That `train.sh` file on your team's server? That's a shell script. It's just a list of commands the shell runs in sequence:

```bash
#!/bin/bash
echo "Starting training run..."
python train.py --epochs 100 --lr 0.001
echo "Training complete. Uploading model..."
aws s3 cp model.pt s3://our-models/latest.pt
```

**The PATH.** When you type `python`, how does the shell know where the `python` program lives? It checks a list of directories called `$PATH`:

```bash
# See your PATH (it's a colon-separated list of directories)
echo $PATH
# /usr/local/bin:/usr/bin:/bin:/home/conner/.local/bin:...
```

The shell searches these directories in order. When you `pip install` something that has a command-line tool, it goes into one of these directories. When someone says "add it to your PATH," they mean "put the directory containing that program into this list so the shell can find it."

This is why `pip install torch` works (pip is in your PATH) but sometimes a freshly installed tool "can't be found" -- its directory isn't in your PATH yet.

### Processes: What "Running a Program" Actually Means

Every time you run a program, the kernel creates a **process**. A process is a running instance of a program -- it has its own private chunk of memory, its own set of open files, and a unique ID number (PID).

```bash
# See every process running on your machine right now
ps aux

# You'll see hundreds of lines like:
# USER    PID   %CPU  %MEM  COMMAND
# conner  1234  95.2  45.1  python train.py
# conner  5678   0.1   2.3  code
# root       1   0.0   0.1  /sbin/launchd
```

Try running `ps aux` right now. You'll see your shell, your browser (probably dozens of Chrome processes), background services, and maybe a training job. Each one is a separate process with its own PID.

Key things about processes:

**They're isolated.** Your training script can't read Slack's memory. Each process thinks it has the machine to itself. The kernel maintains this illusion. This is why one program crashing doesn't take down the whole machine (usually).

**They have a parent.** Every process is started by another process. When you type `python train.py` in your shell, the shell is the parent and the Python process is the child. Run `ps -ef` and you can see the parent PID (PPID) column -- it's a family tree all the way back to PID 1 (the first process the kernel starts at boot).

**They can be in different states.** Running (actually executing on a CPU), sleeping (waiting for something -- disk I/O, network response, a timer), stopped (paused), or zombie (finished but parent hasn't acknowledged it yet).

**Signals.** You talk to processes using signals. When you hit `Ctrl+C` in the terminal, you're sending a SIGINT (interrupt) signal to the running process, asking it to stop. `Ctrl+Z` sends SIGTSTP, which pauses it. `kill -9 1234` sends SIGKILL to PID 1234, which forces it to die immediately (the process can't catch or ignore this one).

```bash
# Kill that hung training job
kill 1234           # politely ask it to stop (SIGTERM)
kill -9 1234        # force kill (SIGKILL) -- use when it won't stop

# See only your Python processes
ps aux | grep python
```

**Why your training job dies when you close your laptop / SSH session.** When you disconnect from an SSH session, the shell dies. When the shell dies, it sends SIGHUP (hangup) to all its child processes. Your training script is a child of the shell, so it gets killed too.

This is why people use `nohup`, `tmux`, or `screen`:

```bash
# Method 1: nohup ignores the hangup signal
nohup python train.py &

# Method 2: tmux creates a persistent terminal session
tmux new -s training
python train.py
# Press Ctrl+B then D to detach. Close your laptop. Come back later:
tmux attach -s training
# Your training is still running!
```

### Daemons: The Invisible Workers

Some processes aren't tied to any terminal. They start at boot, run in the background forever, and do essential work. These are called **daemons** (pronounced "demons" -- the name comes from Maxwell's demon in physics, not from hell).

Right now, on your machine, daemons are:

- Serving your network connections (the `mDNSResponder` daemon handles DNS)
- Managing your SSH server (the `sshd` daemon listens for incoming connections)
- Running your Docker containers (the `dockerd` daemon)
- Handling system logs (`syslogd`)
- Running scheduled tasks (`cron`)
- Managing your database (`postgres` or `mysqld` if you have one running)

```bash
# See some daemons on your Mac
launchctl list | head -20

# On Linux, use systemd:
systemctl list-units --type=service
```

When you run a training job on a cloud GPU server, the web interface you use (like Vast.ai or Lambda) is talking to daemons on that server -- an SSH daemon that lets you connect, a Docker daemon that manages your containers, a monitoring daemon that reports GPU usage.

The naming convention is a hint: programs that end in `d` are often daemons. `sshd` (SSH daemon), `httpd` (HTTP daemon / web server), `dockerd` (Docker daemon), `mongod` (MongoDB daemon), `systemd` (the master daemon on modern Linux).

**Daemons vs. regular processes**: a daemon has no terminal, runs as a background service, and typically restarts automatically if it crashes. A regular process is usually started by a human in a shell and dies when the shell dies (unless you use nohup/tmux).

This is the whole picture of your machine at any given moment:

```
┌────────────────────────────────────────────────────┐
│                    Your Computer                    │
│                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Terminal │  │ Chrome  │  │ VS Code │  ← apps    │
│  │(shell)  │  │(browser)│  │(editor) │   you see  │
│  └────┬────┘  └─────────┘  └─────────┘            │
│       │                                             │
│  ┌────┴─────┐                                      │
│  │ python   │  ← process you started               │
│  │ train.py │                                      │
│  └──────────┘                                      │
│                                                     │
│  sshd  dockerd  cron  mDNSResponder  ← daemons    │
│  (always running, no terminal, invisible)           │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │              Kernel                          │   │
│  │  manages all of the above                    │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ════════════════════════════════════════════════   │
│               Hardware (CPU, RAM, GPU, disk)        │
└────────────────────────────────────────────────────┘
```

---

## Sitting 2: How Code Runs

### Source Code Is Just Text Files

This sounds obvious but it's worth stating: your `train.py` file is just text. It's not "compiled" into some special format sitting on your disk. Open it in any text editor and you can read it. It's a `.py` file by convention, but the computer doesn't care about the extension -- what matters is what program you use to *run* it.

The same is true of C code (`.c` files), JavaScript (`.js`), HTML (`.html`), and shell scripts (`.sh`). They're all plain text. The difference is in what happens when you want to execute them. And that brings us to the most important distinction in how programming languages work.

### The C Compiler: Text to Machine Code

Your CPU doesn't understand Python. It doesn't understand JavaScript or C either. It understands **machine code** -- a specific set of binary instructions designed for that particular CPU architecture. On your Mac, that's ARM64 instructions. On most cloud servers, that's x86-64 instructions.

A **compiler** translates human-readable source code into machine code *before* you run it. C is the classic compiled language:

```
┌──────────────┐      compiler       ┌──────────────┐
│  hello.c     │  ───────────────→   │  hello       │
│  (text file) │     (gcc/clang)     │  (binary)    │
│              │                     │              │
│  #include... │                     │  01001010... │
│  int main(){ │                     │  10110011... │
│    printf... │                     │  (machine    │
│  }           │                     │   code)      │
└──────────────┘                     └──────────────┘

   You can read this.                  CPU can run this.
   CPU can't run this.                 You can't read this.
```

```bash
# Compile a C program
gcc hello.c -o hello

# Run the resulting binary
./hello
```

After compilation, you have a standalone executable binary. It doesn't need `gcc` installed to run. It doesn't need the original `.c` file. It's a self-contained program that talks directly to the kernel through system calls. This is why compiled programs are fast -- no translation step at runtime.

The kernel itself is written in C. Your Python interpreter is written in C. The `ls` command is written in C. Almost every piece of system software is written in C (or its successor, Rust). When you need maximum performance and direct hardware control, you use a compiled language.

**Why C is "dangerous."** In C, you manage memory yourself. You decide how much memory to allocate, and you're responsible for freeing it when you're done. If you mess up:
- **Memory leak**: you forgot to free memory, so your program slowly eats all the RAM
- **Buffer overflow**: you wrote past the end of your allocated memory, corrupting other data (this is the #1 source of security vulnerabilities in history)
- **Use-after-free**: you freed memory but kept using the pointer, reading garbage data

This is why most application code isn't written in C anymore. The performance gain isn't worth the risk of security vulnerabilities and crashes for most applications. But for kernels, drivers, and performance-critical libraries? C (and increasingly Rust) is still king.

### The Python Interpreter: Text to Bytecode to Virtual Machine

Python takes the opposite approach. There's no separate compilation step. You just run:

```bash
python train.py
```

But Python doesn't hand your text file directly to the CPU. That would be impossible -- the CPU can't execute `model = load("weights.pt")`. Instead, here's what actually happens:

```
┌──────────────┐    compile to     ┌──────────────┐    execute on    ┌──────────────┐
│  train.py    │  ─────────────→   │  train.pyc   │  ────────────→  │   Python     │
│  (text file) │    bytecode       │  (bytecode)  │                 │   Virtual    │
│              │                   │              │                 │   Machine    │
│  model =     │                   │  LOAD_NAME 0 │                 │   (written   │
│   load(...)  │                   │  LOAD_CONST 1│                 │    in C)     │
│              │                   │  CALL 1      │                 │              │
└──────────────┘                   └──────────────┘                 └──────┬───────┘
                                                                          │
                                                                    actual CPU
                                                                    instructions
```

Step 1: Python compiles your `.py` file into **bytecode** -- an intermediate format that's simpler than Python but not machine code. (Those `.pyc` files in `__pycache__/` directories? That's cached bytecode.)

Step 2: The **Python virtual machine** (PVM) reads the bytecode instructions one at a time and executes them. The PVM itself is a C program -- so ultimately, C code is running on your CPU, interpreting your Python instructions.

This is why Python is slow. Every Python instruction goes through an intermediary (the VM) instead of running directly on the CPU. For a simple loop that runs a million times, C does a million CPU operations. Python does a million "read the next bytecode, figure out what it means, do the thing" cycles -- easily 10-100x slower.

**But wait -- your training jobs use Python and they're fast?**

Because you're not actually doing the heavy computation in Python. When you call `model.forward(x)` in PyTorch, the Python code immediately hands off to a C++ library (libtorch) which talks to a CUDA library (written in C) which runs on your GPU. Python is the conductor; C/C++ is the orchestra:

```
┌─────────────────────────────────────────────────────┐
│  Python  (slow, but easy to write)                  │
│                                                     │
│  loss = model(input)     ← one line of Python       │
│         │                                           │
│         ▼                                           │
│  ┌─────────────────────────────────┐                │
│  │  PyTorch C++ core (libtorch)    │  ← fast        │
│  │  matrix multiplications, etc.   │                │
│  └──────────────┬──────────────────┘                │
│                 ▼                                    │
│  ┌─────────────────────────────────┐                │
│  │  CUDA / cuDNN (C code on GPU)   │  ← very fast   │
│  │  actual number crunching        │                │
│  └─────────────────────────────────┘                │
└─────────────────────────────────────────────────────┘
```

This is why `pip install numpy` downloads a 20MB binary -- it's not pure Python. It's a thin Python layer on top of heavily optimized C and Fortran code. Same for PyTorch, TensorFlow, OpenCV, and every other performance-critical Python library. Python is the glue language. The real work happens in compiled code underneath.

This pattern -- "easy language on top, fast compiled language underneath" -- is so common it has a name: the **two-language problem**. You get Python's convenience for writing application logic and C's speed for the heavy lifting.

### JIT Compilation: The Best of Both Worlds

There's a third approach that splits the difference: **Just-In-Time (JIT) compilation**. Instead of compiling everything upfront (C) or interpreting everything at runtime (Python), a JIT compiler watches your program run, identifies the hot spots (code that runs many times), and compiles *those parts* to machine code on the fly.

The most important JIT in your life is **V8** -- the JavaScript engine in Chrome and Node.js. JavaScript was originally an interpreted language like Python. Then Google built V8, which JIT-compiles JavaScript to native machine code, making it 10-50x faster than the old interpreters. This is why modern web apps can run complex simulations, 3D games, and video editing in the browser -- the JavaScript is being compiled to near-native speed while it runs.

Another JIT you might encounter: **`torch.compile()`**. If you add one line to your PyTorch code:

```python
model = torch.compile(model)
```

PyTorch will trace your model's operations and JIT-compile them into optimized GPU code, often giving you 20-50% speedup for free. Same idea as V8 -- watch what the code does, then compile an optimized version.

Java also uses a JIT (the HotSpot VM). The pattern is everywhere.

Here's the spectrum:

```
Pure Interpretation           JIT Compilation             Ahead-of-Time Compilation
(Python, Ruby)               (JavaScript/V8, Java,       (C, C++, Rust, Go)
                              torch.compile)
Slow execution               Fast execution               Fastest execution
Instant startup              Medium startup               Must compile first
Easy debugging               Medium debugging             Hard debugging
Very flexible                Flexible                     Less flexible

             ←── slower at runtime ────── faster at runtime ────→
             ←── easier to develop ────── harder to develop ────→
```

### Why This Matters for You

When you `pip install` a package and it says "building wheel" and takes five minutes -- it's compiling C/C++ code. If it fails with some cryptic error about `gcc` or `cmake`, now you know: it's trying to compile C code and something is wrong with your C build tools.

When your training job uses 100% GPU but Python profiling shows it's "idle" -- that's correct. Python dispatched the work to C/CUDA code on the GPU and is just waiting for the result.

When someone says "use Cython" or "write a C extension" -- they mean: take the slow Python bottleneck and rewrite it in C, then call it from Python. Same two-language trick that NumPy and PyTorch use.

When you hear "WebAssembly" (Wasm) -- it's compiled code (from C, Rust, Go, etc.) that runs in the browser alongside JavaScript. It's how apps like Figma get near-native performance in a web browser.

---

## Sitting 3: The Network

### TCP/IP: How Data Gets from A to B

When your laptop sends data to a server (or vice versa), it's not one monolithic system. It's a stack of layers, each solving one problem:

```
┌──────────────────────────────────────────────┐
│  Application Layer (HTTP, SMTP, SSH, DNS)    │  ← "what" you're saying
├──────────────────────────────────────────────┤
│  Transport Layer (TCP or UDP)                │  ← "reliably" or "fast"
├──────────────────────────────────────────────┤
│  Internet Layer (IP)                         │  ← "where" it's going
├──────────────────────────────────────────────┤
│  Link Layer (WiFi, Ethernet)                 │  ← "how" it physically moves
└──────────────────────────────────────────────┘
```

**IP (Internet Protocol)** handles addressing. Every device on the internet has an IP address (like `142.250.80.46` for Google, or `192.168.1.5` for your laptop on your home network). IP takes a chunk of data, slaps a destination address on it, and figures out how to route it there -- through your router, to your ISP, across the internet backbone, to the destination's network.

But IP alone is unreliable. Packets can arrive out of order, get duplicated, or just vanish. This is where TCP comes in.

**TCP (Transmission Control Protocol)** adds reliability on top of IP:
- **Connection**: before sending data, both sides do a "handshake" to establish a connection
- **Ordering**: packets are numbered so the receiver can reassemble them in the right order
- **Acknowledgment**: the receiver confirms every packet; if the sender doesn't hear back, it resends
- **Flow control**: the sender slows down if the receiver is overwhelmed

Think of IP as the postal system (it knows addresses and routes) and TCP as registered mail (guaranteed delivery, signature required, in-order).

```
Your laptop                                      GPU server
┌──────────┐                                    ┌──────────┐
│          │ ── SYN ──────────────────────────→ │          │
│          │ ←── SYN-ACK ─────────────────────  │          │
│          │ ── ACK ──────────────────────────→ │          │
│          │                                    │          │
│          │    (connection established)         │          │
│          │                                    │          │
│          │ ── data packet 1 ───────────────→  │          │
│          │ ── data packet 2 ───────────────→  │          │
│          │ ←── ACK 1 ──────────────────────── │          │
│          │ ←── ACK 2 ──────────────────────── │          │
│          │                                    │          │
│          │ ── FIN ──────────────────────────→ │          │
│          │ ←── FIN-ACK ─────────────────────  │          │
└──────────┘                                    └──────────┘
```

**UDP** is the alternative to TCP: no connection, no ordering, no acknowledgment. Just fire packets and hope they arrive. It's faster (less overhead) but unreliable. Used for video calls, gaming, and DNS lookups -- situations where speed matters more than guaranteed delivery, and a dropped packet here or there is fine.

**Ports.** One server can run many network services. Ports are how you tell them apart. An IP address gets you to the machine; the port gets you to the right program on that machine. Like a building address vs. an apartment number.

```
Your server at 54.23.100.15
├── Port 22:    SSH daemon (remote login)
├── Port 80:    Web server (HTTP)
├── Port 443:   Web server (HTTPS)
├── Port 5432:  PostgreSQL database
└── Port 8080:  Your training dashboard
```

When you SSH into a server, you're making a TCP connection to port 22. When you open a website, your browser connects to port 443 (HTTPS). When your training script downloads a dataset, it's TCP connections to port 443 on wherever the data is hosted.

### HTTP: The Language of the Web

HTTP (HyperText Transfer Protocol) sits on top of TCP. It's the protocol your browser speaks. It's also the protocol that almost every API uses. It's simple: the client sends a **request**, the server sends back a **response**.

```
Request:
┌─────────────────────────────────┐
│ GET /api/models/latest HTTP/1.1 │  ← method + path + version
│ Host: api.yourcompany.com       │  ← headers (metadata)
│ Authorization: Bearer sk-abc... │
│                                 │
│ (no body for GET requests)      │
└─────────────────────────────────┘

Response:
┌─────────────────────────────────┐
│ HTTP/1.1 200 OK                 │  ← status code
│ Content-Type: application/json  │  ← headers
│                                 │
│ {"model": "v2.3", "accuracy":   │  ← body (the actual data)
│  0.94, "size_mb": 340}          │
└─────────────────────────────────┘
```

The status codes you'll see most often:
- **200** OK -- everything worked
- **201** Created -- new resource created (after a POST)
- **301/302** Redirect -- go look over there instead
- **400** Bad Request -- you sent something the server can't understand
- **401** Unauthorized -- you need to log in
- **403** Forbidden -- you're logged in but don't have permission
- **404** Not Found -- that URL doesn't exist
- **500** Internal Server Error -- the server broke

**HTTPS** is just HTTP wrapped in encryption (TLS). The data is the same, but it's encrypted in transit so nobody between you and the server (your ISP, a hacker on the coffee shop WiFi) can read it. That little padlock in your browser means HTTPS.

When you do `pip install torch`, pip makes HTTPS requests to `pypi.org` to download the package. When you `git push`, Git makes HTTPS requests to `github.com` (unless you're using SSH). When your robot hand's software calls your API, it's making HTTPS requests. HTTP is everywhere.

You can make HTTP requests from the command line with `curl`:

```bash
# GET request (fetch data)
curl https://api.github.com/users/torvalds

# POST request (send data)
curl -X POST https://api.example.com/data \
  -H "Content-Type: application/json" \
  -d '{"name": "test"}'
```

### SMTP: How Email Actually Works

Email seems simple but it's one of the oldest and weirdest internet protocols. SMTP (Simple Mail Transfer Protocol) was designed in 1982 and hasn't changed much since.

When you send an email, here's what happens:

```
You (conner@yourstartup.com)              Friend (alex@gmail.com)

┌──────────────┐
│ Your email   │
│ client       │
│ (Gmail web)  │
└──────┬───────┘
       │ SMTP
       ▼
┌──────────────┐     SMTP      ┌──────────────┐
│ Your startup's│─────────────→│ Gmail's       │
│ mail server   │              │ mail server   │
│ (outgoing)    │              │ (incoming)    │
└──────────────┘              └──────┬───────┘
                                     │
                              ┌──────┴───────┐
                              │ Alex's inbox  │
                              │ (stored on    │
                              │  Gmail server)│
                              └──────────────┘
```

Your mail client talks to your outgoing mail server via SMTP. Your mail server looks up where `gmail.com` mail should go (using DNS -- specifically, MX records), then forwards the email to Gmail's mail server via SMTP. Gmail stores it. When Alex opens Gmail, their client fetches it using a different protocol (IMAP or POP3).

Why does this matter for you? Because when your startup sends transactional emails (password resets, notifications, signup confirmations), you're using SMTP under the hood. Services like SendGrid, Mailgun, or AWS SES are just SMTP servers that you send email through via their API. Understanding this helps when emails end up in spam (your SMTP setup is misconfigured) or when you need to debug delivery issues.

Also: SMTP is why email spam exists. The protocol was designed in a trusting era -- there's no built-in authentication that the sender is who they claim to be. Modern additions (SPF, DKIM, DMARC) bolt on sender verification, but the fundamental "anyone can send email claiming to be anyone" design remains. It's patched, not fixed.

### DNS: The Phonebook of the Internet

When you type `github.com` in your browser, your computer doesn't know where GitHub is. It knows how to talk to IP addresses (like `140.82.112.4`), not names. DNS (Domain Name System) translates names to IP addresses.

```
You type: github.com
                │
                ▼
┌──────────────────────┐
│ Your computer's      │
│ DNS resolver         │  "Do I already know github.com? No."
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Your ISP's DNS       │
│ server               │  "Let me look that up..."
└──────────┬───────────┘
           │  (if it doesn't know, it asks the root servers,
           │   then the .com servers, then github's servers)
           │
           ▼
Answer: 140.82.112.4
```

Your computer caches the answer so it doesn't have to look it up every time. That's why DNS changes (like pointing your domain to a new server) take a while to "propagate" -- everyone's caches have to expire first.

```bash
# Look up a domain's IP address
dig github.com

# Or the simpler version
nslookup github.com
```

When someone says "DNS isn't propagated yet" after changing their domain settings, this is what they mean -- the old IP address is still cached in DNS servers around the world.

### The Browser: Almost an Operating System

Here's something that might surprise you: your web browser is one of the most complex pieces of software on your computer. Chrome has more lines of code than the Linux kernel. And when you look at what it does, it starts looking less like an "app" and more like an operating system:

```
┌─────────────────────────────────────────────────────────┐
│                        CHROME                            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  JavaScript Engine (V8)                           │   │
│  │  - JIT compiles JS to machine code               │   │
│  │  - Runs your web app's logic                     │   │
│  │  - This is the same engine Node.js uses          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Rendering Engine (Blink)                         │   │
│  │  - Parses HTML/CSS                               │   │
│  │  - Layouts the page                              │   │
│  │  - GPU-accelerated painting                      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Process Model                                    │   │
│  │  - Each tab = separate process                   │   │
│  │  - One tab crashing won't kill others            │   │
│  │  - Sound familiar? (It's what the kernel does)   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Storage                                          │   │
│  │  - Cookies (small key-value pairs)               │   │
│  │  - localStorage (bigger key-value store)         │   │
│  │  - IndexedDB (actual database in the browser)    │   │
│  │  - Cache API (offline-capable caching)           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Networking                                       │   │
│  │  - HTTP/HTTPS client                             │   │
│  │  - WebSockets (persistent connections)           │   │
│  │  - WebRTC (peer-to-peer, used for video calls)   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Security Sandbox                                 │   │
│  │  - Same-origin policy (a page can't read data    │   │
│  │    from a different domain)                      │   │
│  │  - Each tab runs in an isolated process          │   │
│  │  - JS can't access your filesystem               │   │
│  │  - Permissions system (camera, mic, location)    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  APIs                                             │   │
│  │  - Canvas (2D drawing)                           │   │
│  │  - WebGL/WebGPU (3D/GPU compute)                 │   │
│  │  - Web Audio, Web Bluetooth, Gamepad API         │   │
│  │  - Service Workers (background tasks)            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Look at that list. A JavaScript runtime (JIT compiler). A process model with isolation. Storage systems. A networking stack. A security sandbox with permissions. GPU access. Background workers.

That's an operating system. The browser has become a platform -- an OS within your OS. Web apps like Figma, Google Docs, and Slack run inside this platform, and they're indistinguishable from "native" apps for most users.

**Why this happened**: because the browser is the only runtime that's truly cross-platform. Write a web app once and it runs on Windows, Mac, Linux, Android, iOS, and ChromeOS. No app store approval needed. No installation. Just a URL. That's an incredibly powerful distribution model, and it's why so much software development has moved to the web.

**Why this matters to you**: your startup's dashboard, your training monitoring UI, your demo for investors -- it's probably a web app running in a browser. The browser is doing a *lot* of work to make that possible. And tools like Electron (which Slack, VS Code, and Discord are built on) literally ship a whole Chrome browser bundled with your app. VS Code is a web app in a trenchcoat.

### Putting It All Together

Let's trace what happens when you type `github.com` into your browser and hit Enter. Every layer we've discussed is involved:

```
1. DNS lookup
   Browser → DNS server: "What's the IP for github.com?"
   DNS server → Browser: "140.82.112.4"

2. TCP connection
   Browser → 140.82.112.4:443 : SYN
   Server → Browser: SYN-ACK
   Browser → Server: ACK
   (three-way handshake, connection established)

3. TLS handshake (the S in HTTPS)
   Browser and server negotiate encryption keys
   (now everything is encrypted)

4. HTTP request
   Browser → Server:
   GET / HTTP/1.1
   Host: github.com
   (plus cookies, auth tokens, etc.)

5. Server processing
   GitHub's web server (a process, running on Linux, in a container)
   receives the request, queries databases, builds HTML

6. HTTP response
   Server → Browser:
   HTTP/1.1 200 OK
   Content-Type: text/html
   (plus the HTML, CSS, JavaScript for the page)

7. Browser rendering
   - Parses the HTML (builds a DOM tree)
   - Parses the CSS (figures out how things look)
   - Downloads additional resources (images, scripts, fonts)
   - Executes JavaScript (which may fetch more data via API calls)
   - Paints pixels to the screen using the GPU

8. JavaScript execution
   - GitHub's JS runs in the V8 engine
   - Sets up event listeners (what happens when you click things)
   - May open WebSocket connections for real-time updates
   - Stores data in localStorage/cookies
```

All of that happens in about 1-2 seconds. DNS lookup (UDP). TCP connection (transport layer). TLS encryption (security). HTTP request/response (application layer). Server-side processing (kernel, processes, potentially containers). Browser rendering (HTML/CSS parser, layout engine, GPU). JavaScript execution (V8 JIT compiler).

Every concept in this document played a role.

### A Map of Where You Stand

Now you can place things on the map. When someone says:

- **"Just SSH into the box"** -- they mean: use the SSH protocol (which runs over TCP, port 22) to get a remote shell on a Linux server. You'll be talking to the kernel of that machine through a shell, just like you do locally.

- **"The Docker daemon isn't running"** -- a daemon (background process) called `dockerd` manages your containers. It's not running, so nothing container-related will work. Start it.

- **"Your pip install is failing because it can't compile the C extension"** -- the Python package has C code under the hood. The compilation step (turning C into machine code) failed, probably because you're missing build tools. Install `gcc` or `build-essential`.

- **"Check if port 8080 is already in use"** -- another process already has a TCP socket listening on port 8080. Your program can't bind to the same port. Find the other process (`lsof -i :8080`) and kill it, or use a different port.

- **"The API is returning 503"** -- HTTP status code 503 means "Service Unavailable." The server is overwhelmed or down. Your code is fine; the problem is on the other end.

- **"We need to set up DNS for the new domain"** -- create DNS records that map your domain name to your server's IP address so people can reach your site by name instead of IP.

- **"The training job got OOM-killed"** -- the kernel's out-of-memory killer terminated your process because it was using too much RAM. The kernel chose to sacrifice your process to keep the system alive. Use less memory, get more RAM, or use gradient checkpointing.

---

## Further Reading

These are the best resources for going deeper on each topic. All of them are worth your time.

**The best starting point:**
- *Code: The Hidden Language of Computer Hardware and Software* by Charles Petzold -- starts from first principles (what is a bit? what is a gate?) and builds up to a complete working computer. If you read one book, make it this one.

**Unix and Linux:**
- *How Linux Works* by Brian Ward -- exactly what it sounds like. Covers the boot process, the kernel, filesystems, networking, and more. Practical, not academic.
- Julia Evans' zines (https://wizardzines.com) -- bite-sized, illustrated explanations of Unix tools, networking, and systems. "Bite Size Linux", "Bite Size Networking", and "How DNS Works" are particularly relevant.
- *The Linux Command Line* by William Shotts (free online) -- if you want to get genuinely good at the shell.

**Networking:**
- *Computer Networking: A Top-Down Approach* by Kurose & Ross -- the classic textbook, but the first 4 chapters alone give you solid TCP/IP understanding.
- Julia Evans' "Bite Size Networking" zine -- covers DNS, TCP, HTTP, and more in her visual style.
- Beej's Guide to Network Programming (https://beej.us/guide/bgnet/) -- free, online, surprisingly readable for a sockets tutorial.

**How computers actually work (hardware up):**
- Ben Eater's YouTube channel -- he builds a working computer on breadboards. Watching him construct a CPU from logic gates is the single best way to demystify what "hardware" means.
- *Computer Systems: A Programmer's Perspective* (CS:APP) by Bryant & O'Hallaron -- the book CMU uses for their systems course. Dense but transformative. Read chapters 1, 3, and 7-12 if you want to understand compilation, linking, and processes.

**Compilers and interpreters:**
- *Crafting Interpreters* by Robert Nystrom (free online at craftinginterpreters.com) -- build a programming language from scratch. Written beautifully. You don't need to do the exercises; just reading it builds deep understanding.

**The browser as a platform:**
- *Web Browser Engineering* by Pavel Panchekha and Chris Harrelson (free online at browser.engineering) -- build a web browser from scratch in Python. Shows you exactly what happens between "receive HTML" and "pixels on screen."

**For when you're ready to write systems code:**
- *The Rust Programming Language* (free online, "the Rust Book") -- Rust gives you C-level performance with memory safety. If you ever need to write fast, low-level code, Rust is the modern choice.

---

*That's the map. You don't need to master all of this -- you need to know it exists, roughly how it works, and where to look when something breaks. The next time a build fails, a process dies, or a network request times out, you'll know which layer to investigate. And that's the difference between debugging and guessing.*
