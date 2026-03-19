# Diffusion Models: Creating Images from Noise

*How do you teach a machine to dream? By teaching it to clean up messes.*

You build things. You've used AI image generators -- Midjourney, DALL-E, Stable Diffusion. You've typed a sentence and watched a picture appear. You've probably wondered, at least once, how that actually works. Not "it uses AI" -- you know that already -- but *how*. What is the machine doing, step by step, to turn "a cat riding a bicycle through Tokyo at sunset" into a picture?

By the end of this guide, you'll understand the core mechanism behind diffusion models -- the technology that powers essentially all modern image generation. You'll know why they're called "diffusion" models, what the noise is about, why the process runs backwards, and how text controls the output. You'll also understand img2img, which is how you use a diffusion model to *modify* an existing image -- and why that turns out to be a powerful attack vector against digital watermarks.

No homework. No exams. Just understanding.

---

## Part 1: What's the Problem?

### Generating Images Is Hard

Here is a question that sounds simple and is not: how do you write a program that creates realistic, novel images?

Think about what that requires. A 512×512 color image has 786,432 numbers (512 × 512 × 3 color channels). Each number is a pixel intensity between 0 and 255. So an image is a point in a space with 786,432 dimensions, where each dimension has 256 possible values. The number of possible images is roughly 256^786,432, which is a number with about 1.9 million digits. That's not just a big number -- it's offensively, incomprehensibly large.

But here's the thing: almost none of those possible images look like anything. The overwhelming majority look like television static. The set of images that look like photographs, paintings, or anything a human would recognize as meaningful is a vanishingly thin surface in that vast space.

Generating realistic images means learning to find that thin surface. You need to understand the structure of "images that look like things" well enough to produce new ones on demand.

This is called the **generative modeling** problem, and for decades it was one of the hardest problems in machine learning.

### A Brief History of Trying

People have been attempting this since the 1960s, with limited success. Let's skip ahead to the approaches that actually started to work.

**GANs (2014).** Generative Adversarial Networks took a game-theoretic approach: train two networks against each other, a generator that tries to create fake images and a discriminator that tries to spot them. The generator gets better at faking, the discriminator gets better at detecting, and in theory they push each other to perfection. In practice, GANs produced stunning results -- the first truly photorealistic AI faces -- but were notoriously difficult to train. They'd collapse, oscillate, or produce artifacts. Training a GAN felt less like engineering and more like alchemy.

**VAEs (2013).** Variational Autoencoders took a probabilistic approach: learn a compressed representation of images (a latent space) and then sample from it. They were more stable than GANs but produced blurry outputs. The images were recognizable but looked like they'd been smeared with Vaseline.

**Autoregressive models.** Treat the image as a sequence of pixels and predict each one from the previous ones, like a language model for pixels. Worked, but painfully slow -- you had to generate one pixel at a time.

Then, around 2020, diffusion models arrived and blew everything else away. Better image quality than GANs, more stable training, and a mathematically principled framework that people could actually reason about.

What was the trick?

### The Trick: Destruction Is Easy

Here is the core insight behind diffusion models, and it's almost embarrassingly simple.

**Destroying an image is easy. Creating an image is hard. But if you learn to reverse each small step of destruction, you can create images by running the destruction process backwards.**

That's it. That's the whole idea. Everything else is details.

Let's make this concrete. Take a photograph. Add a tiny amount of random noise to it -- just a slight fuzz, barely visible. You now have a slightly degraded image. Add a tiny bit more noise. And more. And more. Keep going for a thousand steps. By the end, the photograph has been completely obliterated. It's pure television static. There is no trace of the original image left.

```
Step 0          Step 250        Step 500        Step 750        Step 1000
[photograph] → [a bit fuzzy] → [quite noisy] → [mostly noise] → [pure static]
     ↓              ↓               ↓               ↓               ↓
 100% signal    75% signal      50% signal      25% signal       0% signal
  0% noise      25% noise       50% noise       75% noise       100% noise
```

This forward process -- adding noise step by step until the image is destroyed -- requires no intelligence whatsoever. It's just math. Specifically, it's adding Gaussian noise according to a predefined schedule. A first-year undergrad could implement it.

Now imagine you could reverse this process. Imagine you had a machine that could look at a noisy image and remove just a little bit of noise -- not all of it, just enough to take one step back toward the clean image. If you could do that, you could start with pure random static and apply this machine a thousand times, each time removing a tiny bit of noise, until you arrived at a clean, realistic image.

The forward process is trivial. The reverse process is where the learning happens. And it turns out that learning to reverse one tiny step of noise addition is a much more tractable problem than learning to generate images from scratch.

---

## Part 2: The Forward Process -- Adding Noise

### Gaussian Noise: The Universal Solvent

Before we get into how the model learns to denoise, let's understand the noise itself.

The noise used in diffusion models is **Gaussian noise** -- random values drawn from a bell curve (normal distribution). Each pixel gets an independent random nudge. Some pixels get pushed brighter, some darker, and the magnitude of the nudge follows the familiar bell-shaped distribution: small nudges are common, large nudges are rare.

Why Gaussian? Three reasons, and they're all good ones.

**Mathematical convenience.** Gaussians have beautiful mathematical properties. If you add two Gaussian random variables, you get another Gaussian. If you add a thousand Gaussian noise steps, the result is still Gaussian -- you just need to track the total variance. This means you can skip ahead: if you want to know what the image looks like at step 500, you don't need to simulate all 500 steps. You can jump directly there with a single formula.

**Physical naturalness.** By the central limit theorem, any process that involves many small random perturbations tends toward a Gaussian distribution. Camera sensor noise is Gaussian. Thermal fluctuations are Gaussian. If you're modeling "generic randomness," Gaussian is the natural choice.

**The connection to diffusion.** This is where the name comes from. In physics, diffusion is the process by which particles spread out from a region of high concentration to low concentration -- think of a drop of ink in water. Mathematically, diffusion is described by a stochastic differential equation driven by Gaussian noise. The forward process in a diffusion model is literally a discretized version of physical diffusion in pixel space. The image "diffuses" into noise the same way ink diffuses into water.

### The Noise Schedule

Not all steps add the same amount of noise. The progression from clean image to pure noise follows a **noise schedule** -- a predefined plan for how much noise to add at each step.

A typical schedule starts gently and accelerates. In the early steps, you add very little noise, preserving most of the image's structure. In the later steps, you add noise more aggressively, until the image is completely overwhelmed.

Why not add noise uniformly? Because the early steps and the late steps are doing fundamentally different things. The early steps preserve large-scale structure (is this a landscape or a portrait? are there objects on the left or the right?) while destroying fine details. The late steps destroy even the coarse structure. A good noise schedule respects this distinction.

Formally, if we call the original image **x_0** and the image at step *t* is **x_t**, then:

**x_t = √(α_t) · x_0 + √(1 - α_t) · ε**

where **ε** is pure Gaussian noise and **α_t** is a number that decreases from 1 toward 0 as *t* goes from 0 to *T*. When *t* = 0, you get the original image. When *t* = *T*, you get pure noise.

The key thing to notice: this formula lets you jump to *any* step directly. You don't need to simulate all the intermediate steps. Given the original image and a sample of noise, you can compute the noisy version at any point in the schedule.

This is important for training, as we'll see next.

### What the Noise Destroys

Different amounts of noise destroy different kinds of information, and this is worth thinking about carefully.

A little noise (early steps) destroys:
- Fine texture and subtle gradients
- Exact pixel values
- High-frequency details (sharp edges become slightly soft)

Moderate noise (middle steps) destroys:
- Object boundaries
- Color accuracy
- Small objects and fine structure

Heavy noise (late steps) destroys:
- Spatial composition (where things are in the image)
- Color palette
- Whether there are objects at all

At the very end, all information is gone. The noisy image is statistically indistinguishable from random noise -- no matter what photograph you started with, you end up at the same place.

```
Noise level:     Low              Medium            High             Total
What remains:    Overall shape,   Blobs of color,   Vague hints      Nothing.
                 colors, layout   rough positions   of... something  Pure noise.
```

This hierarchy -- fine details die first, coarse structure dies last -- is not a coincidence. It's a consequence of signal processing theory. High-frequency components of an image have less energy per pixel than low-frequency components, so they're overwhelmed by noise first. This is the same reason you can still recognize a friend's face through a dirty window but can't read the text on their shirt.

---

## Part 3: The Reverse Process -- Learning to Denoise

### The Neural Network's Job

Here is the question the neural network must answer:

*Given an image that has been corrupted by a known amount of noise, predict what one step of denoising looks like.*

Or, equivalently (and this is the version that works better in practice):

*Given a noisy image and a timestep t, predict the noise that was added.*

These are equivalent because if you know the noise, you can subtract it to get a cleaner image. But predicting the noise is a more stable learning target, and it's what most diffusion models actually do.

### Training: The Beautiful Simplicity

The training procedure for a diffusion model is almost absurdly simple. Here it is, in its entirety:

1. Take a clean training image **x_0**
2. Pick a random timestep *t* between 1 and T
3. Sample random noise **ε** from a Gaussian distribution
4. Compute the noisy image: **x_t = √(α_t) · x_0 + √(1 - α_t) · ε**
5. Ask the neural network to predict **ε** given **x_t** and *t*
6. Compute the loss: how different was the prediction from the actual noise?
7. Update the network's weights to reduce the loss
8. Repeat

That's it. No adversarial games (like GANs). No complex variational bounds (like VAEs). Just: add noise, predict the noise, get better at predicting the noise.

The loss function is just mean squared error: **||ε - ε_predicted||²**. The simplest possible loss function for a regression problem.

This simplicity is part of why diffusion models won. GANs required delicate balancing of two competing networks. VAEs required a complex loss function balancing reconstruction and regularization. Diffusion models require... predicting noise. Any competent ML engineer can implement the training loop in an afternoon.

### The U-Net: What Predicts the Noise

The neural network architecture used in most diffusion models is called a **U-Net**, and it's worth understanding why.

A U-Net is a convolutional neural network shaped like the letter U. It has an encoder path (going down) that progressively compresses the spatial dimensions while increasing the number of channels, a bottleneck at the bottom, and a decoder path (going up) that progressively expands back to the original resolution. Crucially, there are **skip connections** that link each encoder level to the corresponding decoder level.

```
Input (512×512)  ─────────────────────────────→  Output (512×512)
    ↓                                                ↑
  [256×256] ────────────────────────────────→ [256×256]
      ↓                                          ↑
    [128×128] ──────────────────────────→ [128×128]
        ↓                                    ↑
      [64×64] ────────────────────→ [64×64]
          ↓            ↑
        [32×32] ─── [32×32]         (skip connections: ───→)
              bottleneck
```

Why this shape? Because denoising requires understanding at multiple scales simultaneously. To remove noise from a pixel, you need both:

- **Local context** (what are the neighboring pixels doing? is this an edge, a flat region, a texture?) -- handled by the early encoder layers and late decoder layers
- **Global context** (is this part of a face? a sky? what's the overall composition?) -- handled by the deep, compressed layers near the bottleneck

The skip connections are critical. Without them, the network would have to reconstruct fine-grained spatial details entirely from the compressed bottleneck representation, which loses too much information. The skip connections let the decoder access the original spatial details directly, while the bottleneck provides global context.

### The Timestep Embedding

There's one more input to the network: the timestep *t*.

This matters because denoising at step 100 (light noise, most structure intact) is a fundamentally different task from denoising at step 900 (heavy noise, barely any structure left). The network needs to know *how noisy* the image is so it can calibrate its denoising accordingly.

At early timesteps, the network should focus on recovering fine details -- textures, sharp edges, subtle color gradients. At late timesteps, it should focus on coarse structure -- is there a face? where are the eyes?

The timestep is encoded as a vector (using sinusoidal positional encodings, borrowed from transformers) and injected into the U-Net at every level. It modulates the network's behavior like a volume knob, telling it how aggressive to be.

### Sampling: Running It Backwards

Once the network is trained, generating an image works like this:

1. Start with pure random noise: **x_T ~ N(0, I)**
2. For *t* = *T*, *T*-1, *T*-2, ..., 1:
   - Feed **x_t** and *t* to the trained network
   - Network predicts the noise **ε_predicted**
   - Compute **x_{t-1}** by removing a bit of the predicted noise and adding a small amount of fresh noise (for stochasticity)
3. Return **x_0**: the generated image

The "adding a small amount of fresh noise" in step 2 might seem strange. Why add noise when you're trying to remove it? Two reasons.

First, it maintains the correct probability distribution. The reverse process is itself stochastic -- the denoising step doesn't have a single correct answer. Given a noisy image, many different cleaner images could have produced it. The added noise keeps the process exploring the space of possibilities rather than collapsing to a single, potentially wrong, answer.

Second, it improves sample quality. Without the stochastic term, the process tends to produce "average" images -- blurry compromises between multiple possibilities. The noise injects diversity and sharpness.

### An Analogy: The Art Restorer

Imagine an art restorer who specializes in recovering damaged paintings. She has spent years studying thousands of paintings at every stage of degradation -- from pristine to completely destroyed.

You hand her a canvas that looks like almost nothing -- just vague blotches of color. She squints at it and says: "Given the distribution of colors and the very faint structural hints, this was probably a landscape. The blue in the upper portion suggests sky. I'll make a small step toward what I think it should look like."

She makes one pass, and now instead of random blotches, you can see... something. Maybe a horizon line. Maybe a hint of green below and blue above.

You hand it back. She looks again, this time with more to work with: "Okay, now I can see it's a landscape with hills. The green shapes suggest trees. The light suggests late afternoon." Another pass.

Each time, she works with the result of her previous restoration, and each time she can see more clearly what the painting is supposed to be. After a thousand passes, you have a complete, coherent painting.

She never saw this specific painting before. But she's seen so many paintings at every stage of damage that she knows what "one step less damaged" looks like for any given state. That knowledge, accumulated over thousands of examples, is what the diffusion model learns.

### Why Does This Actually Work?

It's worth pausing to ask: why should this work at all? Why should a neural network trained to remove tiny amounts of noise be able to *generate* entirely new images?

The answer lies in what the network actually learns. When it predicts noise, it's implicitly learning the **structure of natural images**. To predict what noise was added to a slightly noisy face, you need to know what faces look like. To predict noise added to a slightly noisy landscape, you need to know what landscapes look like. The noise prediction task forces the network to build a comprehensive internal model of "what images should look like."

There's a deeper mathematical answer too. The noise prediction network is implicitly learning a quantity called the **score function** -- the gradient of the log probability density of images. In plain English: at any point in the space of possible images, the score function tells you "which direction leads toward more probable (more realistic) images."

Think of it like this. Imagine you're standing on a mountainous landscape shrouded in thick fog. You can't see the peaks, but you can feel the slope of the ground beneath your feet. The slope tells you which direction is uphill. If you keep walking uphill, you'll reach a peak.

The score function is the slope. The peaks are the realistic images. And the diffusion process is the walk uphill, one step at a time, from the flatlands of random noise toward the peaks of photorealistic images.

```
Probability of
being a real      ▲
image             │    *
                  │   * *      *
                  │  *   *    * *
                  │ *     *  *   *    peaks = real images
                  │*       **     *
                  ├─────────────────→  image space
                  ↑ the score function points
                    toward the nearest peak
```

---

## Part 4: Conditioning -- Making It Listen

### The Text-to-Image Problem

So far, we've described an unconditional diffusion model: one that generates random images from the training distribution. That's scientifically interesting but not very useful. You don't want random images -- you want "a cat riding a bicycle through Tokyo at sunset."

The question is: how do you get the diffusion process to follow instructions?

### CLIP: Connecting Words to Images

The first piece of the puzzle is **CLIP** (Contrastive Language-Image Pre-training), developed by OpenAI in 2021.

CLIP is a pair of neural networks trained together: one that processes images and one that processes text. They're trained on hundreds of millions of image-caption pairs from the internet, with a simple objective: make the image embedding and the text embedding similar when the caption describes the image, and dissimilar when it doesn't.

After training, CLIP gives you a shared vector space where images and text live together. The phrase "a golden retriever on a beach" and a photo of a golden retriever on a beach end up as nearby vectors. "A thunderstorm over mountains" ends up near photos of thunderstorms over mountains.

This is remarkable. CLIP doesn't translate images to words or words to images. It creates a shared space of *meaning* where both modalities coexist. The vector for an image and the vector for its correct description are neighbors, even though images and text are fundamentally different kinds of data.

### Cross-Attention: Injecting the Prompt

With CLIP (or a similar text encoder), we can convert a text prompt into a sequence of vectors. Now we need to inject those vectors into the U-Net so they influence the denoising process.

The mechanism is **cross-attention**, borrowed from the transformer architecture. At various points in the U-Net, there are attention layers where the image features can "attend to" the text features. In plain language: the network can look at the prompt and decide which parts of the text are relevant to which parts of the image.

When the network is processing the upper-left region of the image, it might attend strongly to the word "sky" in the prompt. When processing the lower region, it might attend to "field" or "grass." The cross-attention mechanism is the bridge that connects what the text says to where it matters in the image.

```
Text: "a cat riding a bicycle through Tokyo at sunset"
       ↓ (text encoder)
[cat] [riding] [bicycle] [Tokyo] [sunset]    ← text vectors
  ↕       ↕        ↕        ↕       ↕        ← cross-attention
  ┌──────────────────────────────┐
  │         U-Net layers         │
  │  (processing the noisy image)│
  └──────────────────────────────┘
```

During training, the model sees image-caption pairs. The caption is encoded and fed in via cross-attention. The model learns to denoise conditioned on the text -- meaning it learns to remove noise in a way that's consistent with the description.

### Classifier-Free Guidance: The Amplifier

There's a practical problem with conditional generation: if you just train the model on image-caption pairs and then condition on a prompt at generation time, the results tend to be soft. The model follows the prompt, but loosely. "A cat on a bicycle" might give you an image that vaguely involves a cat and vaguely involves a bicycle, but doesn't nail either.

The fix is a technique called **classifier-free guidance**, and it's clever.

During training, some fraction of the time (say, 10%), the text prompt is replaced with an empty prompt. This teaches the model to generate both conditionally (with a prompt) and unconditionally (without one). The model becomes bilingual: it can generate "images in general" and "images matching this prompt."

At sampling time, for each denoising step, the model is run *twice*: once with the prompt (conditional prediction) and once without (unconditional prediction). Then the actual noise prediction is:

**ε_guided = ε_unconditional + w · (ε_conditional - ε_unconditional)**

where **w** is the guidance scale, typically between 1 and 20.

What does this do? The difference **(ε_conditional - ε_unconditional)** isolates the "prompt-following" component of the prediction -- the part that changes because of the text. Multiplying by **w** amplifies that component. A guidance scale of 1 means "follow the prompt normally." A scale of 7.5 (a common default) means "follow the prompt seven and a half times as hard as you naturally would."

High guidance makes images that match the prompt more precisely but at the cost of diversity and sometimes visual quality. It's a volume knob for "how literally should the model interpret the prompt."

```
guidance = 1                guidance = 7.5              guidance = 20
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│ A cat. Also │            │  A cat on a  │            │ EXTREMELY A  │
│ there may be │            │   bicycle.   │            │ CAT ON A     │
│ a bicycle   │            │  Looks right. │            │ BICYCLE.     │
│ somewhere.  │            │              │            │ OVERSATURATED│
└─────────────┘            └─────────────┘            └─────────────┘
  loose, diverse            accurate, balanced          precise, artifacts
```

---

## Part 5: Latent Diffusion -- The Stable Diffusion Breakthrough

### The Problem with Pixel Space

Everything described so far works. You can train a diffusion model on images, run the denoising process, and get good results. But there's a practical problem: it's agonizingly slow.

A 512×512 image has 786,432 pixel values. The U-Net has to process all of them at every single denoising step, and there are typically 50-1000 steps. That's a lot of computation. Early diffusion models required hours on expensive hardware to generate a single image.

The breakthrough came from a simple observation: **most of the pixels in an image are redundant.** A patch of blue sky doesn't need 10,000 individually specified pixel values. It's just... blue sky. The information content of a typical image is far lower than the raw pixel count would suggest.

### The Idea: Compress First, Then Diffuse

**Latent Diffusion Models** (LDMs), introduced in the paper "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022), solved this with a two-stage approach:

**Stage 1: Train an autoencoder** that compresses images into a much smaller "latent" representation and decompresses them back. A 512×512 image might compress to a 64×64 latent code -- 64 times fewer spatial values. The autoencoder (specifically, a VAE) learns to preserve the important structure while discarding pixel-level redundancy.

**Stage 2: Run the diffusion process in the latent space**, not pixel space. Add noise to the 64×64 latents. Train a U-Net to denoise the 64×64 latents. Generate images by denoising random 64×64 noise, then decode back to pixel space with the autoencoder.

```
Pixel-space diffusion:                  Latent diffusion:

[512×512 noise]                         [64×64 latent noise]
     ↓ denoise (slow!)                       ↓ denoise (fast!)
     ↓ 786,432 values/step                   ↓ 12,288 values/step
     ↓                                       ↓
[512×512 image]                         [64×64 clean latent] → VAE decoder → [512×512 image]
```

The speedup is enormous. The U-Net processes 64× fewer spatial values at each step. Combined with other optimizations, this brought generation time from hours down to seconds.

**Stable Diffusion** is a latent diffusion model. That's where the name comes from -- the diffusion process runs in the stable, compressed latent space rather than the chaotic pixel space. (The "stable" also refers to Stability AI, the company that funded its development, but the technical meaning is apt.)

### Why the Autoencoder Works

This might seem like a free lunch: compress the image, do the hard work in the compressed space, decompress. Where's the cost?

The cost is in the autoencoder itself. Training a good autoencoder requires a lot of computation and data. But here's the key: you only train it once. Once you have a good encoder/decoder pair, you can reuse it for any number of diffusion models. The diffusion model operates entirely in the latent space and doesn't need to know anything about pixels.

The autoencoder also introduces a small quality loss -- the decompressed image isn't pixel-identical to the original. But modern autoencoders are good enough that this loss is imperceptible. You get a 64× speedup in exchange for an invisible quality reduction. That's a trade anyone would take.

### The Connection to Watermarking

This two-stage architecture -- autoencoder plus diffusion -- is directly relevant to digital watermarking, and it's worth understanding why.

Some watermarking methods work by modifying the autoencoder itself. **Stable Signature**, for instance, fine-tunes the VAE decoder to embed a watermark during the decode step. Every image generated through that decoder carries the watermark automatically. The watermark lives in the decoder's weights, not in the prompt or the noise.

Other methods, like **Tree-Ring**, inject the watermark into the initial noise pattern -- placing a specific structure in the frequency domain of the starting latent. The diffusion process preserves this structure enough that it can be detected in the final image.

And **VideoSeal** takes a completely different approach: it applies a post-hoc neural encoder to the finished image, adding an imperceptible watermark pattern after generation is complete.

Each of these approaches targets a different stage of the pipeline:

```
Text prompt → Text encoder → Cross-attention → U-Net denoise → VAE decode → Image
                                                                   ↑            ↑
                                              Tree-Ring         Stable      VideoSeal
                                              (noise pattern)   Signature   (post-hoc)
                                                                (decoder)
```

Understanding where each watermark lives in the pipeline is essential for understanding how to attack it.

---

## Part 6: img2img -- Starting Halfway

### Not Starting from Pure Noise

Everything so far describes *text-to-image* generation: you start with pure noise and denoise all the way to a clean image. But there's another mode that turns out to be equally important: **img2img**, where you start with a real image.

The idea is simple. Instead of beginning the reverse process at step *T* (pure noise), you begin at some intermediate step *t* < *T*. How? You take a real image, add noise to it up to step *t*, and then denoise from there.

```
                         Standard generation (txt2img):
                         Start here → [pure noise] → denoise → denoise → ... → [image]
                                      step T

                         img2img:
[real image] → add noise → [partly noisy] → denoise → denoise → ... → [image]
                to step t    step t
```

The parameter that controls this is called **strength**, and it's a number between 0 and 1. A strength of 0 means "don't add any noise; return the original image unchanged." A strength of 1 means "add maximum noise; start from pure noise as if the original image wasn't there." A strength of 0.5 means "add noise halfway, then denoise from there."

### What Strength Controls

Low strength (0.1-0.3): The output looks very similar to the input. Small changes in texture, color, fine details, but the overall composition is preserved. Like applying a filter.

Medium strength (0.4-0.6): The output is recognizably related to the input but significantly different. Objects may shift, colors may change, new elements may appear. Like a reimagining.

High strength (0.7-1.0): The output may bear little resemblance to the input. Only the very coarse structure (overall color distribution, vague spatial layout) survives. Like generating a new image with only a vague influence from the old one.

This connects back to what we discussed about noise levels: low noise destroys fine details while preserving structure, high noise destroys everything. When you set the strength, you're choosing which level of information in the original image to preserve.

### The Palimpsest Analogy

A palimpsest is a medieval manuscript where the original text has been scraped off and new text written on top, but traces of the original show through. Some palimpsests have three or four layers of text, each partially visible through the ones above.

img2img works like writing on a palimpsest. The original image is the underlying text. The noise is the scraping. The denoising process is the new text being written. At low strength, the old text shows through clearly. At high strength, it's almost completely overwritten.

This analogy is more than poetic -- it captures something real about how information flows through the process. The original image's structure literally shows through the noise, and the denoising network writes new content on top of that structure while being influenced by what's underneath.

### Why This Matters for Watermarks

img2img is interesting for watermark research because of a fundamental tension:

**If the strength is too low**, the attacked image is nearly identical to the original, the watermark survives, and the attack fails.

**If the strength is too high**, the watermark is destroyed, but so is the image. The output looks nothing like the input. In any practical scenario, this is useless -- you need the image to still look like the same image.

**The attacker's sweet spot** is somewhere in the middle: enough noise to disrupt the watermark, but not so much that the image is unrecognizable. Whether such a sweet spot exists -- whether there's a strength where the watermark is gone but the image quality is acceptable -- depends on the specific watermark.

```
                    Watermark         Image
Strength            Detected?         Quality         Verdict
───────────────────────────────────────────────────────────
0.0 (no attack)     ✓ Yes             Perfect         Useless
0.1                 ✓ Yes             Very good       Useless
0.2                 ✓ Yes             Good            Useless
0.3                 ✓ Maybe           Moderate        Interesting
0.5                 ✗ No              Poor            Pyrrhic victory
0.7                 ✗ No              Very poor       Unrecognizable
1.0 (full regen)    ✗ No              Totally new     Not an "attack"
```

The table above is idealized. In practice, the transition from "detected" to "not detected" isn't sharp -- it depends on the watermarking method, the specific image, the detector threshold, and the attack parameters beyond just strength (guidance scale, number of denoising steps, the choice of diffusion model).

This is exactly what an attack sweep does: systematically search the space of (strength, guidance, steps) to find combinations that remove the watermark while maintaining acceptable quality.

### Other img2img Parameters

Strength isn't the only knob. The guidance scale and number of inference steps also matter.

**Guidance scale** in img2img controls how much the denoising process follows a text prompt versus the structural hints from the input image. With an empty prompt and low guidance, the model tries to reconstruct something like the original image. With a descriptive prompt and high guidance, the model pushes the output toward the text description while using the original image as a structural scaffold.

For watermark attacks, low guidance with an empty prompt is typical -- you want the output to look like the input, not to be "reimagined" according to some text.

**Number of inference steps** controls the granularity of the denoising process. More steps means finer control and usually better quality, but diminishing returns set in quickly. Going from 20 to 50 steps makes a noticeable difference. Going from 50 to 200 makes barely any difference but takes 4× longer.

---

## Part 7: The Mathematical Perspective

### Score Functions and Langevin Dynamics

If you want to understand diffusion models at one level deeper than "add noise, learn to denoise," the key concept is the **score function**.

The score function is the gradient of the log probability density:

**s(x) = ∇_x log p(x)**

In English: at any point **x** in image space, the score function tells you which direction to move to increase the probability of the image. It's a vector that points "uphill" in probability space.

If you had access to the true score function, you could generate images by a process called **Langevin dynamics**: start at a random point (noise), follow the score function (walk uphill), and add a little random noise at each step (to explore the space properly). After enough steps, you end up at a high-probability image.

The deep connection is this: **the noise-predicting network in a diffusion model is implicitly learning the score function.** When the network predicts what noise was added, it's estimating the direction from the noisy image toward cleaner (higher-probability) images. That direction is, up to a scaling factor, the score function.

This is why the technique works so well. The network doesn't need to learn the entire probability distribution of images (which is impossibly complex). It just needs to learn the *gradient* of that distribution -- which direction is "uphill" from any given point. That's a much simpler function, and a U-Net is well-suited to learn it.

### The Connection to Thermodynamics

There's a beautiful and non-accidental connection between diffusion models and statistical mechanics.

The forward process -- adding noise until the image is destroyed -- is analogous to the second law of thermodynamics. A structured, low-entropy state (the clean image) evolves toward a high-entropy state (random noise) through the accumulation of random perturbations. This happens spontaneously and irreversibly.

The reverse process -- removing noise to create structure -- is analogous to *reversing* the second law. Creating order from disorder. Building structure from chaos.

In thermodynamics, this is possible but requires the expenditure of energy and information. A refrigerator creates a cold region (low entropy) by expending energy and dumping heat elsewhere (increasing entropy even more somewhere else). Similarly, the diffusion model creates structured images (low entropy) by expending computation and using the learned score function (which encodes information about what images look like).

The score function is the diffusion model's equivalent of Maxwell's demon -- the hypothetical creature that could reverse the second law by having perfect information about every molecule. The neural network has (imperfect, learned) information about every pixel, and it uses that information to push the image from high entropy toward low entropy, step by step.

This isn't just an analogy. The mathematical frameworks are the same. The forward process is a stochastic differential equation. The reverse process is the time-reversed SDE, which requires the score function. The techniques for solving these equations come from the same body of mathematics used in statistical physics. The two fields have borrowed from each other throughout the development of diffusion models.

---

## Part 8: What Changed

### Why Diffusion Models Won

Between 2020 and 2024, diffusion models went from an academic curiosity to the dominant paradigm for image generation. Why?

**Training stability.** Unlike GANs, diffusion models don't have a minimax optimization problem. The training objective is simple regression (predict the noise), and it converges reliably. You don't need the dark arts of GAN training -- careful learning rate balancing, gradient penalties, progressive growing, or any of the other tricks that were required to make GANs work.

**Sample quality.** Diffusion models match or exceed GANs in image quality, and they do it more consistently. GANs can produce stunning individual images, but the distribution of samples is uneven -- some are brilliant, some are garbage. Diffusion models produce consistently high-quality samples across the distribution.

**Mode coverage.** GANs are prone to "mode collapse" -- generating only a few types of images and ignoring the rest of the distribution. If you train a GAN on animal photos, it might learn to generate great cats and dogs but completely forget about lizards. Diffusion models don't have this problem; they cover the full diversity of the training data.

**Mathematical foundations.** Diffusion models have clear connections to probability theory, stochastic processes, and thermodynamics. This makes them easier to analyze, improve, and extend. GAN theory, by contrast, is full of open problems and known gaps between theory and practice.

**Scalability.** Diffusion models scale well with data and compute. Bigger models, trained on more data, produce better images -- reliably, without training instabilities. This is the same kind of scaling behavior that made language models so successful, and it's a necessary property for any technology that's going to be deployed at scale.

### What They Made Possible

The practical consequences have been enormous:

**Democratized image creation.** Before 2022, creating a photorealistic image required either a camera, artistic skill, or expensive commercial software. After Stable Diffusion was released as open-source in August 2022, anyone with a GPU could generate photorealistic images from text descriptions. The implications for art, design, media, and misinformation are still playing out.

**Inpainting and editing.** Diffusion models can fill in masked regions of an image (inpainting) or modify specific aspects while preserving others. This is the foundation of many image editing tools, from removing unwanted objects to changing backgrounds.

**Video and 3D.** The same principles extend beyond still images. Video diffusion models generate short clips. 3D diffusion models generate meshes and textures. The framework is general enough that "add noise and learn to denoise" works across modalities.

**Scientific applications.** Diffusion models are being used for protein structure prediction, drug molecule design, weather forecasting, and other scientific tasks where you need to generate structured, high-dimensional data from learned distributions. The physics is different, but the math is the same.

### The Watermark Arms Race

For digital watermarking, diffusion models created both an opportunity and a threat.

The opportunity: watermarks can be embedded at various stages of the generation pipeline. Since the model *generates* the image, you can build the watermark into the generation process itself, making it harder to remove than a post-hoc watermark.

The threat: img2img provides a principled way to attack watermarks. Unlike ad-hoc image processing (blurring, cropping, JPEG compression), img2img *understands* the structure of natural images. It can reconstruct a plausible image after destroying most of the information in the original, including any watermark signal.

The question is whether the watermark can be made robust enough to survive this kind of intelligent reconstruction. The answer, at least for SynthID, appears to be: so far, yes.

---

## Part 9: Open Questions

### Why Do They Take So Many Steps?

One of the most active areas of research is reducing the number of denoising steps. The original DDPM required 1,000 steps. DDIM reduced this to 50. Consistency models and distillation techniques are pushing toward 1-4 steps.

The fundamental question: how much of the information is actually in the individual steps, and how much is in the overall trajectory? Can you skip 95% of the steps and still land in the right place? Recent work suggests yes, but with some quality trade-offs.

### What About Transformers?

The U-Net architecture that made diffusion models work is being replaced by transformers (the same architecture behind language models). DiT (Diffusion Transformers) treats image patches as tokens and processes them with standard transformer blocks instead of convolutional layers.

The advantage: transformers scale better with compute and data. The same scaling laws that made GPT-4 work apply here. The disadvantage: transformers are slower per-step for a given image size because they lack the built-in spatial inductive bias of convolutions.

This is the direction the field is moving. DALL-E 3, Stable Diffusion 3, and Flux all use transformer-based architectures rather than U-Nets.

### The Philosophical Angle

There's something genuinely interesting about the fact that creation and destruction are inverses of each other in this framework.

You can't directly learn to create from nothing -- the space of possibilities is too vast. But you *can* learn to undo damage, one tiny step at a time. And if you compose enough tiny restorations, what you get is creation.

It's Michelangelo's principle mechanized: "I saw the angel in the marble and carved until I set him free." Except the marble is noise, the carving is denoising, and the angel is whatever the training data taught the model to see.

There's also the thermodynamic interpretation: life itself is a process that creates order from disorder, using energy and information to push against entropy. Diffusion models are doing the same thing, in miniature, in pixel space. They're not alive, but they're performing the same fundamental operation that life performs -- using stored information (the trained weights) to build structure from chaos.

Whether this is a deep insight or a poetic overreach, I leave to you.

---

## A Summary: What You Now Know

Here's the chain of ideas, end to end:

1. **Generating images is hard** because the space of possible images is vast and the subspace of realistic images is vanishingly thin.

2. **The key insight**: destruction (adding noise) is easy and requires no learning. If you can learn to reverse each tiny step of destruction, you can create images by running destruction backwards.

3. **The forward process** adds Gaussian noise according to a schedule, gradually destroying the image over ~1,000 steps until it's pure static.

4. **The reverse process** uses a trained neural network (U-Net) to predict the noise at each step, then removes it. The training is simple regression: predict the noise that was added.

5. **Conditioning** via cross-attention lets text prompts guide the generation. Classifier-free guidance amplifies prompt-following by extrapolating the difference between conditional and unconditional predictions.

6. **Latent diffusion** runs the process in a compressed space (64×64 instead of 512×512), using a pre-trained autoencoder. This is the Stable Diffusion architecture.

7. **img2img** starts the reverse process from a partially-noised real image, not pure noise. The *strength* parameter controls how much of the original survives. This is the basis of diffusion-based watermark attacks.

8. **The deep mathematics** connects to score functions (gradient of log probability), Langevin dynamics, and thermodynamics. The neural network is implicitly learning which direction leads toward more probable images.

If someone asks you "How do diffusion models work?", you can say: "They learn to remove noise from images. You train a neural network by showing it clean photos with random noise added, and it learns to predict and remove that noise. Then, to generate a new image, you start with pure static and apply the denoiser a thousand times, each time removing a little noise. The result is a brand-new photorealistic image. It's like teaching someone to restore damaged paintings by showing them thousands of paintings at every stage of damage -- eventually they learn to create paintings from nothing but damage."

That's not a bad answer.

---

## Appendix: Going Deeper

### Key Papers

**"Denoising Diffusion Probabilistic Models" (Ho et al., 2020)**
The paper that made diffusion models practical. Showed that a simple noise-prediction objective with a U-Net produces excellent image quality. This is the DDPM paper, and it's where the modern era of diffusion models begins.

**"Denoising Diffusion Implicit Models" (Song et al., 2020)**
Introduced DDIM -- a way to reduce the number of denoising steps from 1,000 to as few as 50, by making the reverse process deterministic. Also introduced the ability to interpolate in latent space and perform img2img via partial noising.

**"Score-Based Generative Modeling through Stochastic Differential Equations" (Song et al., 2021)**
The theoretical unification paper. Shows that diffusion models are discretizations of continuous stochastic processes, connecting them to score matching and Langevin dynamics. This is where the thermodynamic perspective becomes explicit.

**"High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)**
The Stable Diffusion paper. Introduced the two-stage architecture (autoencoder + latent diffusion) that made diffusion models fast enough for practical use. The most impactful engineering contribution in the field.

**"Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)**
Introduced the technique that made text-to-image generation actually work well. Simple idea, enormous practical impact.

**"Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)**
The DiT paper. Replaced U-Nets with transformers, showing that diffusion models follow the same scaling laws as language models. This is the architectural direction the field is moving in.

### Glossary

| Term | What it means |
|------|--------------|
| **Diffusion** | The physical process of particles spreading out; here, the process of adding noise to an image step by step. |
| **Forward process** | Adding noise progressively until the image is pure static. Requires no learning. |
| **Reverse process** | Removing noise step by step, using a trained neural network. This is the generative process. |
| **Gaussian noise** | Random values from a normal (bell curve) distribution. The type of noise used in diffusion models. |
| **Noise schedule** | The plan for how much noise to add at each step. Typically gentle at first, aggressive later. |
| **Score function** | The gradient of the log probability density. Points toward more probable (more realistic) images. |
| **U-Net** | The neural network architecture that predicts noise. Named for its U-shape: encoder → bottleneck → decoder with skip connections. |
| **DDPM** | Denoising Diffusion Probabilistic Models. The foundational paper by Ho et al. (2020). |
| **DDIM** | Denoising Diffusion Implicit Models. A faster sampling method (50 steps instead of 1,000). |
| **Latent diffusion** | Running the diffusion process in a compressed representation (latent space) rather than pixel space. |
| **VAE** | Variational Autoencoder. The image compressor/decompressor in latent diffusion models. |
| **CLIP** | Contrastive Language-Image Pre-training. Creates a shared vector space for text and images. |
| **Cross-attention** | The mechanism by which text features influence the denoising process. |
| **Classifier-free guidance** | Amplifying the prompt-following behavior by extrapolating between conditional and unconditional predictions. |
| **Guidance scale** | The "volume knob" for classifier-free guidance. Higher = more prompt-adherent but less diverse. |
| **img2img** | Starting the reverse process from a partially-noised real image, not pure noise. |
| **Strength** | In img2img, how much noise to add to the input image (0 = no change, 1 = pure noise). |
| **Inpainting** | Generating content for a masked region of an image. A special case of conditional generation. |
| **DiT** | Diffusion Transformers. Replacing the U-Net with a transformer architecture. The current trend. |
| **Langevin dynamics** | A sampling method that follows the score function with added noise. The theoretical basis for diffusion sampling. |
| **Inference steps** | The number of denoising steps used during generation. More = better quality, diminishing returns. |
| **SSIM** | Structural Similarity Index. A metric for how similar two images are. Used to measure attack quality. |
| **PSNR** | Peak Signal-to-Noise Ratio. Another image similarity metric, measured in decibels. |

### If You Read One Thing Next

If this guide made you curious and you want to go one step deeper, read the Lilian Weng blog post "What are Diffusion Models?" (lilianweng.github.io, 2021). It's the single best technical introduction available online -- it covers the math without drowning you, includes clear diagrams, and connects all the pieces. You'll recognize every concept from this guide and see the precise equations behind them.

For the watermarking connection specifically, the Stable Signature paper (Fernandez et al., 2023) is worth reading. It shows how the VAE decoder in a latent diffusion model can be fine-tuned to embed an invisible watermark -- and it's short enough to read in one sitting.

---

*This guide was written for someone who uses AI image tools every day and wants to understand the mechanism, not just the output. The field moves fast -- by the time you read this, there will be new architectures, new sampling methods, and new applications. But the core ideas -- noise, denoising, score functions, and the reversal of entropy -- those are the foundation everything is built on.*
