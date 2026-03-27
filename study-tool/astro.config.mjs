import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://raggedr.github.io',
  base: '/university_courses',
  integrations: [
    starlight({
      title: 'Uni Notes',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/RaggedR/university_courses' },
      ],
      sidebar: [
        {
          label: 'Examples',
          autogenerate: { directory: 'examples' },
        },
        {
          label: 'Sparse Autoencoders',
          autogenerate: { directory: 'sparse-autoencoders' },
        },
        {
          label: 'Diffusion Models',
          autogenerate: { directory: 'diffusion-models' },
        },
      ],
      customCss: [
        'katex/dist/katex.min.css',
        './src/styles/components.css',
      ],
    }),
    react(),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
