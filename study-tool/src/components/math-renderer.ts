import katex from 'katex';

/**
 * Renders LaTeX math expressions in a string.
 * Supports $...$ for inline and $$...$$ for display math.
 */
export function renderMath(text: string): string {
  // Display math first (greedy within $$...$$)
  let result = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    try {
      return katex.renderToString(math.trim(), { displayMode: true, throwOnError: false });
    } catch {
      return `$$${math}$$`;
    }
  });

  // Inline math
  result = result.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
    try {
      return katex.renderToString(math.trim(), { displayMode: false, throwOnError: false });
    } catch {
      return `$${math}$`;
    }
  });

  return result;
}
