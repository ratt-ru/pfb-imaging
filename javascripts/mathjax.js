window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      // Common mathematical notation
      R: "\\mathbb{R}",
      C: "\\mathbb{C}",
      N: "\\mathbb{N}",
      Z: "\\mathbb{Z}",
      
      // Operators
      argmin: "\\operatorname{argmin}",
      argmax: "\\operatorname{argmax}",
      prox: "\\operatorname{prox}",
      
      // Norms
      norm: ["\\left\\|#1\\right\\|", 1],
      
      // Radio astronomy specific
      vis: "\\mathbf{v}",
      image: "\\mathbf{x}",
      psf: "\\mathbf{P}",
      beam: "\\mathbf{B}",
      
      // Matrices and vectors
      mat: ["\\mathbf{#1}", 1],
      vec: ["\\mathbf{#1}", 1],
      
      // Fourier transform
      ft: "\\mathcal{F}",
      ift: "\\mathcal{F}^{-1}",
      
      // Wavelet transforms
      wt: "\\mathcal{W}",
      iwt: "\\mathcal{W}^{-1}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};