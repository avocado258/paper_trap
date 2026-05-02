/* ============================================================
   PaperTrap — Shared JavaScript (app.js)
   ============================================================ */

'use strict';

// ── Page Transition ──────────────────────────────────────────
const PageTransition = (() => {
  const overlay = document.createElement('div');
  overlay.className = 'page-transition';
  document.body.appendChild(overlay);

  function navigateTo(url) {
    overlay.classList.add('active');
    setTimeout(() => { window.location.href = url; }, 350);
  }

  function init() {
    // Fade in on load
    window.addEventListener('load', () => {
      setTimeout(() => overlay.classList.remove('active'), 50);
    });

    document.querySelectorAll('a[data-transition]').forEach(link => {
      link.addEventListener('click', e => {
        const href = link.getAttribute('href');
        if (href && !href.startsWith('#') && !href.startsWith('http')) {
          e.preventDefault();
          navigateTo(href);
        }
      });
    });
  }

  return { init, navigateTo };
})();

// ── Navbar ────────────────────────────────────────────────────
const Navbar = (() => {
  function init() {
    const navbar = document.querySelector('.navbar');
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');

    if (!navbar) return;

    // Scroll shadow
    const onScroll = () => {
      navbar.classList.toggle('scrolled', window.scrollY > 20);
    };
    window.addEventListener('scroll', onScroll, { passive: true });

    // Mobile menu
    hamburger?.addEventListener('click', () => {
      navLinks?.classList.toggle('open');
      hamburger.classList.toggle('active');
    });

    // Close mobile menu on link click
    navLinks?.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        navLinks.classList.remove('open');
        hamburger?.classList.remove('active');
      });
    });

    // Mark active link
    const current = window.location.pathname.split('/').pop() || 'index.html';
    navLinks?.querySelectorAll('a').forEach(a => {
      const href = a.getAttribute('href');
      if (href === current || (current === '' && href === 'index.html')) {
        a.classList.add('active');
      }
    });
  }

  return { init };
})();

// ── Intersection Observer (Reveal on scroll) ──────────────────
const ScrollReveal = (() => {
  function init() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.reveal').forEach(el => {
      observer.observe(el);
    });
  }

  return { init };
})();

// ── Animated Counter ──────────────────────────────────────────
const Counter = (() => {
  function animateValue(el, from, to, duration = 1600, suffix = '') {
    const start = performance.now();
    const isFloat = String(to).includes('.');
    const decimals = isFloat ? (String(to).split('.')[1] || '').length : 0;

    const step = (timestamp) => {
      const elapsed = timestamp - start;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 4);
      const value = from + (to - from) * ease;
      el.textContent = (isFloat ? value.toFixed(decimals) : Math.floor(value)) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function init() {
    const counters = document.querySelectorAll('[data-count]');
    if (!counters.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const to = parseFloat(el.dataset.count);
          const suffix = el.dataset.suffix || '';
          animateValue(el, 0, to, 1600, suffix);
          observer.unobserve(el);
        }
      });
    }, { threshold: 0.5 });

    counters.forEach(c => observer.observe(c));
  }

  return { init, animateValue };
})();

// ── Progress Bar Animator ─────────────────────────────────────
const ProgressAnimator = (() => {
  function init() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const fill = entry.target.querySelector('.progress-fill');
          if (fill) {
            const target = fill.dataset.width || '0%';
            setTimeout(() => { fill.style.width = target; }, 200);
          }
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.4 });

    document.querySelectorAll('.progress-bar').forEach(bar => observer.observe(bar));
  }

  return { init };
})();

// ── Toast Notifications ───────────────────────────────────────
const Toast = (() => {
  let container;

  function getContainer() {
    if (!container) {
      container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    return container;
  }

  function show(message, type = 'info', duration = 4000) {
    const c = getContainer();
    const toast = document.createElement('div');
    const icons = { info: '◈', success: '✓', error: '✕', warning: '⚠' };
    const colors = {
      info:    'var(--accent-cyan)',
      success: 'var(--accent-green)',
      error:   'var(--accent-red)',
      warning: 'var(--accent-amber)'
    };
    toast.className = 'toast';
    toast.style.borderLeftColor = colors[type];
    toast.style.borderLeft = `3px solid ${colors[type]}`;
    toast.innerHTML = `
      <span style="color:${colors[type]};margin-right:8px;font-family:var(--font-mono)">${icons[type]}</span>
      ${message}
    `;
    c.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(24px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return { show };
})();

// ── Typewriter ────────────────────────────────────────────────
const Typewriter = (() => {
  function write(el, text, speed = 40, delay = 0) {
    el.textContent = '';
    el.style.borderRight = '2px solid var(--accent-cyan)';
    let i = 0;
    setTimeout(() => {
      const interval = setInterval(() => {
        el.textContent += text[i];
        i++;
        if (i >= text.length) {
          clearInterval(interval);
          setTimeout(() => { el.style.borderRight = 'none'; }, 800);
        }
      }, speed);
    }, delay);
  }

  return { write };
})();

// ── Particle Background ───────────────────────────────────────
const Particles = (() => {
  function init(canvasId, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const opts = {
      count: options.count || 60,
      color: options.color || 'rgba(0, 229, 255, 0.4)',
      size: options.size || 1.5,
      speed: options.speed || 0.3,
      connect: options.connect !== false,
      connectDist: options.connectDist || 120,
    };

    let w, h, particles;

    function resize() {
      w = canvas.width = canvas.parentElement.offsetWidth;
      h = canvas.height = canvas.parentElement.offsetHeight;
    }

    function createParticles() {
      particles = Array.from({ length: opts.count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * opts.speed,
        vy: (Math.random() - 0.5) * opts.speed,
        r: Math.random() * opts.size + 0.5,
      }));
    }

    function draw() {
      ctx.clearRect(0, 0, w, h);
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = opts.color;
        ctx.fill();
      });

      if (opts.connect) {
        for (let i = 0; i < particles.length; i++) {
          for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < opts.connectDist) {
              ctx.beginPath();
              ctx.moveTo(particles[i].x, particles[i].y);
              ctx.lineTo(particles[j].x, particles[j].y);
              ctx.strokeStyle = `rgba(0, 229, 255, ${0.08 * (1 - dist / opts.connectDist)})`;
              ctx.lineWidth = 0.5;
              ctx.stroke();
            }
          }
        }
      }
      requestAnimationFrame(draw);
    }

    resize();
    createParticles();
    draw();
    window.addEventListener('resize', () => { resize(); createParticles(); });
  }

  return { init };
})();

// ── File Drag & Drop ──────────────────────────────────────────
const FileDropzone = (() => {
  function init(zoneId, onFileSelected) {
    const zone = document.getElementById(zoneId);
    if (!zone) return;
    const input = zone.querySelector('input[type="file"]');

    ['dragenter', 'dragover'].forEach(ev => {
      zone.addEventListener(ev, e => {
        e.preventDefault();
        zone.classList.add('drag-over');
      });
    });
    ['dragleave', 'drop'].forEach(ev => {
      zone.addEventListener(ev, e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (ev === 'drop') {
          const file = e.dataTransfer.files[0];
          if (file) onFileSelected(file);
        }
      });
    });
    zone.addEventListener('click', () => input?.click());
    input?.addEventListener('change', () => {
      if (input.files[0]) onFileSelected(input.files[0]);
    });
  }

  return { init };
})();

// ── Utility ───────────────────────────────────────────────────
const Utils = {
  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  },
  truncate(str, n) {
    return str.length > n ? str.substring(0, n) + '…' : str;
  },
  randomBetween(a, b) {
    return Math.random() * (b - a) + a;
  }
};

// ── DOM Ready Bootstrap ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  PageTransition.init();
  Navbar.init();
  ScrollReveal.init();
  Counter.init();
  ProgressAnimator.init();
});

// Expose globally
window.PaperTrap = { PageTransition, Navbar, Toast, Typewriter, Particles, FileDropzone, Counter, Utils };