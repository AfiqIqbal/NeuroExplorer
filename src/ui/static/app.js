// Cache DOM elements
const statusEl = document.getElementById('status');
const searchResultsEl = document.getElementById('searchResults');
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const resetButton = document.getElementById('resetButton');
const agentFilter = document.getElementById('agentFilter');
const typeFilter = document.getElementById('typeFilter');
const includeIgnoredCheckbox = document.getElementById('includeIgnored');
const template = document.getElementById('file-card-template');
const heroSphere = document.getElementById('heroSphere');

// Check for required elements
if (!searchResultsEl || !searchInput || !searchButton || !resetButton || 
    !agentFilter || !typeFilter || !includeIgnoredCheckbox) {
  console.error('Required elements not found in the DOM');
}

const API_BASE = '';

function setStatus(message, level = 'info') {
  statusEl.textContent = message;
  statusEl.dataset.level = level;
}

async function fetchJSON(path, params = {}) {
  const url = new URL(path, window.location.origin + API_BASE);
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    if (typeof value === 'boolean') {
      url.searchParams.set(key, value ? 'true' : 'false');
    } else {
      url.searchParams.set(key, value);
    }
  });

  const response = await fetch(url.toString());
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return response.json();
}

function createPreview(item) {
  const container = document.createElement('div');
  container.className = 'preview-content';

  const type = (item.file_type || '').toLowerCase();
  let previewType = 'generic';

  if (type.match(/\.(png|jpe?g|gif|webp|bmp|svg)$/)) {
    previewType = 'image';
  } else if (type.match(/\.(mp3|wav|flac|ogg|m4a)$/)) {
    previewType = 'audio';
  } else if (type.match(/\.(py|js|ts|tsx|java|cs|cpp|c|go|rs|rb|json|yaml|yml)$/)) {
    previewType = 'code';
  }

  container.dataset.previewType = previewType;

  if (previewType === 'code') {
    const pre = document.createElement('pre');
    pre.textContent = item.summary?.slice(0, 220) || 'Code summary unavailable.';
    container.appendChild(pre);
  } else {
    const span = document.createElement('span');
    if (previewType === 'image') {
      span.textContent = 'Image asset';
    } else if (previewType === 'audio') {
      span.textContent = 'Audio asset';
    } else {
      span.textContent = item.summary ? item.summary.slice(0, 120) : 'No preview available.';
    }
    container.appendChild(span);
  }

  return container;
}

function setLoading(isLoading) {
  const surface = document.querySelector('.search-surface');
  surface?.classList.toggle('loading', Boolean(isLoading));
}

function setTyping(isTyping) {
  const surface = document.querySelector('.search-surface');
  surface?.classList.toggle('typing', Boolean(isTyping));
  heroSphere?.classList.toggle('typing', Boolean(isTyping));
}

function triggerSphereImpulse() {
  if (!heroSphere) return;
  heroSphere.classList.remove('typing-impulse');
  // Force reflow for restart animation
  void heroSphere.offsetWidth;
  heroSphere.classList.add('typing-impulse');
}

function renderCards(items, showScores = false) {
  if (!searchResultsEl) {
    console.error('Search results container not found');
    return;
  }
  
  searchResultsEl.innerHTML = '';
  
  if (!items || !Array.isArray(items)) {
    console.error('Invalid items array provided to renderCards');
    return;
  }
  
  if (items.length === 0) {
    searchResultsEl.classList.remove('open');
    const empty = document.createElement('div');
    empty.className = 'search-result-empty';
    empty.textContent = 'No records found.';
    searchResultsEl.appendChild(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const item of items) {
    const row = document.createElement('article');
    row.className = 'search-result-row';

    const header = document.createElement('header');
    const name = document.createElement('span');
    name.textContent = item.file_name;
    header.appendChild(name);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = `
      <span>${item.file_type?.replace('.', '').toUpperCase() || 'FILE'}</span>
      <span>${item.agent || 'Unknown agent'}</span>
      ${showScores && typeof item.score === 'number' ? `<span>Score ${(item.score).toFixed(4)}</span>` : ''}
    `;

    const summary = document.createElement('p');
    summary.className = 'summary';
    summary.textContent = item.summary || 'No summary available.';

    const path = document.createElement('p');
    path.className = 'summary';
    path.textContent = item.file_path;

    const tagWrapper = document.createElement('div');
    tagWrapper.className = 'meta';
    if (item.tags && item.tags.length) {
      for (const tag of item.tags) {
        const chip = document.createElement('span');
        chip.textContent = tag;
        tagWrapper.appendChild(chip);
      }
    }

    row.appendChild(header);
    row.appendChild(meta);
    row.appendChild(summary);
    row.appendChild(path);
    if (tagWrapper.childElementCount) {
      row.appendChild(tagWrapper);
    }

    row.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(item.file_path);
        setStatus(`Copied path to clipboard.`, 'success');
      } catch (error) {
        setStatus(`Failed to copy path: ${error.message}`, 'error');
      }
    });

    fragment.appendChild(row);
  }
  searchResultsEl.appendChild(fragment);
  searchResultsEl.classList.add('open');
  setStatus(`Showing ${items.length} result${items.length === 1 ? '' : 's'}.`, 'success');
}

async function loadMeta() {
  try {
    const meta = await fetchJSON('/api/meta');

    agentFilter.innerHTML = '<option value="">All</option>';
    for (const agent of meta.agents) {
      const option = document.createElement('option');
      option.value = agent;
      option.textContent = agent;
      agentFilter.appendChild(option);
    }

    typeFilter.innerHTML = '<option value="">All</option>';
    for (const type of meta.file_types) {
      const option = document.createElement('option');
      option.value = type;
      option.textContent = type;
      typeFilter.appendChild(option);
    }

    setStatus(`Loaded metadata. ${meta.total} records indexed.`, 'info');
  } catch (error) {
    console.error(error);
    setStatus(`Failed to load metadata: ${error.message}`, 'error');
  }
}

async function loadFiles() {
  const params = {
    agent: agentFilter.value,
    file_type: typeFilter.value,
    include_ignored: includeIgnoredCheckbox.checked,
  };

  try {
    setLoading(true);
    setStatus('Loading files...', 'muted');
    const files = await fetchJSON('/api/files', params);
    renderCards(files, false);
  } catch (error) {
    console.error(error);
    setStatus(`Failed to load files: ${error.message}`, 'error');
  } finally {
    setLoading(false);
  }
}

async function performSearch() {
  const query = searchInput.value.trim();
  if (!query) {
    if (searchResultsEl) {
      searchResultsEl.innerHTML = '';
      searchResultsEl.classList.remove('open');
    }
    setStatus('Enter a search to begin.', 'muted');
    return;
  }

  const params = {
    query,
    agent: agentFilter.value,
    file_type: typeFilter.value,
    include_ignored: includeIgnoredCheckbox.checked,
  };

  try {
    setLoading(true);
    setStatus('Performing semantic search...', 'muted');
    const results = await fetchJSON('/api/search', params);
    renderCards(results, true);
  } catch (error) {
    console.error(error);
    setStatus(`Search failed: ${error.message}`, 'error');
  } finally {
    setLoading(false);
  }
}

function attachEvents() {
  searchButton.addEventListener('click', performSearch);
  resetButton.addEventListener('click', () => {
    searchInput.value = '';
    agentFilter.value = '';
    typeFilter.value = '';
    includeIgnoredCheckbox.checked = false;
    if (searchResultsEl) {
      searchResultsEl.innerHTML = '';
      searchResultsEl.classList.remove('open');
    }
    setStatus('Ready when you are.', 'muted');
  });

  searchInput.addEventListener('keydown', (event) => {
    setTyping(true);
    triggerSphereImpulse();
    if (event.key === 'Enter') {
      performSearch();
    }
  });

  searchInput.addEventListener('keyup', () => {
    const active = searchInput.value.trim().length > 0;
    setTyping(active);
    if (active) {
      triggerSphereImpulse();
    }
  });

  searchInput.addEventListener('blur', () => setTyping(false));

  agentFilter.addEventListener('change', loadFiles);
  typeFilter.addEventListener('change', loadFiles);
  includeIgnoredCheckbox.addEventListener('change', loadFiles);
}

async function init() {
  attachEvents();
  await loadMeta();
  setStatus('Ready when you are.', 'muted');
}

init();
