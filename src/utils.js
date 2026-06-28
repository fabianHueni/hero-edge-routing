// helpers: timing and small utilities
export function nowMs() {
    return performance.now();
}

export function sleep(ms) {
    return new Promise(res => setTimeout(res, ms));
}


export function measureAsync(fn) {
    const start = nowMs();
    return Promise.resolve()
        .then(() => fn())
        .then(res => ({res, ms: nowMs() - start}));
}


/**
 * Log text to a given HTML element with timestamp to show the log in the UI
 *
 * @param el - HTML element to log to
 * @param evt - Event object with job, route, latency, response, and timing metrics
 */
export function logTo(el, evt) {
    console.log(evt.job)
    if (!el) return;
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${evt.job.id}</td>
        <td>${new Date().toLocaleTimeString()}</td>
        <td>${evt.route}</td>
        <td>${evt.totalLatency?.toFixed(2) || evt.latency?.toFixed(2) || 0}ms</td>
        <td>${evt.queueingTime?.toFixed(2) || 0}ms</td>
        <td>${evt.inferenceTime?.toFixed(2) || evt.latency?.toFixed(2) || 0}ms</td>
        <td title="${escapeHtml(evt.job.prompt)}">${escapeHtml(evt.job.prompt.substring(0, 30))}...</td>
        <td title="${evt.response || ''}">${(evt.response || '').substring(0, 30)}</td>
        <td>${evt.evalRes.exactMatch}</td>
    `;
    el.appendChild(row);
    el.scrollTop = el.scrollHeight;
}

/**
 * Escapes HTML special characters in a string to prevent HTML injection
 *
 * @param str - Input string
 * @returns {string} - Escaped string
 */
function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (char) => {
        const escapeMap = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;',
        };
        return escapeMap[char];
    });
}

/**
 * Approximates the number of words in a given text string
 *
 * @param text - Input text string
 * @returns {number} - Approximate number of words
 */
export function getNumberOfWords(text) {
    return text.trim().split(/\s+/).length;
}
