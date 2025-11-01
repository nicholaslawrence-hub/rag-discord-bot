
// Theme toggle function
function toggleTheme() {
    const body = document.body;
    body.classList.toggle('dark-mode');
    
    // Save preference to localStorage
    const darkMode = body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', darkMode);
    
    // Update charts if they exist
    if (typeof updateChartsTheme === 'function') {
        updateChartsTheme();
    }
}

// Apply saved theme preference on load
document.addEventListener('DOMContentLoaded', () => {
    const darkMode = localStorage.getItem('darkMode') === 'true';
    if (darkMode) {
        document.body.classList.add('dark-mode');
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.checked = true;
        }
    }
});
    