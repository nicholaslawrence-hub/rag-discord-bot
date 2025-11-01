
function copyTemplate() {
    const templateText = document.getElementById('template-json').textContent;
    navigator.clipboard.writeText(templateText).then(() => {
        const alert = document.getElementById('copy-alert');
        alert.style.display = 'block';
        setTimeout(() => {
            alert.style.display = 'none';
        }, 3000);
    });
}
    